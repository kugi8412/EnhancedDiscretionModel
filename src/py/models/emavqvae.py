import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

# ==========================================
# 1. HELPER BLOCKS (LegNet & VQ)
# ==========================================

class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(EMAVectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, inputs):
        # inputs: [B, C, L] -> [B, L, C]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_inputs = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                     + torch.sum(self.embed**2, dim=1) 
                     - 2 * torch.matmul(flat_inputs, self.embed.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embed).view(input_shape)

        if self.training:
            cluster_size = torch.sum(encodings, dim=0)
            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            embed_sum = torch.matmul(encodings.t(), flat_inputs)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = torch.sum(self.cluster_size.data)
            self.cluster_size.data = (
                (self.cluster_size.data + self.epsilon) 
                / (n + self.num_embeddings * self.epsilon) * n
            )
            embed_normalized = self.embed_avg.data / self.cluster_size.data.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight-Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[1])

class SELayer(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(inp, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), inp),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y

class EffBlock(nn.Module):
    def __init__(self, in_ch, ks=5, resize_factor=4, activation=nn.SiLU):
        super().__init__()
        inner_dim = in_ch * resize_factor
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, inner_dim, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(inner_dim),
            activation(),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=ks, groups=inner_dim, padding='same', bias=False),
            nn.BatchNorm1d(inner_dim),
            activation(),
            SELayer(inner_dim, reduction=resize_factor),
            nn.Conv1d(inner_dim, in_ch, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(in_ch),
            activation(),
        )
    def forward(self, x):
        return x + self.block(x) # Built-in residual connection

# ==========================================
# 2. MAIN ARCHITECTURE (LegNet-VQ-VAE)
# ==========================================

@register_model("LegNet_VQVAE")
class LegNet_VQVAE(nn.Module):
    def __init__(self, in_ch=4, stem_ch=128, vq_dim=64, num_embeddings=512, **kwargs):
        super().__init__()
        
        self.activation = nn.SiLU()

        # --- ENCODER ---
        # Spatial compression (reduce sequence length)
        self.encoder_stem = nn.Sequential(
            nn.Conv1d(in_ch, stem_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(stem_ch),
            self.activation
        )
        self.encoder_blocks = nn.Sequential(
            EffBlock(stem_ch),
            nn.MaxPool1d(2), # L/2
            EffBlock(stem_ch),
            nn.MaxPool1d(2), # L/4
            EffBlock(stem_ch),
            nn.MaxPool1d(2)  # L/8
        )
        self.pre_vq_conv = nn.Conv1d(stem_ch, vq_dim, kernel_size=1)

        # --- VECTOR QUANTIZER (VQ-EMA) ---
        self.vq_layer = EMAVectorQuantizer(
            num_embeddings=num_embeddings, 
            embedding_dim=vq_dim, 
            commitment_cost=0.25
        )

        # --- PREDICTION HEAD (Expression) ---
        self.predictor = nn.Sequential(
            nn.Conv1d(vq_dim, stem_ch, kernel_size=1),
            nn.BatchNorm1d(stem_ch),
            self.activation,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(stem_ch, stem_ch),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(stem_ch, 2)  # Dev, Hk
        )

        # --- DECODER HEAD (Sequence reconstruction) ---
        self.post_vq_conv = nn.Conv1d(vq_dim, stem_ch, kernel_size=1)
        # ConvTranspose1d for upsampling
        self.decoder_blocks = nn.Sequential(
            nn.ConvTranspose1d(stem_ch, stem_ch, kernel_size=4, stride=2, padding=1), # L*2
            EffBlock(stem_ch),
            nn.ConvTranspose1d(stem_ch, stem_ch, kernel_size=4, stride=2, padding=1), # L*4
            EffBlock(stem_ch),
            nn.ConvTranspose1d(stem_ch, stem_ch, kernel_size=4, stride=2, padding=1), # L*8
            EffBlock(stem_ch)
        )
        # Reconstruct to original 4 channels (one-hot)
        self.decoder_out = nn.Conv1d(stem_ch, in_ch, kernel_size=5, padding=2)

    def forward(self, x):
        original_length = x.size(2)

        # 1. Encode
        e = self.encoder_stem(x)
        e = self.encoder_blocks(e)
        e = self.pre_vq_conv(e)

        # 2. Quantize (latent space)
        quantized, vq_loss, _ = self.vq_layer(e)

        # 3. Predict expression from quantized latent
        preds = self.predictor(quantized)
        
        # 4. Decode (reconstruct)
        d = self.post_vq_conv(quantized)
        d = self.decoder_blocks(d)
        x_recon = self.decoder_out(d)
        
        # Dynamic length alignment (e.g. 248 -> 249 bp due to division remainder)
        if x_recon.size(2) != original_length:
            x_recon = F.interpolate(x_recon, size=original_length, mode='linear', align_corners=False)

        # Return all elements needed for loss computation
        return preds[:, 0:1], preds[:, 1:2], x_recon, vq_loss
