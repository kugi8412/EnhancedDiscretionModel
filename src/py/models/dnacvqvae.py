import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
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
            self.cluster_size.data = (self.cluster_size.data + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            self.embed.data.copy_(self.embed_avg.data / self.cluster_size.data.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[1])

class SELayer(nn.Module):
    def __init__(self, inp, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(inp, int(inp // reduction)), nn.SiLU(), nn.Linear(int(inp // reduction), inp), nn.Sigmoid())
    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y

class EffBlock(nn.Module):
    def __init__(self, in_ch, ks=5, resize_factor=4):
        super().__init__()
        inner_dim = in_ch * resize_factor
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, inner_dim, kernel_size=1, padding='same', bias=False), nn.BatchNorm1d(inner_dim), nn.SiLU(),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=ks, groups=inner_dim, padding='same', bias=False), nn.BatchNorm1d(inner_dim), nn.SiLU(),
            SELayer(inner_dim, reduction=resize_factor),
            nn.Conv1d(inner_dim, in_ch, kernel_size=1, padding='same', bias=False), nn.BatchNorm1d(in_ch), nn.SiLU(),
        )
    def forward(self, x):
        return x + self.block(x)


@register_model("HydraDNA_cVQVAE")
class HydraDNA_cVQVAE(nn.Module):
    def __init__(self, in_ch=4, stem_ch=128, gru_dim=128, vq_dim=64, num_embeddings=2048, depth=3, **kwargs):
        super().__init__()
        
        # Encoder
        self.cnn_stem = nn.Sequential(nn.Conv1d(in_ch, stem_ch, kernel_size=7, padding=3, bias=False), nn.BatchNorm1d(stem_ch), nn.SiLU())
        
        enc_layers = []
        for _ in range(depth):
            enc_layers.extend([EffBlock(stem_ch), nn.MaxPool1d(2)])
        self.cnn_blocks = nn.Sequential(*enc_layers)
        
        self.encoder_gru = nn.GRU(stem_ch, gru_dim, batch_first=True, bidirectional=True)
        self.pre_vq_conv = nn.Conv1d(gru_dim * 2, vq_dim, kernel_size=1)

        # VQ-VAE
        self.vq_layer = EMAVectorQuantizer(num_embeddings, vq_dim, commitment_cost=0.25)

        # FiLM Generator
        self.film_generator = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, vq_dim * 2)
        )

        # Decoder
        self.decoder_cond_proj = nn.Conv1d(vq_dim, gru_dim * 2, kernel_size=1)
        self.decoder_gru = nn.GRU(gru_dim * 2, gru_dim, batch_first=True, bidirectional=True)
        
        dec_layers = []
        in_channels_dec = gru_dim * 2
        for _ in range(depth):
            dec_layers.extend([
                nn.ConvTranspose1d(in_channels_dec, stem_ch, kernel_size=4, stride=2, padding=1),
                EffBlock(stem_ch)
            ])
            in_channels_dec = stem_ch
            
        self.decoder_blocks = nn.Sequential(*dec_layers)
        self.decoder_out = nn.Conv1d(stem_ch, in_ch * 2, kernel_size=5, padding=2)

    def encode_strand(self, x):
        x = self.cnn_stem(x)
        x = self.cnn_blocks(x)
        x = x.permute(0, 2, 1)
        x, _ = self.encoder_gru(x)
        x = x.permute(0, 2, 1)
        return self.pre_vq_conv(x)

    def forward(self, x, y_dev=None, y_hk=None, tau=1.0):
        original_length = x.size(2)
        batch_size = x.size(0)

        z = self.encode_strand(x)
        quantized, vq_loss, _ = self.vq_layer(z)

        # FiLM Conditioning
        if y_dev is not None and y_hk is not None:
            smooth_dev = torch.round(y_dev * 10.0) / 10.0
            smooth_hk = torch.round(y_hk * 10.0) / 10.0
            cond_vector = torch.cat([smooth_dev.view(-1, 1), smooth_hk.view(-1, 1)], dim=1)
        else:
            cond_vector = torch.zeros(batch_size, 2, device=x.device)

        if self.training and torch.rand(1).item() < 0.20:
            cond_vector = torch.zeros_like(cond_vector)
        if y_dev is None and not self.training:
            cond_vector = torch.zeros_like(cond_vector)

        film_params = self.film_generator(cond_vector)
        gamma = film_params[:, :quantized.size(1)].unsqueeze(2)
        beta = film_params[:, quantized.size(1):].unsqueeze(2)

        cond_quantized = (1.0 + gamma) * quantized + beta

        # Decoding
        d = self.decoder_cond_proj(cond_quantized)
        d = d.permute(0, 2, 1)
        d, _ = self.decoder_gru(d)
        d = d.permute(0, 2, 1)
        
        d = self.decoder_blocks(d)
        x_logits_8ch = self.decoder_out(d)
        
        if x_logits_8ch.size(2) != original_length:
            x_logits_8ch = F.interpolate(x_logits_8ch, size=original_length, mode='linear', align_corners=False)

        x_fwd_logits = x_logits_8ch[:, 0:4, :]
        x_rc_logits  = x_logits_8ch[:, 4:8, :]
        
        x_gumbel_fwd = F.gumbel_softmax(x_fwd_logits, tau=tau, hard=True, dim=1)
        x_gumbel_rc  = F.gumbel_softmax(x_rc_logits, tau=tau, hard=True, dim=1)

        return x_logits_8ch, (x_gumbel_fwd, x_gumbel_rc), vq_loss
