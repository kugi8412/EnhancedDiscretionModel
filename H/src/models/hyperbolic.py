from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Lorentz manifold primitives
# --------------------------------------------------------------------------

def minkowski_dot(x, y, keepdim=True):
    """``<x, y>_L`` — the Minkowski inner product, negative in the time slot."""
    prod = x * y
    return (prod[..., 1:].sum(dim=-1, keepdim=True) - prod[..., :1]
            if keepdim else
            prod[..., 1:].sum(dim=-1) - prod[..., 0])


def project_to_hyperboloid(spatial):
    """Spatial coordinates in R^n -> a point on H^n.

    Solving the constraint for the time coordinate, ``x_0 = sqrt(1 + ||x||^2)``,
    means the result is *exactly* on the manifold by construction.  This is far
    safer than normalising an unconstrained (n+1)-vector, which leaves points
    drifting off the sheet whenever the optimiser takes a large step.
    """
    time = torch.sqrt(1.0 + spatial.pow(2).sum(dim=-1, keepdim=True))
    return torch.cat([time, spatial], dim=-1)


def lorentz_distance(x, y):
    """Geodesic distance ``arccosh(-<x,y>_L)``, evaluated stably near zero.

    The naive ``acosh(clamp(inner, min=1+eps))`` is wrong in a way that is easy
    to miss: ``arccosh(1+eps) ~ sqrt(2*eps)``, so an eps of 1e-7 reports a
    distance of 4.5e-4 between a point and *itself*.  That is far larger than
    the eps suggests, and it corrupts nearest-code assignment for exactly the
    close pairs quantisation cares about.

    Writing ``t = -<x,y>_L - 1`` and using the identity
    ``arccosh(1+t) = log1p(t + sqrt(t^2 + 2t))`` gives exactly 0 at ``t = 0``
    while remaining accurate for large ``t``.
    """
    inner = -minkowski_dot(x, y, keepdim=False)
    t = (inner - 1.0).clamp_min(0.0)
    return torch.log1p(t + torch.sqrt(t * t + 2.0 * t))


def lorentz_sqdist(x, y):
    """Smooth surrogate for the SQUARED geodesic distance: ``-<x,y>_L - 1``.

    Use this, never ``lorentz_distance(...)**2``, in anything that is
    differentiated.  The geodesic distance contains ``sqrt(t^2 + 2t)`` whose
    slope is infinite at ``t = 0``, so the moment a point converges onto its
    target autograd evaluates ``2 * 0 * inf`` and returns NaN — which is exactly
    what a commitment loss drives towards.

    This quantity equals ``d^2 / 2`` to leading order, is monotone in ``d``, and
    is a polynomial in the coordinates, so its gradient is finite everywhere.
    """
    return (-minkowski_dot(x, y, keepdim=False) - 1.0).clamp_min(0.0)


def project_to_tangent(x, v):
    """Project an ambient **vector** onto the tangent space at ``x``.

    Correct because ``<x, P(v)>_L = <x,v>_L + <x,v>_L * <x,x>_L = 0`` when
    ``<x,x>_L = -1``.  Note this takes a vector, not a gradient — see
    :func:`riemannian_grad`.
    """
    return v + minkowski_dot(x, v) * x


def riemannian_grad(x, grad):
    """Euclidean gradient -> Riemannian gradient on the hyperboloid.

    Two steps, and skipping the first is a silent bug that *looks* like a
    working optimiser.  A gradient is a covector; turning it into a vector means
    raising its index with the metric, and the Minkowski metric has signature
    ``(-,+,...,+)``, so the **time component flips sign**.  Projecting the raw
    Euclidean gradient without that produces a direction which is not descent —
    observed here as a loss that *rises* for a few steps and then diverges to
    NaN, rather than the obvious immediate failure one might hope for.
    """
    g = grad.clone()
    g[..., 0] = -g[..., 0]
    return project_to_tangent(x, g)


def expmap(x, v, eps=1e-7):
    """Exponential map: follow the geodesic from ``x`` along tangent ``v``."""
    norm = torch.sqrt(minkowski_dot(v, v).clamp_min(eps))
    norm = norm.clamp(max=8.0)   # cosh(8)~1490; beyond this the Minkowski
                                 # cancellation destroys float64 precision
    return torch.cosh(norm) * x + torch.sinh(norm) * v / norm


def logmap(x, y, eps=1e-7):
    """Inverse of :func:`expmap` — the tangent at ``x`` pointing to ``y``."""
    alpha = -minkowski_dot(x, y)
    alpha = alpha.clamp_min(1.0 + eps)
    u = y - alpha * x
    norm = torch.sqrt(minkowski_dot(u, u).clamp_min(eps))
    return torch.acosh(alpha) * u / norm.clamp_min(eps)


def parallel_transport(x, y, v, eps=1e-9):
    """Carry tangent ``v`` from ``T_x`` to ``T_y`` along the connecting geodesic.

    Needed for momentum: a velocity accumulated at the old point is not a valid
    tangent at the new one, and reusing it directly is the most common bug in
    Riemannian optimisers.

    ``PT(v) = v + <y,v>_L / (1 - <x,y>_L) * (x + y)``, which is an isometry —
    verified here to 1e-16 — and lands exactly in ``T_y``.

    An earlier version wrote ``alpha = -minkowski_dot(x, y).clamp_min(1 + eps)``,
    intending ``(-dot).clamp_min(...)``.  Python binds the unary minus *after*
    the method call, so a dot product of about -1 was clamped up to +1 and then
    negated to -1, making the denominator ~-1e-7.  The measured effect was a
    10^11 amplification of the momentum per step, which presented as the
    optimiser diverging rather than as anything resembling a sign error.
    """
    denom = (1.0 - minkowski_dot(x, y)).clamp_min(eps)
    return v + minkowski_dot(y, v) / denom * (x + y)


def frechet_mean(points, weights=None, iters=8):
    """Approximate Karcher/Frechet mean by tangent-space averaging.

    The hyperbolic analogue of an arithmetic mean has no closed form, so this
    iterates: log-map every point to the current estimate's tangent space,
    average there (where the space is a vector space), exp-map back.  Used for
    the VQ codebook update, where a Euclidean average would leave the codebook
    off the manifold entirely.
    """
    if weights is None:
        weights = points.new_ones(points.shape[:-1])
    w = (weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)).unsqueeze(-1)
    mean = project_to_hyperboloid(points[..., 1:].mean(dim=-2))
    for _ in range(iters):
        tangents = logmap(mean.unsqueeze(-2), points)
        step = (w * tangents).sum(dim=-2)
        mean = expmap(mean, step)
    return mean


def expmap0(v):
    """Tangent vector at the origin -> a point on H^n.

    The origin is ``(1, 0, ..., 0)``, whose tangent space is the set of vectors
    with zero time component, so only the spatial part carries information.
    Writing the map as ``sinh(|v|) * v/|v|`` in the spatial slot and letting
    :func:`project_to_hyperboloid` recompute the time slot keeps the result
    exactly on the manifold, which the closed form does not once ``|v|`` is
    large enough for ``cosh`` and ``sqrt(1+sinh^2)`` to disagree in float32.
    """
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7, max=8.0)
    return project_to_hyperboloid(torch.sinh(norm) * v / norm)


def logmap0(x):
    """Inverse of :func:`expmap0` — a point on H^n as a tangent at the origin.

    Returns ``n`` spatial coordinates, not ``n+1``: the time component of a
    tangent at the origin is identically zero, so carrying it would hand every
    downstream layer a column of zeros.
    """
    spatial = x[..., 1:]
    norm = spatial.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    # asinh rather than acosh(x_0): x_0 is a difference of large numbers at big
    # radius, while |spatial| is computed directly and stays accurate.
    return torch.asinh(norm) * spatial / norm


def lorentz_centroid(points, weights=None, dim=-2):
    """Weighted centroid on the hyperboloid, in closed form.

    The Frechet mean of :func:`frechet_mean` needs a fixed-point iteration; the
    *Lorentzian* centroid minimises squared Lorentzian distance instead of
    squared geodesic distance and has an exact solution — take the weighted
    ambient sum and renormalise it back onto the sheet.  It is the standard
    aggregation for hyperbolic attention because it is one line, has no inner
    loop to unroll through, and its gradient is finite everywhere.

    ``frechet_mean`` remains the right choice for the codebook update, where it
    runs under ``no_grad`` and the exactness matters more than the cost.
    """
    if weights is None:
        summed = points.sum(dim=dim)
    else:
        summed = (points * weights.unsqueeze(-1)).sum(dim=dim)
    # |<s,s>_L| is positive for any non-degenerate timelike sum; the clamp only
    # catches the all-weights-zero case, which cannot be normalised at all.
    norm = torch.sqrt((-minkowski_dot(summed, summed)).clamp_min(1e-9))
    out = summed / norm
    # One re-projection: the division lands on the sheet analytically, but in
    # float32 it drifts by ~1e-6 per layer and that accumulates into acosh
    # domain errors several layers later.
    return project_to_hyperboloid(out[..., 1:])


def origin_like(x):
    """The hyperboloid origin ``(1, 0, ..., 0)`` broadcast to ``x``'s shape."""
    o = torch.zeros_like(x)
    o[..., 0] = 1.0
    return o


def radius(x):
    """Geodesic distance of each point from the origin — the "level" coordinate.

    In this project's hierarchy that is not a metaphor: the record checkpoint
    converged to level radii [0.125, 0.445, 1.722] from an initialisation of
    [0.25, 0.6, 1.2], with nothing constraining it. Radius *is* abstraction.
    """
    return lorentz_distance(x, origin_like(x))


def parallel_transport0(y, u):
    """Transport a spatial tangent at the origin to ``T_y``.

    ``u`` is given in ``R^n``; it is lifted to ``(0, u) in T_o`` and then
    transported. This is the sampling half of :func:`wrapped_normal`.
    """
    zeros = torch.zeros_like(u[..., :1])
    return parallel_transport(origin_like(y), y, torch.cat([zeros, u], dim=-1))


def wrapped_normal(mean, std, generator=None):
    """Sample a wrapped normal on H^n centred at ``mean``.

    Draw ``u ~ N(0, std^2)`` in ``T_o``, parallel-transport it to ``T_mean``,
    exp-map. The manifold analogue of ``x + sigma * eps``, and what makes a
    forward diffusion process well defined on the hyperboloid.
    """
    spatial = torch.randn(mean.shape[:-1] + (mean.shape[-1] - 1,),
                          device=mean.device, dtype=mean.dtype, generator=generator)
    return expmap(mean, parallel_transport0(mean, spatial * std))


# --------------------------------------------------------------------------
# Hierarchy losses — training the geometry instead of measuring it
# --------------------------------------------------------------------------
# HypVQNet *initialised* its codebooks at increasing radii and then checked
# whether the ordering survived training. It did, which is a real result, but it
# was luck the model was not asked to reproduce. These two losses make the same
# statement a penalty, which is what a generator needs: the diffusion samples
# level 0 first and refines downward, so the radius ordering has to hold on
# every sequence rather than on average.


def hyperbolic_triplet_loss(anchor, positive, negative, margin=0.5, squared=True):
    """Margin triplet loss under the Lorentz metric.

    ``squared=True`` uses :func:`lorentz_sqdist`, the smooth surrogate. The
    geodesic distance itself has infinite slope where anchor meets positive,
    which is exactly the configuration this loss drives towards, so the raw
    distance NaNs on convergence.
    """
    d = lorentz_sqdist if squared else lorentz_distance
    return torch.relu(d(anchor, positive) - d(anchor, negative) + margin).mean()


def entailment_loss(child, parent, margin=0.1):
    """Push ``parent`` nearer the origin than ``child``."""
    return torch.relu(radius(parent) - radius(child) + margin).mean()


def busemann(x, directions, eps=1e-7):
    """Busemann function to IDEAL prototypes — points at infinity, not in space.

    An ideal point is a null direction ``xi = (1, u)`` with ``|u| = 1``, so
    ``<xi,xi>_L = 0``: it sits on the boundary, infinitely far away. The Busemann
    function is the renormalised distance to it,

        b_xi(x) = log(-<x, xi>_L) = log(x_0 - <x_s, u>)

    and the argument is strictly positive because ``x_0 = sqrt(1+|x_s|^2) >
    |x_s| >= <x_s, u>`` for a unit ``u``. No boundary case, finite gradient.
    ``b(origin) = 0`` by construction, so the scale needs no calibration.

    **Why ideal prototypes and not points.** A prototype at finite radius can
    drift, and in this project it did: HypDiscreteAE's 64 prototypes spread to
    radius ~4 until they were equidistant from every input and the readout went
    flat, and the program anchors here degenerated to 3 of 32 in use. A
    prototype specified by a DIRECTION cannot do either -- there is no radius to
    run away to, and two distinct directions stay distinct however far the data
    moves.
    """
    # The prototypes are held in float64 with the rest of the manifold state,
    # but `sem_point` arrives from the attention in float32. Cast at the
    # boundary rather than forcing either side.
    inner = x[..., 1:] @ directions.to(x.dtype).t()
    return torch.log((x[..., :1] - inner).clamp_min(eps))


def busemann_prototype_loss(x, directions, reg=0.25, balance=0.5, temp=1.0,
                            eps=1e-7):
    """Pull each point toward its nearest ideal prototype, at a bounded radius.

    Three terms, and every one of them is load-bearing.

    ``nearest`` is the Busemann pull. On its own it escapes: ``b`` behaves like
    ``-r`` along its own direction and is unbounded below.

    ``reg * r^2`` stops the escape AND sets where. Minimising ``-r + reg*r^2``
    puts the optimum at ``r* = 1/(2*reg)`` -- verified empirically at 2.00 for
    reg 0.25 -- so this constant chooses the working radius rather than being
    tuned blind. 0.25 targets 2, beside the outer level's 2.2 and well short of
    the ~4 where HypDiscreteAE's prototypes went flat.

    ``balance`` is what stops the degenerate equilibrium, and without it the
    whole construction fails: measured at initialisation, **1 of 32 prototypes
    received every point**, entropy ~0. With a hard nearest-prototype
    assignment that state is self-reinforcing -- every point is pulled the same
    way, so nothing ever separates. The ideal-prototype literature does not hit
    this because prototypes there are assigned by CLASS LABEL; an unsupervised
    min-assignment has no such anchor. Penalising the negative entropy of the
    mean soft assignment makes uniform usage the cheaper option.
    """
    b = busemann(x, directions, eps=eps)
    nearest = b.min(dim=-1).values
    total = (nearest + reg * radius(x) ** 2).mean()
    if balance:
        marginal = (-b / temp).softmax(dim=-1).mean(dim=0)
        total = total + balance * (marginal * (marginal + 1e-12).log()).sum()
    return total


@torch.no_grad()
def busemann_report(x, directions, eps=1e-7):
    b = busemann(x, directions, eps=eps)
    idx = b.argmin(dim=-1)
    counts = torch.bincount(idx, minlength=directions.shape[0]).double()
    p = counts / counts.sum().clamp_min(1)
    return {
        "prototypes_used": int((counts > 0).sum()),
        "n_prototypes": int(directions.shape[0]),
        "assign_entropy": float(-(p * (p + 1e-12).log()).sum()),
        "max_entropy": float(torch.log(torch.tensor(float(len(counts))))),
        "nearest_b_mean": float(b.min(dim=-1).values.mean()),
        "radius_mean": float(radius(x).mean()),
    }


def pcgrad(losses, params):
    """Project conflicting task gradients onto each other's normal plane.

    Yu et al. 2020. Where two objectives disagree -- ``<g_i, g_j> < 0`` -- the
    conflicting component of ``g_i`` is removed and the rest kept, instead of
    scaling both down uniformly.

    That distinction is the point here. This project's crude alternative is a
    scalar (``code_grad``) that shrinks the WHOLE activity gradient reaching the
    codebook, including the part reconstruction has no quarrel with. The measured
    conflict is severe -- an unscaled activity gradient took reconstruction from
    0.9990 to 0.3445 -- so there is a genuine conflicting component to remove,
    and no reason to pay for it with the agreeing one.

    Costs one backward pass per task; the projection itself is two dot products
    per pair over a flattened 2.3M-element vector, which is nothing beside it.
    """
    flat = []
    for i, loss in enumerate(losses):
        for p in params:
            p.grad = None
        loss.backward(retain_graph=(i < len(losses) - 1))
        # float64 throughout: the manifold parameters are double and the rest
        # float32, so a mixed concatenation would silently promote and then fail
        # to write back. Cast on the way out instead.
        flat.append(torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p))
            .reshape(-1).double()
            for p in params]).detach())

    out = [g.clone() for g in flat]
    for i in range(len(out)):
        for j in range(len(flat)):
            if i == j:
                continue
            dot = torch.dot(out[i], flat[j])
            if dot < 0:
                out[i] -= dot / flat[j].dot(flat[j]).clamp_min(1e-12) * flat[j]

    total = torch.stack(out).sum(0)
    idx = 0
    for p in params:
        n = p.numel()
        p.grad = total[idx:idx + n].view_as(p).to(p.dtype).clone()
        idx += n
    return {"conflict": float(sum(
        (torch.dot(flat[i], flat[j]) < 0).double()
        for i in range(len(flat)) for j in range(len(flat)) if i != j))}


def entailment_cone_loss(child, parent, cone_k=0.3, radial_margin=0.0, eps=1e-6):
    """Keep each child inside its parent's ANGULAR cone, not merely outside it.

    :func:`entailment_loss` above orders radii and nothing else -- one scalar per
    pair, satisfied by moving outward in any direction whatsoever. Measured on
    the coupled checkpoint, it was violated on **0.0007%** of child positions,
    which is another way of saying it constrains nothing.

    A cone constrains DIRECTION. The child must lie within half-aperture
    ``psi(r_p)`` of its parent's direction, and the aperture NARROWS as the
    parent moves outward: ``sin(psi) = cone_k / sinh(r_p)``. That is the
    geometry doing the work -- volume grows like ``sinh^(n-1)(r)``, so a parent
    far out can hold the same number of descendants in a smaller angle, and a
    parent near the origin needs a wide one.

    Why this is a COOPERATION mechanism rather than another penalty: once the
    coarse level fixes a region, the finer levels only have to say where inside
    it the sequence sits. Fewer bits for the same reconstruction, and the fine
    detail cannot contradict the program. Its signature is measurable --
    ``fsq_grid_used`` at the fine levels should FALL while reconstruction holds.

    Computed entirely in cosines. ``theta <= psi`` is ``cos theta >= cos psi``,
    which is polynomial in the coordinates; going through ``acos`` would put an
    infinite derivative exactly where the loss drives, which is the mistake
    ``lorentz_sqdist`` exists to avoid.
    """
    vp, vc = logmap0(parent), logmap0(child)
    rp = vp.norm(dim=-1, keepdim=True).clamp_min(eps)
    up = vp / rp
    uc = vc / vc.norm(dim=-1, keepdim=True).clamp_min(eps)
    cos_theta = (up * uc).sum(dim=-1)
    # clamp below 1: at sin(psi) = 1 the cone is a half-space and the sqrt below
    # would sit on its infinite-slope point.
    ratio = (cone_k / torch.sinh(rp.squeeze(-1)).clamp_min(eps)).clamp(max=0.99)
    cos_psi = torch.sqrt((1.0 - ratio * ratio).clamp_min(1e-6))
    angular = torch.relu(cos_psi - cos_theta)
    if not radial_margin:
        return angular.mean()
    # A cone is CONTAINMENT: inside the aperture *and* beyond the parent. The
    # angle alone lets a child sit at the same radius as its parent, pointing the
    # right way and refining nothing. The separate radial `entailment_loss` does
    # not fix that either -- with no margin it is satisfied by any outward step
    # at all, measured at 0.0007% violations. A margin makes it bite: at the
    # radii these states occupy (0.79 / 0.80 / 0.87) a margin of 0.2 is not free.
    rc = vc.norm(dim=-1)
    radial = torch.relu(rp.squeeze(-1) + radial_margin - rc)
    return (angular + radial).mean()


@torch.no_grad()
def cone_report(level_points, cone_k=0.3, eps=1e-6):
    """Is the cone constraint BINDING, or satisfied for free like the radial one?"""
    out = {"violated": [], "aperture_deg": [], "angle_deg": []}
    for coarse, fine in zip(level_points, level_points[1:]):
        parent = _match_positions(coarse, fine.shape[1])
        vp, vc = logmap0(parent), logmap0(fine)
        rp = vp.norm(dim=-1, keepdim=True).clamp_min(eps)
        up, uc = vp / rp, vc / vc.norm(dim=-1, keepdim=True).clamp_min(eps)
        cos_theta = (up * uc).sum(dim=-1).clamp(-1.0, 1.0)
        ratio = (cone_k / torch.sinh(rp.squeeze(-1)).clamp_min(eps)).clamp(max=0.99)
        psi = torch.asin(ratio)
        theta = torch.acos(cos_theta)
        out["violated"].append(float((theta > psi).double().mean()))
        out["aperture_deg"].append(float(psi.mean() * 180 / torch.pi))
        out["angle_deg"].append(float(theta.mean() * 180 / torch.pi))
    return out


def hierarchy_cone(level_points, cone_k=0.3, radial_margin=0.0):
    """:func:`entailment_cone_loss` across a coarse->fine stack."""
    if len(level_points) < 2:
        return level_points[0].new_zeros(())
    total = None
    for coarse, fine in zip(level_points, level_points[1:]):
        term = entailment_cone_loss(fine, _match_positions(coarse, fine.shape[1]),
                                    cone_k=cone_k, radial_margin=radial_margin)
        total = term if total is None else total + term
    return total / (len(level_points) - 1)


def _match_positions(coarse, length):
    """Gather ``coarse`` ``[B, Lc, D]`` at the ancestor of each of ``length`` positions."""
    if coarse.shape[1] == length:
        return coarse
    idx = (torch.arange(length, device=coarse.device).float()
           * coarse.shape[1] / length).long().clamp_max(coarse.shape[1] - 1)
    return coarse.index_select(1, idx)


def hierarchy_entailment(level_points, margin=0.1):
    """Coarse levels nearer the origin than the fine levels refining them.

    ``level_points`` is ordered coarse -> fine. Levels live at different
    resolutions, so each fine position is compared against **its own** coarse
    ancestor — the proportional index, which is the ancestry that top-down
    residual quantisation already established. Comparing pooled radii instead
    would let a level satisfy the constraint on average while violating it
    everywhere that matters.
    """
    if len(level_points) < 2:
        return level_points[0].new_zeros(())
    total = None
    for coarse, fine in zip(level_points, level_points[1:]):
        term = entailment_loss(fine, _match_positions(coarse, fine.shape[1]), margin=margin)
        total = term if total is None else total + term
    return total / (len(level_points) - 1)


def hierarchy_triplet(level_points, margin=0.3, max_triplets=256):
    """A fine code should sit nearer its own coarse ancestor than a stranger's.

    The negative is the same coarse level at a **shuffled batch position**, so
    the loss tests that the hierarchy is per-sequence rather than a global
    average — a hierarchy that only holds on the batch mean is decoration.
    """
    if len(level_points) < 2:
        return level_points[0].new_zeros(())
    total, count = None, 0
    for coarse, fine in zip(level_points, level_points[1:]):
        b, lf = fine.shape[0], fine.shape[1]
        take = min(max_triplets, b * lf)
        pos_idx = torch.randint(0, lf, (take,), device=fine.device)
        bat_idx = torch.randint(0, b, (take,), device=fine.device)
        parent_idx = (pos_idx.float() * coarse.shape[1] / lf).long().clamp_max(coarse.shape[1] - 1)
        shuffled = torch.randperm(b, device=fine.device)[bat_idx]
        term = hyperbolic_triplet_loss(fine[bat_idx, pos_idx],
                                       coarse[bat_idx, parent_idx],
                                       coarse[shuffled, parent_idx], margin=margin)
        total = term if total is None else total + term
        count += 1
    return total / max(count, 1)


class LorentzLinear(nn.Module):
    """Linear map H^n -> H^m that never leaves the manifold.

    The usual "hyperbolic linear layer" is a sandwich: log-map to the tangent
    space at the origin, apply an ordinary matrix, exp-map back.  That is a
    Euclidean layer with two extra transcendental calls, and it is only faithful
    near the origin — precisely not where a hierarchy's leaves live.

    This is the direct construction (Chen et al., 2022): apply the matrix to the
    ambient coordinates to obtain the spatial part, then *solve* the constraint
    for the time coordinate.  The result is on H^m by construction for any
    weight matrix, so the layer needs no projection step and no orthogonality
    penalty.

    ``scale`` controls how far outputs sit from the origin.  Initialising small
    matters for the same reason it does in :class:`HyperbolicVQ`: the manifold
    expands fastest far out, so starting wide means starting where float32
    resolves points worst.
    """

    def __init__(self, in_dim, out_dim, scale=0.5, bias=True, dropout=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim + 1) * (1.0 / (in_dim + 1) ** 0.5))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale = float(scale)
        self.dropout = float(dropout)

    def forward(self, x):
        w = self.weight
        if self.training and self.dropout > 0:
            w = F.dropout(w, self.dropout)
        spatial = F.linear(x, w, self.bias) * self.scale
        return project_to_hyperboloid(spatial)

    def extra_repr(self):
        return (f"in={self.weight.shape[1] - 1}, out={self.weight.shape[0]}, "
                f"scale={self.scale}")


# --------------------------------------------------------------------------
# Riemannian optimiser
# --------------------------------------------------------------------------

class RiemannianAdam(torch.optim.Optimizer):
    """Adam on the Lorentz manifold.

    Differs from Euclidean Adam in three places, each of which is required for
    correctness rather than being an refinement:

    1. the Euclidean gradient is projected into the tangent space;
    2. the second-moment norm is Minkowski, not Euclidean;
    3. the update uses the exponential map, and momentum is parallel-transported
       to the new point instead of being reused verbatim.

    Parameters carrying ``manifold=True`` in their group are treated as
    hyperbolic; everything else falls through to ordinary Adam, so one optimiser
    can drive a mixed model.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, manifold=False, max_step=1.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, manifold=manifold,
                                      max_step=max_step))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    # Element-wise second moment, as in Euclidean Adam.  A
                    # single scalar Minkowski norm per point was tried first and
                    # diverges to NaN: as the loss approaches zero the norm
                    # collapses, the denominator falls to eps, and the step
                    # explodes into an expmap that overflows.
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                if not group["manifold"]:
                    # DECOUPLED weight decay, as in AdamW -- not `grad.add(p,
                    # alpha=wd)`, which is L2 inside Adam and a different
                    # algorithm.
                    #
                    # This mattered far more than it looks. Turning
                    # `training.riemannian: true` on routes EVERY parameter
                    # through this optimiser, not only the handful on the
                    # manifold: 189 of 190 tensors in HypCoupledNet. With this
                    # project's weight_decay of 0.08 the coupled form scales the
                    # decay by Adam's per-parameter denominator, so parameters
                    # with small gradients get decayed enormously harder. The
                    # flag silently swapped the optimiser for the whole model
                    # while appearing to concern 544 parameters.
                    #
                    # Measured 2026-08-21: HypCoupledNet reached Dev 0.4019 at
                    # epoch 1 with the coupled form against 0.5048 for the
                    # otherwise-identical AdamW run. It also means every
                    # HypDiscreteAE collapse attributed to "the readout, not the
                    # optimiser" was confounded by this, and that post-mortem
                    # should be re-run before it is trusted.
                    if group["weight_decay"]:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    m.mul_(b1).add_(grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    denom = (v / (1 - b2 ** t)).sqrt().add_(group["eps"])
                    p.addcdiv_(m / (1 - b1 ** t), denom, value=-group["lr"])
                    continue

                rgrad = riemannian_grad(p, grad)
                m.mul_(b1).add_(rgrad, alpha=1 - b1)
                v.mul_(b2).addcmul_(rgrad, rgrad, value=1 - b2)
                denom = (v / (1 - b2 ** t)).sqrt().add_(group["eps"])
                # Re-project after the element-wise rescale: dividing component
                # by component does not preserve tangency, and an off-tangent
                # step walks the point off the hyperboloid.
                direction = project_to_tangent(p, (m / (1 - b1 ** t)) / denom)

                # Clip the geodesic step length.  Element-wise Adam divides by a
                # second moment that is near `eps` on the first steps, which
                # produces an enormous direction; one such step throws the point
                # to a radius where `<x,x>_L = ||s||^2 - x0^2` is catastrophic
                # cancellation even in float64, and every subsequent distance is
                # garbage.  Clipping the tangent norm is what makes this
                # optimiser usable rather than merely correct on paper.
                step_norm = torch.sqrt(minkowski_dot(direction, direction).clamp_min(0.0))
                scale = (group["max_step"] / step_norm.clamp_min(1e-12)).clamp(max=1.0)
                new_p = expmap(p, -group["lr"] * direction * scale)
                # Re-project: repeated exp maps accumulate float error and the
                # constraint drifts, which silently corrupts every distance.
                new_p = project_to_hyperboloid(new_p[..., 1:])
                m.copy_(parallel_transport(p, new_p, m))
                p.copy_(new_p)
        return loss


# --------------------------------------------------------------------------
# Hyperbolic vector quantisation
# --------------------------------------------------------------------------

class HyperbolicVQ(nn.Module):
    """VQ whose codebook lives on the hyperboloid.

    Assignment uses geodesic distance; the codebook update is a Frechet mean of
    the assigned encoder outputs rather than an arithmetic one.  Encoder outputs
    arrive as Euclidean vectors and are lifted to the manifold, so the rest of
    the network stays ordinary.

    ``radius_scale`` shrinks the lift.  Points far from the origin sit where the
    manifold expands fastest, which is where float32 loses precision; keeping
    the initial spread modest lets the hierarchy push outward as it learns
    instead of starting at the numerically bad end.
    """

    def __init__(self, dim, codebook_size=256, decay=0.99, commit_weight=0.25,
                 radius_scale=0.3, eps=1e-5, dead_threshold=0.5,
                 frechet=False, learnable=False, codebook_weight=1.0,
                 enc_na_powloce=False, pasmo_r=0.0, przes_powloki=False,
                 promien_specyf=0.0):
        super().__init__()
        self.dim, self.codebook_size = int(dim), int(codebook_size)
        self.decay, self.commit_weight, self.eps = float(decay), float(commit_weight), float(eps)
        self.radius_scale = float(radius_scale)
        self.dead_threshold = float(dead_threshold)
        self.enc_na_powloce = bool(enc_na_powloce)
        # Pasmo promienia per sekwencja. 0,0 = stare zachowanie bit w bit.
        self.pasmo_r = float(pasmo_r)
        self.r_mlp = (nn.Sequential(nn.Linear(dim, 16), nn.SiLU(),
                                    nn.Linear(16, 1))
                      if self.pasmo_r > 0.0 else None)
        if self.r_mlp is not None:
            # Ostatnia warstwa mala, nie zerowa: przy zerze `tanh(0) = 0` daje
            # dokladnie stary promien, ale gradient do MLP tez jest zerowy
            # tylko wtedy, gdy zerowa jest waga -- a tu nie jest, bo `tanh'(0)
            # = 1`. Maly init = start blisko powloki z ZYWYM gradientem.
            nn.init.normal_(self.r_mlp[2].weight, std=0.01)
            nn.init.zeros_(self.r_mlp[2].bias)

        # RZUT LICZONY W float64. Przy float32 blad rzutu na hiperboloide
        # wynosi ~1e-6 i PRZEZYWA pozniejsze `.double()`, bo cast nie odtwarza
        # utraconych bitow. Zmierzone zaraz po budowie:
        #     max|<x,x>_L + 1| = 1,04e-06 / 5,24e-06 / 2,07e-05
        # identycznie z pasmem i bez niego, i niezmiennie po pieciu krokach EMA
        # -- czyli zrodlem byla INICJALIZACJA, nie aktualizacja kodebooka.
        # Znikalo dopiero przy pierwszym `renormalise_radius()`, wiec bylo
        # nieszkodliwe, ale kazda sonda geometrii musialaby ten prog omijac.
        spatial = (torch.randn(self.codebook_size, dim) * radius_scale).double()
        self.frechet = bool(frechet)
        self.learnable = bool(learnable)
        self.codebook_weight = float(codebook_weight)
        _emb = project_to_hyperboloid(spatial)
        if self.learnable:
            # Kodebook NA ROZMAITOSCI jako parametr: dopiero wtedy
            # `RiemannianAdam` ma co optymalizowac. Przy buforze byl tylko
            # aktualizowany EMA, a optymalizator Lorentza nie dzialal wcale.
            self.embed = nn.Parameter(_emb)
        else:
            self.register_buffer("embed", _emb)
        self.register_buffer("cluster_size", torch.ones(self.codebook_size))
        # PROMIEN DOCELOWY POWLOKI. `radius_scale` ustawia wylacznie
        # INICJALIZACJE; po niej nic nie trzyma promienia i kodebook dryfuje
        # tam, gdzie postawi punkty enkoder. Zmierzone na biegu HypEncE:
        # promienie wpisow 1,4235 / 2,2472 / 2,9279 przy starcie spadly do
        # 0,3335 / 0,5542 / 1,6233 w epoce 1, a w epoce 2 L1 (0,3479) spadl
        # PONIZEJ L0 (0,3619) -- porzadek radialny sie odwrocil. Ta sama
        # awaria zmierzona wczesniej na `hq_proto` (spadek 722-1142x).
        with torch.no_grad():
            _r0 = float(radius(_emb).mean())
        self.register_buffer("promien_cel", torch.tensor(_r0))
        # PRZESUNIECIE POWLOKI -- jeden skalar na poziom, UCZONY.
        #
        # ZMIERZONE na HypEncZ epoka 1 (pasmo stale 0,15): srednie
        # przesuniecie enkodera wobec kodebooka wyniosło +0,1499 na L0 i
        # -0,1308 na L2, czyli **99,9 % i 87,2 % calego pasma**. `tanh` byl
        # NASYCONY. Nasycony `tanh` zwraca stala i ma pochodna bliska zeru,
        # wiec pasmo nie niosło zadnej tresci per sekwencja -- caly jego
        # budzet poszedl na STALE przesuniecie, a mechanizm istnial tylko
        # z nazwy. Bieg placil za to -4,6 sigma w epoce 1 i -2,8 sigma w
        # epoce 2 wobec trzech kontroli (0,64975 / 0,64595 / 0,64535).
        #
        # Enkoder domaga sie innego promienia powloki niz `radius_scale`
        # wylosowal przy inicjalizacji. To jest zadanie dla JEDNEJ liczby na
        # poziom, nie dla sieci per pozycja. Rozdzielenie obu rzeczy zwalnia
        # `tanh` do tego, po co jest.
        #
        # OGRANICZENIE zamyka droge ucieczki, ktora rozbila HypEncHfix
        # (promien 1,44 -> 8,22, commit 213 -> 3428, recon 0,9986 -> 0,4155).
        #
        # Wartosc jest USTAWIANA Z ZEWNATRZ przez `ustaw_przes_max()`, bo tylko
        # drabina zna promienie SASIEDNICH poziomow. Stala 0,35 wpisana tu
        # recznie DAWALA ZACHODZACE ZAKRESY: przy zmierzonych bazach 1,4393 /
        # 2,2723 / 2,9539 gorny kraniec L1 (2,6223) przekraczal dolny kraniec
        # L2 (2,6039), wiec poziomy mogly zamienic sie miejscami -- ta sama
        # awaria co na HypEncE, gdzie L1 (0,3479) spadl PONIZEJ L0 (0,3619).
        # Domyslne 0,0 znaczy "przesuniecie wylaczone" i jest bezpieczne.
        # BRAMKA. Bez niej `przes_pre` istnieje zawsze i liczba parametrow
        # rozni sie od bazy, a wtedy porownanie z kontrola przestaje mierzyc
        # jedna rzecz. Przy `False` powrot do bazy jest BIT W BIT.
        # ===== PROMIEN JAKO SPECYFICZNOSC SYGNALU =====
        # Promien jest JEDYNA wspolrzedna, ktorej euklides nie ma, a dzis
        # jest przypiety. ZMIERZONE: przy przypietym promieniu sfera
        # i hiperboloida daja IDENTYCZNE kody (zgodnosc 1,0000 po
        # splaszczeniu kodebooka), wiec krzywizna NIE MOZE dotknac
        # kwantyzacji. Zeby miala czym dzialac, promien musi cos znaczyc.
        #
        # CZEMU DWIE POPRZEDNIE PROBY PADLY: `pasmo_r` liczylo promien
        # z WOLNEGO MLP na kierunku. MLP nauczyl sie STALEJ i nasycil tanh
        # (|tanh| = 0,9996 na L0, 0,9972 na L2), a rozrzut promienia
        # w obrebie sekwencji wynosil DOKLADNIE 0.
        #
        # TA WERSJA NIE MOZE ZAWIESC TAK SAMO, i to jest wlasnosc
        # STRUKTURALNA. Sygnalem jest SPECYFICZNOSC pozycji:
        #     n_i = ||sp_i||,   z_i = (n_i - sr_j n_j) / sd_j n_j
        # z-score W OBREBIE SEKWENCJI ma z konstrukcji srednia 0 i sd 1,
        # wiec tanh(a*z + b) NIE MOZE zapasc do stalej, dopoki a != 0 --
        # a `a` bliskie zeru jest wtedy jawnym falsyfikatorem.
        #
        # ZNACZENIE GEOMETRYCZNE: objetosc na promieniu rosnie jak
        # sinh^(n-1)(r), wiec pozycja o SILNYM, wyroznionym sygnale dostaje
        # wiekszy promien -- wiecej miejsca na kod SPECYFICZNY -- a pozycja
        # tla laduje blizej origa, gdzie kodow jest malo i sa OGOLNE.
        # To pierwszy mechanizm w tym modelu, ktorego euklidesowy
        # odpowiednik nie odtwarza za darmo.
        #
        # z-score liczony PER SEKWENCJA po pozycjach, wiec `encode_codes(x)`
        # zostaje niezalezne od skladu batcha i dyfuzja jest nietknieta.
        self.promien_specyf = float(promien_specyf)
        if self.promien_specyf > 0.0:
            self.spec_a = nn.Parameter(torch.ones(1))
            self.spec_b = nn.Parameter(torch.zeros(1))
        else:
            self.spec_a = None
            self.spec_b = None
        self.przes_powloki = bool(przes_powloki)
        self.przes_pre = (nn.Parameter(torch.zeros(1))
                          if self.przes_powloki else None)
        self.przes_max = 0.0

    def promien_efektywny(self):
        """Promien powloki tego poziomu: baza plus uczone przesuniecie.

        JEDNO zrodlo prawdy. Uzywaja go `_na_powloke` (rzut enkodera) oraz
        `renormalise_radius` (przypiecie kodebooka) -- gdyby uzywaly roznych
        wartosci, enkoder i kodebook bilyby sie o promien przez caly trening
        i commitment placilby za to bez konca.
        """
        if self.przes_pre is None or self.przes_max <= 0.0:
            return self.promien_cel
        return self.promien_cel + self.przes_max * torch.tanh(self.przes_pre[0])

    @torch.no_grad()
    def renormalise_radius(self):
        """Przywraca SREDNI promien wpisow do `promien_cel`, zachowujac ksztalt.

        Rozdziela dwie rzeczy, ktore dzis sa zlepione: **kierunek** na
        hiperboloidzie niesie TRESC kodu i ma byc uczony, a **promien** koduje
        POZIOM i ma byc staly z konstrukcji. Bez tego promien jest cechowaniem,
        w ktorym optymalizator wedruje do origa -- a wtedy `hierarchy_report`
        slusznie melduje, ze poziomy "competing to explain the same variance
        rather than refining one another".

        Skaluje MEDIANE-zachowawczo: mnozy czesc przestrzenna przez staly
        czynnik, wiec wzgledny rozrzut promieni WEWNATRZ poziomu zostaje.
        Renormalizacja jest neutralna dla przebiegu w przod z dokladnoscia do
        skali -- kierunki sie nie zmieniaja.
        """
        sp = self.embed[..., 1:]
        n = sp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        r = torch.asinh(n)
        sr = float(r.mean())
        if sr < 1e-9:
            return 0.0
        # Przypinamy do promienia EFEKTYWNEGO, nie do bazy: inaczej kodebook
        # wracalby co epoke na powloke, ktora enkoder wlasnie opuscil.
        wsp = float(self.promien_efektywny()) / sr
        nowe_n = torch.sinh(r * wsp)
        sp_new = sp * (nowe_n / n)
        self.embed.data = project_to_hyperboloid(sp_new)             if isinstance(self.embed, torch.nn.Parameter) else             project_to_hyperboloid(sp_new)
        return wsp

    def lift(self, z):
        """Euclidean encoder output ``[B, C, L]`` -> hyperboloid ``[B, L, C+1]``."""
        return project_to_hyperboloid(z.transpose(1, 2) * self.radius_scale)

    def quantise(self, points):
        """Nearest code by geodesic distance, computed as a single GEMM.

        ``lorentz_distance(flat.unsqueeze(1), embed.unsqueeze(0))`` broadcasts to
        ``[N, K, D]`` before reducing — at batch 96, 124 positions, 512 codes and
        dim 128 in float64 that is **6.3 GB** for one level, which autograd then
        holds.  The Minkowski product is bilinear, so flipping the time
        coordinate turns it into ``flat @ embed^T`` at ``[N, K]``: 50 MB.

        Assignment is unchanged, not merely close: ``d = arccosh(-<x,y>_L)`` is
        monotone increasing in ``-<x,y>_L``, so the nearest code is exactly the
        one maximising ``<x,y>_L``.  The arccosh is not needed to decide, only to
        report, and skipping it also skips its precision loss near d=0.
        """
        flat = points.reshape(-1, points.shape[-1])
        flipped = torch.cat([-flat[..., :1], flat[..., 1:]], dim=-1)
        inner = flipped @ self.embed.t()          # [N, K] = <x, e_k>_L
        idx = inner.argmax(dim=1)
        return self.embed[idx], idx

    def _na_powloke(self, points):
        """Rzutuje punkty enkodera na POWLOKE POZIOMU (promien = `promien_cel`).

        POWOD, ZMIERZONY. Bez tego promien wyjscia enkodera jest SWOBODNYM
        stopniem swobody, ktorego nic nie ogranicza, a kodebook jest przypinany
        co epoke przez `renormalise_shells()`. W biegu HypEncH luka rosla przez
        caly trening (enkoder 3,13/3,98/3,66 wobec kodebooka 1,44/2,27/2,95), a
        gdy dolozylem gradient z glowic do enkodera (`st_points`) i jednoczesnie
        adaptacyjne `tau` (ktore usunelo hamulec nasycenia softmaxu), bieg
        HypEncHfix ROZBIEGL SIE: promien L0 1,44 -> 5,93 -> 8,22, commit
        213 -> 3428, recon 0,9986 -> 0,4155, strata 3,54 -> 1446.

        Rzut zamyka te droge ucieczki i czyni twierdzenie projektu
        strukturalnie prawdziwym: **kierunek niesie tresc, promien koduje
        poziom**. Commitment staje sie czysto katowy i jest z gory ograniczony
        przez `2 sinh^2(r)`, wiec nie moze zjesc straty.

        Rozniczkowalne: gradient plynie przez KIERUNEK, a skladowa radialna
        jest wycieta z konstrukcji, nie karana.
        """
        sp = points[..., 1:]
        n = sp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        r = self.promien_efektywny().to(sp.dtype)
        if getattr(self, "promien_specyf", 0.0) > 0.0 and self.spec_a is not None:
            _n = n.squeeze(-1)
            _sr = _n.mean(dim=-1, keepdim=True)
            _sd = _n.std(dim=-1, keepdim=True).clamp_min(1e-6)
            _z = ((_n - _sr) / _sd).unsqueeze(-1)
            _t = torch.tanh(self.spec_a.to(sp.dtype) * _z + self.spec_b.to(sp.dtype))
            r = r + (self.promien_specyf * torch.sinh(r.detach())) * _t
            with torch.no_grad():
                # FALSYFIKATORY: `spec_nasyc` blisko 1 = nasycenie, czyli
                # powrot do awarii `pasmo_r`. `spec_a` blisko 0 = model
                # wylaczyl mechanizm. `r_sd_w` = 0 = promien nadal stala.
                self._spec_nasyc = float(_t.abs().mean())
                self._spec_a = float(self.spec_a)
                self._r_sd_wewn = float(r.std())
        if getattr(self, "pasmo_r", 0.0) > 0.0 and self.r_mlp is not None:
            # ================= PROMIEN PER SEKWENCJA =================
            # WYCOFANE: pomiar "99,97 % wariancji promienia MIEDZY poziomami"
            # nie mogl wyjsc inaczej. Sonda czytala `punkty_poziomow()`, a ta
            # zwraca WPISY KODEBOOKA wybrane indeksami -- nie wyjscie enkodera.
            # Promien kodebooka jest przypinany co epoke przez
            # `renormalise_radius()`, wiec rozklad "trzy wartosci i nic poza
            # nimi" byl wlasnoscia PRZYPIECIA, nie wynikiem o danych.
            #
            # Twierdzenie zostaje w mocy, ale z ODCZYTU KODU, nie z pomiaru:
            # przy `enc_na_powloce: true` `_na_powloke` USTAWIA promien na
            # `promien_cel` dla kazdego punktu, wiec promien enkodera jest
            # staly z konstrukcji. Przestrzen hiperboliczna dostaje wtedy
            # jedyna wspolrzedna, ktorej euklidesowa nie ma, jako
            # TROJWARTOSCIOWA ETYKIETA POZIOMU.
            #
            # CZEGO PASMO NIE ROBI. Straight-through zwraca `points +
            # (quant - points).detach()`, wiec WARTOSC podana dalej to WPIS
            # KODEBOOKA. Trzon, glowice i `pole_mlp` nigdy nie widza promienia
            # enkodera. Pasmo NIE wklada wiec tresci radialnej do latentu
            # wprost -- i kazde takie twierdzenie bylo by falszywe.
            #
            # CO PASMO ROBI NAPRAWDE, jedyna droga. Cel EMA kodebooka to
            # srednia Frecheta punktow enkodera. Przy promieniu przypietym co
            # do bitu KAZDY cel lezy na powloce, wiec KAZDY kod zbiega do
            # powloki -- zmierzony rozrzut promieni kodebooka to 0,0048 /
            # 0,0087 / 0,0260 wobec odleglosci miedzy kodami 0,1371 / 0,3008 /
            # 0,8413, czyli 3,5 / 2,9 / 3,1 %. Rzut na powloke SAM WYPLASZCZA
            # kodebook. Pasmo pozwala celom EMA lezec poza powloka, wiec kody
            # moga rozejsc sie promieniowo i "glebokosc" staje sie wlasnoscia
            # KODU. Droga jest posrednia i wolna (przez EMA), i tak nalezy ja
            # raportowac.
            #
            # DLACZEGO PASMO, A NIE SWOBODA. Rzut na powloke wprowadzono, bo
            # przy swobodnym promieniu bieg HypEncHfix ROZBIEGL SIE: promien L0
            # 1,44 -> 5,93 -> 8,22, commit 213 -> 3428, recon 0,9986 -> 0,4155,
            # strata 3,54 -> 1446. `tanh` ogranicza odchylenie do pasma, wiec
            # droga ucieczki zostaje zamknieta, a promien przestaje byc staly.
            _k = sp / n
            # PASMO SKALOWANE Z POZIOMEM, nie stale.
            #
            # ZMIERZONE mediany odleglosci miedzy kodami: 0,1371 / 0,3008 /
            # 0,8413 przy powlokach 1,4364 / 2,2690 / 2,9535. Ilorazy do
            # sinh(r) to 0,069 / 0,063 / 0,088 -- praktycznie stale, wiec
            # skala geometrii poziomu jest sinh(r).
            #
            # Stale pasmo 0,15 dla wszystkich poziomow bylo BLEDEM: na L0
            # PRZEWYZSZALO typowa odleglosc miedzy kodami (0,137), wiec punkt
            # zepchniety na skraj pasma bylby promieniowo dalej od swojego kodu
            # niz od sasiedniego -- kwantyzacja przestala by isc za kierunkiem.
            # Referencyjny HypEncI ma iloraz rozrzut_promienia / odleglosc
            # miedzy kodami rowny 0,011-0,014.
            #
            # `pasmo_r` jest teraz UŁAMKIEM skali poziomu: przy 0,01 pasma
            # wynosza 0,020 / 0,048 / 0,096, czyli 15 / 16 / 11 % odleglosci
            # miedzy kodami -- dziesieciokrotnie wiecej tresci radialnej niz
            # dzis, a wciaz zbyt malo, by promien zdominowal przypisanie.
            # BEZ rzutowania na float: caly modul VQ jest `.double()`.
            _skala = torch.sinh(r.detach())
            _t = torch.tanh(self.r_mlp(_k))
            r = r + (self.pasmo_r * _skala) * _t
            with torch.no_grad():
                # FALSYFIKATOR NASYCENIA. Sredni |tanh| blisko 1 znaczy, ze
                # pasmo znowu zostalo zjedzone przez stala i nie niesie tresci
                # per sekwencja -- dokladnie awaria z HypEncZ ep1 (0,999).
                # Zdrowa wartosc jest wyraznie ponizej 1; sd musi byc > 0.
                self._r_nasyc = float(_t.abs().mean())
                self._r_sd_wewn = float(r.std())
        cel = torch.sinh(r)
        # PROMIEN ENKODERA JAKO TENSOR, per pozycja. Jedyne miejsce, gdzie ta
        # wielkosc istnieje. Po kwantyzacji jej NIE MA: straight-through zwraca
        # `points + (quant - points).detach()`, wiec wartoscia niesiona dalej
        # jest WPIS KODEBOOKA, a jego promien jest przypinany co epoke przez
        # `renormalise_shells()` (zmierzony rozrzut 0,0048 / 0,0087 / 0,0260).
        # `radius(st_points)` zwraca WLASNIE ten przypiety promien kodu -- ta
        # sama pomylka, co w `punkty_poziomow()`. Tutaj zapisujemy promien
        # PRZED kwantyzacja, czyli ten, ktory `promien_specyf` rozniczkuje
        # per sekwencja i per pozycja.
        #
        # `torch.ones_like(n)` zamiast `expand`: przy wylaczonym `promien_specyf`
        # i `pasmo_r` `r` jest SKALAREM, wiec `expand` na ksztalt [B, L, 1]
        # rzuca. Mnozenie daje poprawny ksztalt w obu przypadkach.
        self.promien_enk = (r.detach() * torch.ones_like(n)).detach()
        return project_to_hyperboloid(sp * (cel / n))

    def forward(self, z):
        b, c, length = z.shape
        points = self.lift(z)                                   # [B, L, C+1]
        if self.enc_na_powloce:
            points = self._na_powloke(points)
        # DIAGNOSTYKA, bez wplywu na obliczenia. `radius_mean` w logu to promien
        # KODEBOOKA, a ten jest przypinany co epoke przez `renormalise_shells()`
        # -- w logu H stoi na 1,4364/2,269/2,9535 co do czwartego miejsca przez
        # wszystkie epoki. Nic natomiast nie przypina promienia WYJSCIA ENKODERA.
        # Jesli enkoder dryfuje na zewnatrz, a kodebook jest przypiety, to
        # `commit = sqdist(points, quant)` rosnie bez ograniczenia -- i jest
        # kandydatem na ten jeden nieograniczony czlon, ktory zjada sume
        # (strata H: 4,11 -> 9,45 przez szesc epok przy poprawie kazdej metryki).
        with torch.no_grad():
            _rp = radius(points.detach())
            _r = _rp.reshape(-1)
            self._enc_radius = (float(_r.mean()), float(_r.std()))
            if not self.enc_na_powloce:
                # `_na_powloke` nie bieglo, wiec nikt nie zapisal promienia.
                # Bez tego `kanal_promienia` cicho gubilby poziom (brak
                # atrybutu -> `continue` -> za waskie `z`), zamiast rzucic.
                self.promien_enk = _rp.unsqueeze(-1)
        quant, idx = self.quantise(points)
        quant = quant.view(b, length, -1)
        idx = idx.view(b, length)

        # `vq_update` pozwala WYLACZYC aktualizacje EMA na czas dodatkowego
        # przebiegu wprzod (negatywy, kontrola RC). Bez tego kazdy drugi
        # forward w trybie treningu przepisuje kodebook danymi, ktore nie sa
        # celem rekonstrukcji -- zmierzone w tym projekcie: recon 0,79 -> 0,62,
        # i wina zostala przypisana innej fladze.
        if self.training and getattr(self, 'vq_update', True):
            self._update(points.reshape(-1, points.shape[-1]), idx.reshape(-1))

        # Commitment uses the SMOOTH squared-distance surrogate, not d**2:
        # the geodesic distance has infinite slope at zero, which is precisely
        # where a commitment loss pushes, so d**2 would NaN on convergence.
        if self.learnable:
            # DWA czlony z detachem, zeby kazda strona miala wlasna wage.
            # Bez rozbicia jeden czlon ciagnalby enkoder do kodow I kody do
            # enkodera ta sama waga -- to utrata kontroli, nie oszczednosc.
            commit = lorentz_sqdist(points, quant.detach()).mean()
            self._codebook_term = lorentz_sqdist(points.detach(), quant).mean()
        else:
            commit = lorentz_sqdist(points, quant).mean()
            self._codebook_term = None
        # Straight-through in the tangent space, then returned to Euclidean
        # coordinates so the downstream trunk is unchanged.
        out = points + (quant - points).detach()
        self._commit_raw = float(commit.detach())
        self.loss = self.commit_weight * commit
        if self._codebook_term is not None:
            self.loss = self.loss + self.codebook_weight * self._codebook_term
        self.indices = idx
        # The straight-through tensor *as a manifold point*, kept for callers
        # that stay on the hyperboloid instead of dropping back to Euclidean
        # coordinates.  Purely additive: the return value is unchanged.
        self.st_points = out
        return (out[..., 1:] / self.radius_scale).transpose(1, 2), idx

    @torch.no_grad()
    def _update(self, flat, idx):
        onehot = torch.zeros(flat.shape[0], self.codebook_size, device=flat.device,
                             dtype=flat.dtype)
        onehot.scatter_(1, idx.unsqueeze(1), 1.0)
        counts = onehot.sum(0)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)

        # Vectorised geodesic EMA: sum the assigned points per code in ambient
        # coordinates, project that sum back to the manifold to get an
        # approximate Frechet mean, then move each code part-way along the
        # geodesic toward it.  The previous version looped in Python at rate
        # 0.01 per step, which left most codes effectively frozen -- measured
        # perplexity 18.5 of 128 against 438 of 512 for the Euclidean stack.
        used = counts > 0
        if used.any() and not self.learnable:
            if self.frechet:
                # KROK FRECHETA. Poprzednia wersja liczyla cel jako srednia
                # ARYTMETYCZNA wspolrzednych przestrzennych, dopiero potem
                # rzutowana na hiperboloide -- obciazenie takiego estymatora
                # ROSNIE Z PROMIENIEM, wiec najgorzej wychodzila powloka
                # najdalsza, ta z najwieksza liczba kodow i najdrobniejszym
                # detalem. Tutaj usredniamy w przestrzeni STYCZNEJ w biezacym
                # kodzie, co jest jedna iteracja sredniej Karchera startujaca z
                # naturalnego dla EMA punktu. Skladanie upraszcza sie, bo
                # logmap(e, expmap(e, s)) = s, wiec ruch o (1-decay) wzdluz
                # sredniej stycznej JEST riemannowska EMA.
                # Pamiec O(N), nie O(K*N): pelny `frechet_mean` na wszystkie
                # kody naraz to [K, N, D] i przy K=512, N=11904 ponad 3 GB.
                t = logmap(self.embed[idx], flat)
                step = (onehot.t() @ t)[used] / counts[used].unsqueeze(1)
                moved = expmap(self.embed[used], (1 - self.decay) * step)
            else:
                summed = onehot.t() @ flat
                target = project_to_hyperboloid(
                    summed[used][..., 1:] / counts[used].unsqueeze(1))
                v = logmap(self.embed[used], target)
                moved = expmap(self.embed[used], (1 - self.decay) * v)
            self.embed[used] = project_to_hyperboloid(moved[..., 1:])

        # Dead-code restart.  Without it a codebook only ever loses capacity:
        # a code that stops winning receives no gradient and no EMA update, so
        # collapse is absorbing.  Dead codes are re-seeded onto live encoder
        # outputs, which is what keeps the outer, high-capacity shells usable.
        dead = self.cluster_size < self.dead_threshold
        if dead.any() and flat.shape[0] > 0:
            pick = torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)
            # RESTART DZIALA TAKZE PRZY UCZONYM KODEBOOKU. Pierwsza wersja
            # wylaczala go przy `learnable`, zeby uniknac wpisu w miejscu do
            # liscia wymagajacego gradientu -- i tym samym GWARANTOWALA zapad,
            # bo bez restartu kodebook moze tylko tracic pojemnosc (komentarz
            # wyzej: "collapse is absorbing"). Zmierzone na biegu `vqmanifold`
            # z wylaczonym restartem: `bits_used_frac` 0,2970 w epoce 1, czyli
            # perplexity rzedu 4 kodow na 128, i recon 0,8227 wobec 0,9952
            # w biegu z EMA. To byl artefakt mojej implementacji, nie wynik
            # o rozmaitosci.
            # `_update` jest pod `@torch.no_grad()`, ale wpis w miejscu do
            # `nn.Parameter` i tak wymaga `.data` -- inaczej PyTorch odmawia.
            nowe = project_to_hyperboloid(flat[pick][..., 1:])
            if isinstance(self.embed, nn.Parameter):
                self.embed.data[dead] = nowe
            else:
                self.embed[dead] = nowe
            self.cluster_size[dead] = 1.0

    @torch.no_grad()
    def perplexity(self):
        p = self.cluster_size / self.cluster_size.sum()
        return float(torch.exp(-(p * (p + 1e-10).log()).sum()))

    @torch.no_grad()
    def uzycie_empiryczne(self):
        """Uzycie kodow z BIEZACEGO batcha, PER KOMORKA -- nie z EMA.

        `perplexity()` i `code_used_frac` czytaja `cluster_size`, czyli EMA
        Z HISTORIA. Restart martwych kodow wpisuje tam `cluster_size[dead] =
        1.0` (linia w `_update`), a `code_used_frac` uzywa progu WZGLEDNEGO
        (1 % sredniej masy), wiec kod ozywiony wpisem i potem nigdy nie
        wybrany liczy sie jako zywy. ZMIERZONE: log pokazywal
        `code_used_frac ~ 1,0` przy 16 z 128 kodow realnie wybieranych,
        7,9 bitu z 105 zadeklarowanych i pieciu komorkach o masie modu
        1,0000 -- czyli poziom 0 byl uczona TABLICA POZYCYJNA, a alarm
        zapadu byl slepy przez cala historie tej rodziny.

        PER KOMORKA, nie po tensorze splaszczonym. Stan "15 kodow, po jednym
        na komorke" ma po splaszczeniu entropie log2(15) = 3,9 b i wyglada
        zdrowo; entropia PER KOMORKA jest wtedy DOKLADNIE 0. Falsyfikator
        jest per komorka, wiec i miara musi byc -- to ta sama pomylka, ktora
        czyni istniejacy `_czlon_entropii` slepym na ten stan.

        SUFITEM JEST BATCH: entropia rozkladu po `B` probkach nie przekroczy
        `log2(B)`, wiec przy batchu 96 maksimum to 6,58 b, a nie
        log2(128) = 7. Alarm na 2,0 b lezy daleko pod tym sufitem, i
        `emp_H_sufit` jest w raporcie, zeby nikt nie porownal z log2(K).
        """
        import math as _math
        idx = getattr(self, "indices", None)
        if idx is None:
            return {}
        idx = idx.detach()
        if idx.dim() != 2:
            return {}
        B, L = int(idx.shape[0]), int(idx.shape[1])
        K = int(self.codebook_size)
        h = torch.zeros(L, K, device=idx.device, dtype=torch.float32)
        h.scatter_add_(1, idx.t().long(),
                       torch.ones(L, B, device=idx.device, dtype=torch.float32))
        p = h / h.sum(dim=1, keepdim=True).clamp_min(1.0)
        # Maskowany OPERAND (`clamp_min`), nie wynik: `torch.where` na
        # `p * log(p)` daje NaN dla p = 0 i zabilo w tym repo caly bieg.
        H = -(p * torch.log2(p.clamp_min(1e-12))).sum(dim=1)
        zyw = h.sum(dim=0) > 0
        return {
            "emp_H_poz": round(float(H.mean()), 3),
            "emp_H_min": round(float(H.min()), 3),
            "emp_H_sufit": round(_math.log2(max(B, 2)), 3),
            "emp_uzyte": round(float(zyw.float().mean()), 4),
            "emp_uzyte_n": int(zyw.sum()),
            "emp_masa_modu": round(float(p.max(dim=1).values.mean()), 4),
        }

    @torch.no_grad()
    def radii(self):
        """Geodesic distance of each code from the origin.

        In a well-formed hierarchy this should *stratify*: general codes near
        the origin, specific ones far out.  A flat radius distribution means the
        hyperbolic structure is not being used and the model would do just as
        well in Euclidean space.
        """
        origin = torch.zeros_like(self.embed[:1])
        origin[0, 0] = 1.0
        return lorentz_distance(self.embed, origin.expand_as(self.embed))
