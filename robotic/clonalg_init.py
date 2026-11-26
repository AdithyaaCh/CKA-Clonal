"""GPU batched, sample based initialisation for CKA-RL adaptation weights."""

from dataclasses import dataclass
import math
import time

import torch


@dataclass
class SearchConfig:
    antibody_length: int = 1000
    population: int = 50
    generations: int = 50
    selected: int = 10
    clone_beta: float = 2.0
    rho: float = 5.0
    replacement_fraction: float = 0.1
    projections: int = 256
    chunk_size: int = 16
    affinity: str = "swd"
    top_m: int = 1
    per_id_cap: int = 5000
    temperature: float = 1.0
    alpha_floor: float = 0.02


def weights_to_beta(weights, temperature, floor):
    k = weights.numel()
    if floor < 0 or k * floor >= 1 or temperature <= 0:
        raise ValueError("Require temperature > 0 and 0 <= K * alpha_floor < 1")
    weights = weights / weights.sum()
    return temperature * (floor + (1 - k * floor) * weights).log()


def beta_to_weights(beta, temperature):
    return torch.softmax(beta / temperature, dim=-1)


class VectorStores:
    """Sample and learned-alpha histories indexed by stable vector IDs."""

    def __init__(self, cap=5000):
        self.cap = cap
        self.samples = {}
        self.alphas = {}
        self.next_id = 0

    def add(self, samples):
        vector_id = self.next_id
        self.next_id += 1
        self.samples[vector_id] = samples[:self.cap].clone()
        return vector_id

    def record_alpha(self, task, values):
        self.alphas[task] = dict(values)

    def merge(self, left, right, generator=None):
        if left == right or left not in self.samples or right not in self.samples:
            raise ValueError("Merge requires two active distinct IDs")
        a, b = self.samples[left], self.samples[right]
        n = min(self.cap, 2 * min(len(a), len(b)))
        n_a = n // 2
        n_b = n - n_a
        ia = torch.randperm(len(a), device=a.device, generator=generator)[:n_a]
        ib = torch.randperm(len(b), device=b.device, generator=generator)[:n_b]
        merged = torch.cat([a[ia], b[ib]])
        del self.samples[left], self.samples[right]
        new_id = self.add(merged)
        for weights in self.alphas.values():
            weights[new_id] = weights.pop(left, 0.0) + weights.pop(right, 0.0)
        return new_id


def project(pool, probe, count, generator=None):
    direction = torch.randn(pool.shape[1], count, device=pool.device, generator=generator)
    direction = torch.nn.functional.normalize(direction, dim=0)
    return pool.float() @ direction, probe.float() @ direction


def projected_distance(pool_proj, probe_proj, indices, metric="swd", chunk_size=16):
    """Return one distance per antibody, with all gathers and sorting on device."""
    probe_sorted = probe_proj.sort(dim=0).values
    out = []
    for part in indices.split(chunk_size):
        chosen = pool_proj[part].sort(dim=1).values
        if metric == "swd":
            # Equal length empirical quantiles; interpolate when probe length differs.
            target = torch.nn.functional.interpolate(
                probe_sorted.T.unsqueeze(0), size=part.shape[1], mode="linear", align_corners=True
            ).squeeze(0).T
            d = (chosen - target.unsqueeze(0)).abs().mean(dim=(1, 2))
        elif metric == "ks":
            vals = chosen.permute(0, 2, 1).contiguous()
            reference = probe_sorted.T.unsqueeze(0).expand(len(part), -1, -1).contiguous()
            # searchsorted expects contiguous sorted sequences.
            at_probe = torch.searchsorted(vals, reference).float() / vals.shape[-1]
            at_pool = torch.searchsorted(reference, vals).float() / reference.shape[-1]
            grid_probe = torch.arange(1, reference.shape[-1] + 1, device=part.device) / reference.shape[-1]
            grid_pool = torch.arange(1, vals.shape[-1] + 1, device=part.device) / vals.shape[-1]
            d = torch.maximum(
                (at_probe - grid_probe).abs().amax(dim=-1),
                (at_pool - grid_pool).abs().amax(dim=-1),
            ).mean(dim=-1)
        else:
            raise ValueError(metric)
        out.append(d)
    return torch.cat(out)


def search(stores, probe, config, generator=None):
    """CLONALG over integer indices into immutable feature samples."""
    start = time.perf_counter()
    ids = list(stores.samples)
    if not ids or any(len(stores.samples[i]) == 0 for i in ids):
        raise ValueError("Every pool ID needs nonempty samples")
    device = probe.device
    pool = torch.cat([stores.samples[i].to(device).float() for i in ids])
    owner = torch.cat([torch.full((len(stores.samples[i]),), j, device=device, dtype=torch.long) for j, i in enumerate(ids)])
    projection, target = project(pool, probe, config.projections, generator)
    by_id = [torch.where(owner == j)[0] for j in range(len(ids))]
    length, size = config.antibody_length, config.population
    if size < len(ids):
        raise ValueError("Population must contain one pure antibody per ID")

    def fresh(count):
        # Squared standard normals are Gamma(1/2) draws with a common scale.
        # Multinomial normalizes the rows, giving the same symmetric Dirichlet mixture.
        composition = torch.randn((count, len(ids)), device=device, generator=generator).square()
        chosen_ids = torch.multinomial(composition, length, replacement=True, generator=generator)
        rows = torch.empty((count, length), device=device, dtype=torch.long)
        for j, source in enumerate(by_id):
            mask = chosen_ids == j
            rows[mask] = source[torch.randint(len(source), (int(mask.sum()),), device=device, generator=generator)]
        return rows

    pure = torch.stack([source[torch.randint(len(source), (length,), device=device, generator=generator)] for source in by_id])
    population = torch.cat([pure, fresh(size - len(ids))])
    distance = projected_distance(projection, target, population, config.affinity, config.chunk_size)
    d_ref = distance.median().clamp_min(1e-8)
    curve = []
    for _ in range(config.generations):
        affinity = torch.exp(-distance / d_ref)
        winners = affinity.topk(min(config.selected, size)).indices
        parents = population[winners]
        parent_affinity = affinity[winners]
        clones = []
        for rank, parent in enumerate(parents, 1):
            count = max(1, round(config.clone_beta * size / rank))
            batch = parent.expand(count, -1).clone()
            rate = math.exp(-config.rho * float(parent_affinity[rank - 1]))
            change = torch.rand((count, length), device=device, generator=generator) < rate
            same = torch.rand((count, length), device=device, generator=generator) >= rate
            old_ids = owner[batch]
            new_ids = torch.where(same, old_ids, (old_ids + torch.randint(1, len(ids), (count, length), device=device, generator=generator)) % len(ids)) if len(ids) > 1 else old_ids
            for j, source in enumerate(by_id):
                mask = change & (new_ids == j)
                batch[mask] = source[torch.randint(len(source), (int(mask.sum()),), device=device, generator=generator)]
            batch[0] = parent  # elitism within each clone group
            clones.append(batch)
        candidates = torch.cat([population, *clones])
        candidate_distance = projected_distance(projection, target, candidates, config.affinity, config.chunk_size)
        best = candidate_distance.topk(size, largest=False).indices
        population, distance = candidates[best], candidate_distance[best]
        replacements = min(size - 1, round(size * config.replacement_fraction))
        if replacements:
            population[-replacements:] = fresh(replacements)
            distance[-replacements:] = projected_distance(projection, target, population[-replacements:], config.affinity, config.chunk_size)
        curve.append(float(torch.exp(-distance.min() / d_ref)))
    best = distance.topk(config.top_m, largest=False).indices
    fractions = torch.stack([torch.bincount(owner[row], minlength=len(ids)).float() / length for row in population[best]]).mean(dim=0)
    return {"ids": ids, "weights": fractions, "distance": distance.min().item(), "curve": curve, "seconds": time.perf_counter() - start}


def selftest(device="cpu"):
    torch.manual_seed(42)
    g = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(200, 8, device=device, generator=g)
    p, q = project(x, x, 64, g)
    ix = torch.arange(len(x), device=device).expand(2, -1)
    same = projected_distance(p, q, ix).max().item()
    shifted = projected_distance(p, q + 3, ix).min().item()
    assert same < 1e-5 and shifted > 1, (same, shifted)
    stores = VectorStores(256)
    centers = [-6.0, -2.0, 2.0, 6.0]
    for center in centers:
        stores.add(torch.randn(256, 4, device=device, generator=g) * 0.25 + center)
    probe = torch.cat([torch.randn(140, 4, device=device, generator=g) * 0.25 - 6,
                       torch.randn(60, 4, device=device, generator=g) * 0.25 - 2])
    before = {i: t.clone() for i, t in stores.samples.items()}
    cfg = SearchConfig(antibody_length=200, population=24, generations=30, selected=5, projections=32, chunk_size=8)
    result = search(stores, probe, cfg, g)
    assert all(torch.equal(before[i], stores.samples[i]) for i in before)
    assert abs(float(result["weights"][0]) - 0.7) < 0.2, result["weights"]
    assert abs(float(result["weights"][1]) - 0.3) < 0.2, result["weights"]
    assert float(result["weights"][2:].sum()) < 0.15, result["weights"]
    pool, target = project(torch.cat(list(stores.samples.values())), probe, 32, g)
    offsets = [0, 256, 512, 768]
    pure = torch.stack([torch.arange(start, start + 200, device=device) for start in offsets])
    assert projected_distance(pool, target, pure).argmin().item() == 0
    w = torch.tensor([0.7, 0.3], device=device)
    beta = weights_to_beta(w, 0.7, 0.02)
    assert torch.allclose(beta_to_weights(beta, 0.7), 0.02 + 0.96 * w)
    stores.record_alpha(1, {0: 0.7, 1: 0.3})
    merged = stores.merge(0, 1, g)
    assert merged == 4 and len(stores.samples) == 3 and set(stores.samples) == {2, 3, 4}
    assert abs(sum(stores.alphas[1].values()) - 1) < 1e-6
    return result
