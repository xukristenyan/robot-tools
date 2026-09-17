"""Deterministic batched FPS using Torch operations supported on Blackwell.

Same farthest-first algorithm and index-zero start as torch_cluster FPS.
Tied distances may choose a different equivalent point than its CUDA
reduction. Sparse clouds repeat the sampled order to the requested size.
"""

import torch


@torch.no_grad()
def native_fps(data, number, *, random_start=False, return_idx=False):
    batch_size, count, _ = data.shape
    if count == 0 or number is None or number <= 0:
        raise ValueError("FPS needs a nonempty cloud and a positive sample count")
    number = int(number)
    unique_count = min(count, number)
    indices = torch.empty((batch_size, unique_count), device=data.device, dtype=torch.long)
    distances = torch.full((batch_size, count), float("inf"), device=data.device, dtype=data.dtype)
    farthest = (
        torch.randint(count, (batch_size,), device=data.device)
        if random_start
        else torch.zeros(batch_size, device=data.device, dtype=torch.long)
    )
    batch = torch.arange(batch_size, device=data.device)
    for i in range(unique_count):
        indices[:, i] = farthest
        center = data[batch, farthest, None, :]
        distances = torch.minimum(distances, ((data - center) ** 2).sum(dim=-1))
        farthest = distances.argmax(dim=-1)
    if number > count:
        indices = indices.repeat(1, (number + count - 1) // count)[:, :number]
    sampled = data.gather(1, indices[..., None].expand(-1, -1, data.shape[-1]))
    return (sampled, indices) if return_idx else sampled
