"""Peer-to-peer copy bandwidth between GPU pairs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class P2PResult:
    src: int
    dst: int
    gbps: float


def measure_p2p_gbps(src_gpu: int, dst_gpu: int, size_gb: float, iterations: int, warmup: int) -> P2PResult | None:
    import torch

    if src_gpu == dst_gpu:
        return P2PResult(src_gpu, dst_gpu, 0.0)

    # Peer access API varies; attempt copy and fail soft if P2P disabled.
    try:
        if not torch.cuda.can_device_peer_access(dst_gpu, src_gpu):
            return None
    except Exception:
        pass

    n_elements = int(size_gb * 1024**3 / 4)
    src_tensor = torch.randn(n_elements, device=f"cuda:{src_gpu}")
    dst_tensor = torch.empty(n_elements, device=f"cuda:{dst_gpu}")

    for _ in range(warmup):
        dst_tensor.copy_(src_tensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    try:
        start.record()
        for _ in range(iterations):
            dst_tensor.copy_(src_tensor)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    except RuntimeError:
        return None
    total_gb = size_gb * iterations
    gbps = total_gb / (elapsed_ms / 1000.0)
    return P2PResult(src_gpu, dst_gpu, float(gbps))
