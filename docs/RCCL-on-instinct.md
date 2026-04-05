# RCCL on AMD Instinct — companion note

**Scope:** Single AMD Instinct MI300X VF — no multi-GPU numbers in this document.
**Purpose:** Systems-level analysis of RCCL vs NCCL, what limits collective performance
on Instinct hardware, and what would be measured with two or more GPUs.
**Related:** The main repo covers NCCL P2P on NVIDIA H200 with a real 2-node cluster.
This note is the AMD counterpart: same questions, different hardware path.

---

## 1. RCCL vs NCCL — API parity and key differences

RCCL (ROCm Communication Collectives Library) is AMD's equivalent to NVIDIA's NCCL.
The public API surface is intentionally compatible — most PyTorch distributed code that
works with NCCL switches to RCCL by setting the backend to `"nccl"` (PyTorch maps this
to RCCL on ROCm builds).

| Aspect | NCCL (NVIDIA) | RCCL (AMD) | Notes |
|--------|--------------|------------|-------|
| PyTorch backend string | `"nccl"` | `"nccl"` | Same string; PyTorch selects library at build time |
| Package | `libnccl` | `librccl` | Both in distro-standard GPU stacks |
| Init env vars | `NCCL_DEBUG`, `NCCL_SOCKET_IFNAME` | `RCCL_DEBUG`, `NCCL_SOCKET_IFNAME` | Most `NCCL_*` vars respected; RCCL-specific ones add `RCCL_` prefix |
| Intra-node transport | NVLink / NVSwitch | **XGMI** (Infinity Fabric) | Equivalent role; different protocol |
| Inter-node transport | RoCE v2 / InfiniBand | RoCE v2 / InfiniBand | Same fabric; RCCL uses UCX or built-in socket transport |
| Profiling | Nsight Systems `nvtx` | rocprof / `roctx` markers | `NCCL_ALGO` / `NCCL_PROTO` tuning vars work in RCCL |
| `nccl-tests` | Official NVIDIA repo | RCCL fork (`ROCmSoftwarePlatform/rccl-tests`) | Same binary interface; recompile with hipcc |

**Practical check on this VM:**
```bash
dpkg -l | grep -i rccl
# Expected: librccl2, librccl-dev (ROCm 6.4.1)

python3 -c "import torch; print(torch.distributed.is_nccl_available())"
# Expected: True  (RCCL exposed as nccl backend on ROCm PyTorch)
```

---

## 2. Intra-node transport: XGMI on MI300X

On a bare-metal MI300X node (8 GPUs), inter-GPU communication uses **XGMI** (the AMD
equivalent of NVLink). The `mi300x-amd-rocm-validation` repo documents measured RCCL
all-reduce bandwidth on an 8-GPU node.

On a **single VF** (this VM), there is no second GPU to form a peer pair — XGMI is not
applicable. This note focuses on the inter-node path and what single-GPU RCCL init
verifies.

| Topology | Transport | Bandwidth range |
|----------|-----------|-----------------|
| MI300X 8-GPU node (XGMI, 1-hop) | Infinity Fabric | ~200–400 GB/s bidirectional per pair (varies by RCCL algo) |
| MI300X → host CPU (PCIe) | PCIe Gen4/5 x16 | ~32–64 GB/s |
| Inter-node RoCE v2 (25 GbE) | Ethernet | ~3 GB/s |
| Inter-node RoCE v2 (200 GbE HDR) | Ethernet / IB | ~20–25 GB/s |

---

## 3. What breaks or degrades at scale on Instinct

### 3a. XGMI topology sensitivity

Unlike NVSwitch (full non-blocking crossbar), MI300X XGMI uses a ring-mesh topology
at the node level. `rocm-smi --showxgmi` reports the hop count between each GPU pair.
AllReduce algorithms that assume symmetric bandwidth (Ring, Tree) can see imbalance if
GPU pairs span a 2-hop XGMI path. Omniperf's topology view shows this; `NCCL_ALGO=TREE`
sometimes outperforms `RING` on non-uniform topologies.

### 3b. VF / virtualisation boundary

On a VF, the guest sees a single GPU. The XGMI fabric is not exposed cross-VF —
collective operations between two VFs on the same physical host would travel **PCIe →
host → PCIe** rather than XGMI. This is a significant bandwidth reduction (from
~300+ GB/s to ~32 GB/s). Bare-metal deployment is required to exploit XGMI for
multi-GPU collectives.

### 3c. RoCE v2 — same failure modes as NCCL

The RoCE v2 failure documented in this repo's main investigation
([`docs/roce-v2-investigation.md`](roce-v2-investigation.md)) applies identically to
RCCL inter-node AllReduce. `IBV_WC_RETRY_EXC_ERR` is a fabric / MTU / PFC configuration
issue, not a NCCL vs RCCL difference. The diagnostic steps are the same:

```bash
# Check RoCE link on AMD node
ibstat
ibv_devinfo | grep -E "port|state|mtu|link"

# Test connectivity (RCCL path)
RCCL_DEBUG=INFO python3 -m torch.distributed.launch \
  --nproc_per_node=1 --nnodes=2 \
  --node_rank=0 --master_addr=<node1-ip> \
  test_rccl_init.py
```

### 3d. `NCCL_SOCKET_IFNAME` and interface selection

RCCL honours `NCCL_SOCKET_IFNAME` for the socket/TCP fallback path. On AMD nodes with
multiple NICs (bond + management), setting this correctly is critical:

```bash
export NCCL_SOCKET_IFNAME=bond0   # or eth0, ib0, etc.
export NCCL_DEBUG=INFO             # prints selected transport
export RCCL_DEBUG=INFO             # RCCL-specific messages
```

---

## 4. Single-GPU RCCL init — what you can verify on 1 VF

Even without a second GPU, you can verify RCCL is installed and initialises correctly:

```python
import os, torch
import torch.distributed as dist

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

dist.init_process_group(backend="nccl", rank=0, world_size=1)
print("RCCL init OK — rank 0 of 1")
print("backend:", dist.get_backend())

# Single-rank allreduce (no-op but exercises the path)
t = torch.ones(4, device="cuda")
dist.all_reduce(t)
print("all_reduce result:", t)  # expected: tensor([1., 1., 1., 1.])

dist.destroy_process_group()
```

**Expected output on this VM:**
```
RCCL init OK — rank 0 of 1
backend: nccl
all_reduce result: tensor([1., 1., 1., 1.], device='cuda:0')
```

This confirms: RCCL package present, PyTorch ROCm backend wired correctly, GPU reachable
via RCCL transport layer.

---

## 5. What I would measure with 2+ MI300X GPUs

If a second VF or bare-metal node were available, the priority measurements would be:

| Experiment | Command | What it reveals |
|------------|---------|-----------------|
| XGMI peer bandwidth (intra-node) | `rccl-tests/all_reduce_perf -b 1G -e 4G -f 2 -g 2` | Raw XGMI bandwidth vs PCIe fallback |
| VF-to-VF vs bare-metal | Same test, VF pair vs bare-metal pair | Quantify virtualisation overhead on collectives |
| AllReduce algorithm sweep | `NCCL_ALGO=RING`, `TREE`, `COLLNET_DIRECT` | Optimal algorithm for MI300X XGMI topology |
| Inter-node TCP baseline | 2 nodes, `NCCL_P2P_DISABLE=1` | Same test as H200 TCP in main repo — establishes AMD TCP baseline |
| Inter-node RoCE v2 | 2 nodes with RoCE NIC | Confirm or reproduce `IBV_WC_RETRY_EXC_ERR` pattern; root-cause on AMD fabric |
| Bus bandwidth vs message size | Sweep 1 KB → 4 GB | Find inflection where RCCL switches from latency-optimised to bandwidth-optimised path |

---

## 6. Repo relationship

| Repo | Hardware | Content |
|------|----------|---------|
| `multi-node-nccl-p2p-benchmarks` (this repo) | 2× NVIDIA H200 (16 GPU) | Real NCCL AllReduce numbers, NVLink intra-node, RoCE failure investigation |
| `mi300x-amd-rocm-validation` | 8× AMD MI300X (bare metal) | RCCL AllReduce bandwidth, XGMI topology, peak-load thermal |
| This document | 1× AMD MI300X VF | RCCL/NCCL parity, VF constraints, measurement plan for multi-GPU |

---

## References

- [RCCL GitHub](https://github.com/ROCmSoftwarePlatform/rccl) — source, issue tracker
- [rccl-tests](https://github.com/ROCmSoftwarePlatform/rccl-tests) — benchmark suite (hipcc build)
- [ROCm docs: collective communications](https://rocm.docs.amd.com) — RCCL env vars, topology guide
- [MI300X XGMI topology](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) — hardware specs
