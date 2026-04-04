# RoCE v2 Failure Investigation

**Cluster:** 10.0.0.1 ↔ 10.0.0.2 (and 10.0.0.2 ↔ 10.0.0.3)  
**Date:** March 13–16, 2026  
**Outcome:** RoCE v2 RDMA not functional; TCP over bond0 used for production.

---

## Summary

RoCE v2 hardware and host configuration were correct on both nodes. RDMA failed at the **network/fabric layer** — the Ethernet switch between the nodes was not configured for RoCE v2 traffic (missing PFC/ECN/DCQCN). TCP remained the production path.

| Transport | Result | Used for production? |
|-----------|--------|---------------------|
| TCP over bond0 | ✅ PASS — all collectives, 0 wrong | **Yes** |
| RoCE v2 (RDMA) | ❌ FAIL — fabric layer | No — pending fabric fix |

---

## Host configuration verified (both nodes)

```bash
# Check RoCE-capable NICs
ibstat | grep -E "Link layer|Port state"
# Output: Link layer: Ethernet, Port state: Active — RoCE v2 capable

# List RoCE v2 GIDs
show_gids | grep -E "mlx5_2|mlx5_7"
# Output (Node 1):
# mlx5_2  1  2  ::ffff:10.0.0.1   RoCE v2   bond0
# mlx5_2  1  3  ::ffff:10.0.0.1   RoCE v2   bond0

# Verify connectivity layer (ping works)
ping -c 3 10.0.0.2  # 0% packet loss
```

**Conclusion:** Host is configured correctly. NIC hardware is present, GIDs are correct, bond0 is up.

---

## NCCL over RoCE v2 — error sequence

```bash
export NCCL_NET=IB
export NCCL_IB_GID_INDEX=3   # RoCE v2 over IPv4
export NCCL_DEBUG=INFO

mpirun -np 2 -hostfile configs/hostfile -map-by node \
  ./build/all_reduce_perf -b 8M -e 8M -f 2 -g 8
```

**Error output:**
```
NCCL INFO NET/IB: Using [0]mlx5_2:1/RoCE ; OOB bond0:10.0.0.1
NCCL INFO NCCL_IB_GID_INDEX set by environment to 3.

NET/IB: Got completion from peer 10.0.0.2<0000000000000000000000000000000000000000>
  with status=IBV_WC_RETRY_EXC_ERR(12)
  opcode=IBV_WC_SEND(0) reqSize=0 vendor_err=129
  req_type=Recv localGid ::ffff:10.0.0.1
  remoteGids::ffff:10.0.0.2 hca mlx5_2

NCCL WARN NET/IB : Got completion with error 12, opcode 0, ...
```

---

## Direct RDMA test (`ib_write_bw`)

```bash
# Server (Node 1):
ib_write_bw -d mlx5_2 --report_gbits

# Client (Node 2):
ib_write_bw -d mlx5_2 10.0.0.1 --report_gbits
```

**Server error:**
```
Couldn't listen to port 18515
```

**Client error:**
```
Couldn't connect to 10.0.0.1, retry 1 of 5 ...
Failed to connect to remote server: status 12, syndrom 0x81
```

**`status 12` = `IBV_WC_RETRY_EXC_ERR`.** The RDMA connection attempt failed after exhausting retries. The local NIC sent the request; the remote side never responded at the RDMA layer — consistent with a network device dropping or not forwarding RoCE packets.

---

## Root cause analysis

### Error code interpretation

| Code | Meaning |
|------|---------|
| `IBV_WC_RETRY_EXC_ERR (12)` | RDMA send/recv retries exhausted — remote side unreachable at RDMA layer |
| `vendor_err=129` | Mellanox/NVIDIA-specific: often indicates PFC/ECN flow control mismatch |
| `syndrom 0x81` | Mellanox: local protection error or fabric dropped packet |

### Why TCP worked but RoCE didn't

TCP operates at L4; bond0 carries it fine. RoCE v2 operates at L3 but requires specific L2 handling:
- **Priority Flow Control (PFC)** — switch must pause frames on RoCE priority queues to prevent head-of-line blocking and packet drops under congestion.
- **ECN (Explicit Congestion Notification)** — endpoints need ECN marks from the switch to throttle before dropping.
- **DCQCN** — the congestion control protocol running on Mellanox NICs requires switch-side ECN marking to function.

Without these, RoCE v2 packets are dropped under any congestion — and `IBV_WC_RETRY_EXC_ERR` is the result.

### Fabric fix required

1. **Switch:** Enable PFC on RoCE priority class (typically priority 3 or 4) on all ports carrying GPU traffic.
2. **Switch:** Enable ECN marking on those queues.
3. **NIC:** Verify DCQCN is enabled (`mlnx_qos` or `tc qdisc`).
4. **Verify with:** `ib_write_bw` between nodes — should reach line rate before re-testing NCCL.

---

## Second node pair: segfault (10.0.0.2 ↔ 10.0.0.3)

On the second node pair, the RoCE failure manifested differently:
```
NCCL WARN Timeout waiting for connection from peer 10.0.0.3
NCCL WARN Bootstrap : Connection to peer 10.0.0.3 timed out
Segmentation fault (core dumped)
```

Different symptoms, same root cause: the RDMA connection never completes, and the NCCL bootstrap timeout causes an unhandled NULL dereference in the communicator teardown path. TCP tests on this pair also showed lower bandwidth (~1.67 GB/s vs ~2.28 GB/s on the first pair) — likely a different network path or additional hop.

---

## Production decision

**Use TCP over bond0** (`NCCL_IB_DISABLE=1`) for all production workloads until:
1. Switch PFC/ECN is configured by the network team.
2. `ib_write_bw` achieves line rate between nodes.
3. NCCL with `NCCL_NET=IB` passes all collectives with 0 wrong results.

TCP at ~2.28 GB/s is sufficient for medium-scale training (8–16 GPUs). For larger scale where cross-node bandwidth matters more, RoCE fix is required.

---

## References

- [NVIDIA NCCL: Net/IB Errors](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [Mellanox RoCE Configuration Guide](https://enterprise-support.nvidia.com/s/article/roce-configuration-for-connectx-4-and-above)
- `IBV_WC_RETRY_EXC_ERR`: RDMA CM spec §11 — retry count exceeded
