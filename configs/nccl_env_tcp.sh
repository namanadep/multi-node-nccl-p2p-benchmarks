#!/bin/bash
# NCCL environment for TCP production path (NCCL_IB_DISABLE=1)
# Source this before running mpirun or torchrun on multi-node jobs.
# Use when RoCE/InfiniBand is not available or not yet validated.

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0        # adjust to your inter-node interface
export NCCL_DEBUG=WARN                 # set to INFO for verbose topology logs
export NCCL_P2P_DISABLE=0             # keep intra-node NVLink P2P enabled
export NCCL_SHM_DISABLE=0             # keep shared memory for intra-node

echo "NCCL TCP environment set:"
echo "  NCCL_IB_DISABLE=1  (RDMA disabled)"
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
