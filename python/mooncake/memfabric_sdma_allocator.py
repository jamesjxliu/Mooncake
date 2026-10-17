"""Opt-in process-local MemFabric VMM DRAM allocator for SDMA buffers.

Initialize after binding the NPU and before native Store.setup(). The shared
pool and all tensors live until process exit; close Store clients before
explicitly uninitializing the MemFabric offload module.
"""

import os
import socket
import threading

_lock = threading.RLock()
_pool = None


def initialize_sdma_host_pool():
    global _pool
    import torch
    import torch_npu
    from memfabric_hybrid import offload

    if os.environ.get("MC_MEMFABRIC_SDMA_OFFLOAD_POOL") != "1":
        raise RuntimeError("MC_MEMFABRIC_SDMA_OFFLOAD_POOL=1 is required")
    if os.environ.get("MF_PROTOCOL") != "device_sdma":
        raise RuntimeError("The VMM host pool requires MF_PROTOCOL=device_sdma")
    device = torch.npu.current_device()
    with _lock:
        if _pool is not None:
            if _pool["device"] != device:
                raise RuntimeError("One SDMA host pool per NPU worker is supported")
            return dict(_pool)
        gib = 1024**3
        capacity = int(os.environ.get("MC_MEMFABRIC_SDMA_POOL_BYTES", 8 * gib))
        if capacity <= 0:
            raise ValueError("MC_MEMFABRIC_SDMA_POOL_BYTES must be positive")
        capacity = ((capacity + gib - 1) // gib) * gib
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        config = offload.OffloadConfig()
        config.device_id = device
        config.reserve_size = capacity
        config.alloc_size = capacity
        config.world_size = 1
        config.rank_id = 0
        config.scene = offload.Scene.SHARED
        config.store_url = f"tcp://127.0.0.1:{port}"
        rc = offload.initialize(config)
        if rc != 0:
            raise RuntimeError(f"MemFabric VMM DRAM pool initialize failed: rc={rc}")
        _pool = {"device": device, "capacity": capacity, "store_url": config.store_url}
        return dict(_pool)


def empty(sizes, dtype):
    from memfabric_hybrid import offload

    initialize_sdma_host_pool()
    return offload.empty(sizes, dtype=dtype)
