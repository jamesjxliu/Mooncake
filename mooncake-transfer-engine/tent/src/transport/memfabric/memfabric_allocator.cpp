#include "memfabric_allocator.h"

#include <dlfcn.h>
#include <glog/logging.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>

namespace mooncake {
namespace {

bool useOffloadPool(const std::string& protocol) {
    const char* enabled = std::getenv("MC_MEMFABRIC_SDMA_OFFLOAD_POOL");
    const char* transport = std::getenv("MF_PROTOCOL");
    return protocol == "ascend" && enabled && std::strcmp(enabled, "1") == 0 &&
           transport && std::strcmp(transport, "device_sdma") == 0;
}

struct OffloadApi {
    using Alloc = uint64_t (*)(uint64_t, uint64_t);
    using Free = void (*)(uint64_t, uint64_t);
    void* library = nullptr;
    Alloc alloc = nullptr;
    Free free = nullptr;
};

const OffloadApi& offloadApi() {
    static const OffloadApi api = [] {
        OffloadApi result;
        result.library = dlopen("libmf_acc_offload.so", RTLD_NOW | RTLD_GLOBAL);
        if (!result.library) {
            LOG(ERROR) << "Cannot load MemFabric SDMA DRAM allocator: " << dlerror();
            return result;
        }
        result.alloc = reinterpret_cast<OffloadApi::Alloc>(
            dlsym(result.library, "offload_malloc"));
        result.free = reinterpret_cast<OffloadApi::Free>(
            dlsym(result.library, "offload_free"));
        return result;
    }();
    return api;
}

std::mutex pool_mutex;
std::unordered_set<void*> pool_allocations;

}  // namespace

void* memfabric_allocate_memory(size_t total_size,
                                const std::string& protocol) {
    if (useOffloadPool(protocol)) {
        const auto& api = offloadApi();
        if (!api.alloc || !api.free) {
            LOG(ERROR) << "MemFabric offload_malloc/offload_free is unavailable";
            return nullptr;
        }
        void* buffer = reinterpret_cast<void*>(api.alloc(total_size, 0));
        if (!buffer) {
            LOG(ERROR) << "MemFabric SDMA DRAM allocation failed, bytes="
                       << total_size << "; initialize the process-local pool first";
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(pool_mutex);
        pool_allocations.insert(buffer);
        return buffer;
    }
    void* buffer = nullptr;
    auto ret = aclrtMallocHost(&buffer, total_size);
    if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtMallocHost failed: " << ret;
        return nullptr;
    }
    return buffer;
}

void* memfabric_allocate_memory_best_effort(size_t target_size,
                                            const std::string& protocol,
                                            size_t* actual_size,
                                            MemFabricHostAllocFn host_alloc,
                                            size_t host_alignment,
                                            bool defer_hugetlb_population) {
    if (actual_size == nullptr) {
        LOG(ERROR)
            << "memfabric_allocate_memory_best_effort: actual_size is null";
        return nullptr;
    }
    *actual_size = 0;
    void* ptr = !useOffloadPool(protocol) && host_alloc
                    ? host_alloc(target_size, host_alignment,
                                 defer_hugetlb_population)
                    : memfabric_allocate_memory(target_size, protocol);
    if (ptr) *actual_size = target_size;
    return ptr;
}

void memfabric_free_memory(const std::string& protocol, void* ptr) {
    {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (pool_allocations.erase(ptr)) {
            offloadApi().free(reinterpret_cast<uint64_t>(ptr), 0);
            return;
        }
    }
    if (protocol == "ascend") aclrtFreeHost(ptr);
}

}  // namespace mooncake
