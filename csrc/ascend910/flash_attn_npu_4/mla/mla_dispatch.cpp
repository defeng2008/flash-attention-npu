/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#include "mla_dispatch.h"

namespace flash_attn::mla {
thread_local MlaBatchLaunchState* activeMlaLaunch = nullptr;
}

extern "C" aclError __real_aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg*,
                                                         void*, size_t, aclrtPlaceHolderInfo*, size_t);
// CANN 9.2.0's Bisheng ASC launcher emits aclrtLaunchKernelWithArgsArray for the
// `<<<>>>` launch instead of aclrtLaunchKernelWithHostArgs (CANN 9.0/9.1). Wrap
// both so one thread-local scope still forces ScheduleMode::BATCH_MODE.
extern "C" aclError __real_aclrtLaunchKernelWithArgsArray(void*, uint32_t, aclrtStream, aclrtLaunchKernelCfg*, void**);

namespace {
// Copy every caller attribute except the scheduling mode, then append the fixed
// MLA ScheduleMode::BATCH_MODE. CANN 9.2.0 grew the attribute id set to nine
// (ACL_RT_LAUNCH_KERNEL_ATTR_ENABLE_PROFILING = 9), so size the buffer above that.
constexpr size_t kMlaLaunchAttrCapacity = 16;
aclError MlaAppendBatchMode(aclrtLaunchKernelCfg* config, aclrtLaunchKernelAttr* attrs, size_t& count)
{
    count = 0;
    if (config) {
        if (config->numAttrs > kMlaLaunchAttrCapacity - 1 || (config->numAttrs && !config->attrs)) {
            return ACL_ERROR_INVALID_PARAM;
        }
        for (size_t i = 0; i < config->numAttrs; ++i) {
            if (config->attrs[i].id != ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE) {
                attrs[count++] = config->attrs[i];
            }
        }
    }
    attrs[count].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attrs[count++].value.schemMode = 1; // fixed CANN MLA ScheduleMode::BATCH_MODE
    return ACL_SUCCESS;
}
} // namespace

extern "C" __attribute__((visibility("hidden"))) aclError __wrap_aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle function, uint32_t blocks, aclrtStream stream, aclrtLaunchKernelCfg* config, void* args,
    size_t argsSize, aclrtPlaceHolderInfo* placeholders, size_t placeholderCount)
{
    auto* state = flash_attn::mla::activeMlaLaunch;
    if (!state) {
        return __real_aclrtLaunchKernelWithHostArgs(function, blocks, stream, config, args, argsSize, placeholders,
                                                    placeholderCount);
    }
    ++state->calls;
    aclrtLaunchKernelAttr attrs[kMlaLaunchAttrCapacity]{};
    size_t count = 0;
    const auto error = MlaAppendBatchMode(config, attrs, count);
    if (error != ACL_SUCCESS) {
        state->error = error;
        return state->error;
    }
    aclrtLaunchKernelCfg batchConfig{attrs, count};
    state->error = __real_aclrtLaunchKernelWithHostArgs(function, blocks, stream, &batchConfig, args, argsSize,
                                                        placeholders, placeholderCount);
    return state->error;
}

extern "C" __attribute__((visibility("hidden"))) aclError __wrap_aclrtLaunchKernelWithArgsArray(
    void* function, uint32_t blocks, aclrtStream stream, aclrtLaunchKernelCfg* config, void** args)
{
    auto* state = flash_attn::mla::activeMlaLaunch;
    if (!state) {
        return __real_aclrtLaunchKernelWithArgsArray(function, blocks, stream, config, args);
    }
    ++state->calls;
    aclrtLaunchKernelAttr attrs[kMlaLaunchAttrCapacity]{};
    size_t count = 0;
    const auto error = MlaAppendBatchMode(config, attrs, count);
    if (error != ACL_SUCCESS) {
        state->error = error;
        return state->error;
    }
    aclrtLaunchKernelCfg batchConfig{attrs, count};
    state->error = __real_aclrtLaunchKernelWithArgsArray(function, blocks, stream, &batchConfig, args);
    return state->error;
}
