/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#include "mla_layout.h"
#include "mla_metadata.h"
#include <algorithm>
#include "mla_forward.h"
#include "kernel_operator.h"
#include "mla_dispatch.h"
#include "mla_metadata_consumer.h"
#include "mla_split.h"
#include "mla_tiling.h"
#include "mla_validation.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "runtime/rt_ffts.h"
#include <c10/core/DeviceGuard.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

extern __global__ __aicpu__ uint32_t ComputeMlaMetadata(void* args);

namespace flash_attn::mla {
namespace {
template <typename Launch>
void RunMlaLaunch(const char* name, Launch&& launch)
{
    if (std::getenv("ASCEND_LAUNCH_BLOCKING") || std::getenv("TORCH_NPU_SANITIZER")) {
        at_npu::native::OpCommand::RunOpApiV2(name, std::forward<Launch>(launch));
        return;
    }
    // Drain pending host submissions (including length updates) before using
    // the ASC launcher directly. stream() does not wait for device completion;
    // subsequent kernels and allocator reuse remain ordered on this stream.
    TORCH_CHECK(c10_npu::getCurrentNPUStream().stream() != nullptr, "MLA: host submission queue failed");
    RECORD_FUNCTION(name, std::vector<c10::IValue>({}));
    TORCH_CHECK(launch() == ACL_SUCCESS, "MLA: causal kernel submission failed");
}

MlaPlatformInfo DensePlatform()
{
    const char* soc = aclrtGetSocName();
    // CANN names the 910C family Ascend910_93; both families use the 2201
    // kernels. Keep resource validation below for each actual device/partition.
    TORCH_CHECK(soc && (std::strncmp(soc, "Ascend910B", 10) == 0 || std::strncmp(soc, "Ascend910_93", 12) == 0),
                "MLA forward requires an Ascend 910B/C device");
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform, "MLA: platform information is unavailable");
    MlaPlatformInfo p{};
    p.aicNum = platform->GetCoreNumAic();
    p.aivNum = platform->GetCoreNumAiv();
    TORCH_CHECK(p.aicNum > 0 && p.aicNum <= kMlaMaxAic && p.aivNum == 2 * p.aicNum,
                "MLA: this dense launch requires validated C1V2 resources");
    using Mem = platform_ascendc::CoreMemType;
    platform->GetCoreMemSize(Mem::L1, p.localMemory.l1Bytes);
    platform->GetCoreMemSize(Mem::UB, p.localMemory.ubBytes);
    platform->GetCoreMemSize(Mem::L0_A, p.localMemory.l0aBytes);
    platform->GetCoreMemSize(Mem::L0_B, p.localMemory.l0bBytes);
    platform->GetCoreMemSize(Mem::L0_C, p.localMemory.l0cBytes);
    p.libApiWorkspaceBytes = platform->GetLibApiWorkSpaceSize();
    for (uint32_t c = 1; c <= p.aicNum; ++c) {
        p.blockDims[c - 1] = platform->CalcTschBlockDim(c * 2, c, c * 2);
    }
    return p;
}

MlaPlatformInfo DevicePlatform(c10::DeviceIndex device, bool fullLaunch)
{
    // These chip resources are immutable. Keep each host thread/device's
    // validated descriptor separate; no tensor, length or schedule is cached.
    static thread_local std::unordered_map<c10::DeviceIndex, MlaPlatformInfo> platforms;
    auto found = platforms.find(device);
    if (found == platforms.end()) {
        found = platforms.emplace(device, DensePlatform()).first;
    }
    auto p = found->second;
    if (fullLaunch) {
        p.blockDims.fill(p.blockDims[p.aicNum - 1]);
    }
    return p;
}

template <typename View = MlaBsndTensorView>
View PagedTensorView(const at::Tensor& t)
{
    View view{};
    for (int64_t i = 0; i < t.dim(); ++i) {
        view.sizes[i] = t.size(i);
        view.strides[i] = t.stride(i);
    }
    view.storageOffset = t.storage_offset();
    view.storageBytes = t.storage().nbytes();
    return view;
}

// Validate the entire accessible span, not numel() for a strided view. Empty
// descriptors have no address requirement; Q-used presence is restored below.
MlaPagedMetadataArray PagedTensorArray(const at::Tensor& t, uint64_t span)
{
    if (span == 0) {
        return {};
    }
    const uint64_t elements = t.storage().nbytes() / t.element_size();
    TORCH_CHECK(t.storage_offset() >= 0 && uint64_t(t.storage_offset()) <= elements &&
                    span <= elements - uint64_t(t.storage_offset()),
                "MLA: paged tensor view exceeds storage bounds");
    const auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK(address != 0 && address % t.element_size() == 0 && span <= (UINTPTR_MAX - address) / t.element_size(),
                "MLA: tensor address overflow");
    return {address, span};
}

MlaPlatformInfo ShapePlatform()
{
    MlaPlatformInfo p{};
    p.aicNum = kMlaMaxAic;
    p.aivNum = 2 * p.aicNum;
    p.localMemory = {512 * 1024, 256 * 1024, 64 * 1024, 64 * 1024, 128 * 1024};
    for (uint32_t i = 0; i < p.aicNum; ++i)
        p.blockDims[i] = i + 1;
    return p;
}

uint64_t TensorAddress(const at::Tensor& tensor)
{
    return reinterpret_cast<uint64_t>(tensor.data_ptr());
}
void RecordInput(const at::Tensor& tensor, const c10_npu::NPUStream& stream)
{
    if (tensor.numel())
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}
} // namespace

namespace {
MlaContext MakeContext(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t valueDim,
                       at::ScalarType dtype, int64_t splits, int64_t left, int64_t right, bool causal, uint32_t flags,
                       const MlaPlatformInfo& platform, uint32_t layout, int64_t totalQ, int64_t totalK,
                       int64_t pageSize, bool hasSoftcap, std::optional<bool> pack, int64_t margin, MlaLayout& l)
{
    TORCH_CHECK(FitsDenseShape(0, sq, heads, sk) && batch >= 0 && uint64_t(batch) <= UINT32_MAX && kvHeads > 0 &&
                    heads % kvHeads == 0 && splits >= 0 && splits <= 256 && totalQ >= 0 && totalQ <= INT32_MAX &&
                    totalK >= 0 && totalK <= INT32_MAX && pageSize >= 0 && pageSize <= 1024 &&
                    (valueDim == 128 || valueDim == 512),
                "MLA: shape/split bounds exceeded");
    MlaGqaResources resources{};
    TORCH_CHECK(ResolveGqaResources(heads, kvHeads, {pack ? int64_t(*pack) : -1, margin}, platform, resources) ==
                    MlaScheduleStatus::Success,
                "MLA: invalid pack_gqa or sm_margin; sm_margin must leave at least one physical AIC");
    TORCH_CHECK(resources.platform.blockDims[resources.platform.aicNum - 1] == resources.platform.aicNum,
                "MLA: unsupported C1V2 blockDim mapping");
    MlaWindow window{};
    TORCH_CHECK(NormalizeWindow({sq, sk, left, right, causal}, window) == MlaCausalStatus::Success,
                "MLA: window bounds exceeded");
    MlaContext c{uint32_t(batch),
                 uint32_t(sq),
                 uint32_t(sk),
                 uint32_t(heads),
                 uint32_t(kvHeads),
                 uint32_t(splits),
                 uint32_t(dtype == at::kHalf ? MlaMetadataDtype::Fp16 : MlaMetadataDtype::Bf16),
                 platform.aicNum,
                 uint32_t(margin),
                 uint32_t(resources.packed),
                 layout,
                 uint32_t(totalQ),
                 uint32_t(totalK),
                 uint32_t(pageSize),
                 flags,
                 uint32_t(valueDim),
                 uint32_t(hasSoftcap),
                 {uint32_t(window.kind), window.left, window.right}};
    TORCH_CHECK(MlaRecordLayout(c, l), "MLA: metadata shape bounds or workspace/accumulator overflow");
    TORCH_CHECK(l.bytes <= uint64_t(INT64_MAX) - platform.libApiWorkspaceBytes, "MLA: workspace overflow");
    // Reuse the accepted C1V2 resource check. Platform capacities exclude a
    // runtime reserve, so nominal chip capacities are not valid lower bounds.
    MlaSchedule resourceCheck{};
    TORCH_CHECK(BuildDenseSchedule({1, 1, 1, 1}, platform, resourceCheck) == MlaScheduleStatus::Success,
                "MLA: invalid C1V2 resources");
    // Dynamic CANN M can be 128..256. Bound storage for every tile without
    // copying schedules or synchronizing allocation/used lengths to the host.
    if (valueDim == 512) {
        MlaLayout small{};
        TORCH_CHECK(MlaRecordLayout(c, small, 128, l.n), "MLA: workspace bounds exceeded");
        const uint64_t rows = small.groups * small.capacity * 256;
        const bool direct = splits == 1 && MlaNativeCompressed(c);
        const uint64_t reserve = MlaNativeCompressed(c) && !splits ? 2ULL * platform.aicNum * 256 : rows;
        const uint64_t bound = direct ? l.scratch : l.outOffset + reserve * (512 + 16) * 4;
        TORCH_CHECK(bound <= uint64_t(INT64_MAX) - platform.libApiWorkspaceBytes, "MLA: workspace overflow");
        if (bound > l.bytes)
            l.bytes = bound;
    }
    c.workspaceBytes = l.bytes;
    return c;
}
} // namespace

std::vector<at::Tensor> Forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                const std::optional<at::Tensor>& qv, int64_t splits, int64_t left, int64_t right,
                                bool causal, float scale, const std::optional<at::Tensor>& usedQ,
                                const std::optional<at::Tensor>& usedK, const std::optional<at::Tensor>& supplied,
                                const std::optional<at::Tensor>& cuQ, const std::optional<at::Tensor>& cuK,
                                int64_t maxQ, int64_t maxK, const std::optional<at::Tensor>& pageTable, float softcap,
                                std::optional<bool> pack, int64_t margin)
{
    const bool tnd = bool(cuQ);
    TORCH_CHECK(!supplied || !qv || !cuK || margin || (pack && !*pack) || splits != 1 || softcap > 0,
                "MLA: compact TND explicit metadata is unsupported");
    TORCH_CHECK(!usedQ || !supplied || (tnd && (!qv || ((margin || (pack && !*pack)) && cuK))),
                "MLA: Q-used with explicit metadata is not implemented yet");
    if (cuK)
        CheckTndInputs(q, k, *cuQ, *cuK, maxQ, maxK);
    else if (tnd && !pageTable)
        CheckTndBsndForwardInputs(q, k, *cuQ, maxQ, usedK, std::nullopt);
    else if (tnd) {
        TORCH_CHECK(cuQ->dim() == 1 && cuQ->numel() >= 1, "MLA: cu_seqlens_q must have shape [batch_size + 1]");
        CheckMetadataQueryPrefix(*cuQ, cuQ->numel() - 1, q.device());
    }
    const int64_t batch = tnd ? cuQ->numel() - 1 : q.size(0);
    CheckUsedInputs(q, batch, usedQ, usedK);
    MlaPagedCacheLayout paged{};
    if (pageTable) {
        TORCH_CHECK(pageTable->dim() == 2 && pageTable->size(0) == batch && pageTable->scalar_type() == at::kInt &&
                        pageTable->device() == q.device(),
                    "MLA: page_table must be same-device int32[batch_size, max_pages]");
        TORCH_CHECK(BuildPagedCacheLayout(PagedTensorView<MlaPageTableView>(*pageTable), PagedTensorView(k),
                                          PagedTensorView(v), paged, !qv) == MlaLayoutStatus::Success,
                    "MLA: paged cache exceeds shape, stride, storage or signed byte bounds");
    }
    const int64_t kvCapacity = pageTable ? int64_t(paged.logicalCapacity) : cuK ? maxK : k.size(1);
    const auto currentK = !cuK && supplied && !usedK
                              ? std::optional<at::Tensor>(at::full({batch}, kvCapacity, q.options().dtype(at::kInt)))
                              : usedK;
    const auto platform = q.is_meta() ? ShapePlatform() : DevicePlatform(q.device().index(), false);
    MlaLayout l{};
    const auto c =
        MakeContext(batch, tnd ? maxQ : q.size(1), kvCapacity, q.size(-2), k.size(-2), v.size(-1), q.scalar_type(),
                    splits, left, right, causal, uint32_t(bool(usedQ)) | (uint32_t(bool(currentK)) << 1), platform,
                    pageTable ? (tnd ? 4 : 3)
                    : cuK     ? 1
                    : tnd     ? 2
                              : 0,
                    tnd ? q.size(0) : 0, cuK ? k.size(0) : 0, pageTable ? k.size(1) : 0, softcap > 0, pack, margin, l);
    if (supplied)
        CheckMetadataBuffer(*supplied, q.device(), int64_t(MlaUnifiedBytes(l)), "MLA: scheduler_metadata");
    const int64_t b = c.batch, sq = c.sq, h = c.heads, dim = c.valueDim, capacity = splits ? splits : 26;
    auto out = tnd ? at::empty({c.totalQ, h, dim}, q.options()) : at::empty({b, sq, h, dim}, q.options());
    auto lse = tnd ? at::empty({h, c.totalQ}, q.options().dtype(at::kFloat))
                   : at::empty({b, h, sq}, q.options().dtype(at::kFloat));
    const auto emptyOptions = at::TensorOptions().device(q.device());
    auto accum = splits == 1 ? at::empty({0}, emptyOptions)
                 : tnd       ? at::empty({capacity, h, c.totalQ, dim}, q.options().dtype(at::kFloat))
                             : at::empty({capacity, b, h, sq, dim}, q.options().dtype(at::kFloat));
    auto accumLse = splits == 1 ? at::empty({0}, emptyOptions)
                    : tnd       ? at::empty({capacity, h, c.totalQ}, q.options().dtype(at::kFloat))
                                : at::empty({capacity, b, h, sq}, q.options().dtype(at::kFloat));
    if (q.is_meta())
        return {out, lse, accum, accumLse};
    const uint64_t checkedOffset = (l.bytes + 63) / 64 * 64;
    auto workspace =
        at::empty({int64_t(checkedOffset + sizeof(MlaCheckedAutoSplitSchedule) + platform.libApiWorkspaceBytes)},
                  q.options().dtype(at::kByte));
    auto metadata = supplied ? *supplied : at::empty({int64_t(MlaUnifiedBytes(l))}, q.options().dtype(at::kByte));
    MlaPagedCacheInputs cache{};
    if (pageTable)
        cache = {paged, PagedTensorArray(*pageTable, paged.tableSpanElements), PagedTensorArray(k, paged.k.spanElements),
                 PagedTensorArray(v, paged.v.spanElements)};
    auto stream = c10_npu::getCurrentNPUStream();
    // Local outputs and scratch stay on their allocation stream and remain captured below.
    for (const auto& tensor : {q, k, v})
        RecordInput(tensor, stream);
    if (supplied)
        RecordInput(*supplied, stream);
    for (const auto& tensor : {qv, cuQ, cuK, usedQ, usedK, pageTable})
        if (tensor)
            RecordInput(*tensor, stream);
    MlaLaunchArgs a{};
    a.stream = stream.stream(false);
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE, "MLA: FFTS address unavailable");
    a.q = TensorAddress(q);
    a.qv = qv ? TensorAddress(*qv) : 0;
    a.k = TensorAddress(k);
    a.v = TensorAddress(v);
    a.out = TensorAddress(out);
    a.lse = TensorAddress(lse);
    a.accum = TensorAddress(accum);
    a.accumLse = TensorAddress(accumLse);
    a.workspace = TensorAddress(workspace) + platform.libApiWorkspaceBytes;
    a.metadata = TensorAddress(metadata);
    a.cuQuery = cuQ ? TensorAddress(*cuQ) : 0;
    a.cuKv = cuK ? TensorAddress(*cuK) : 0;
    a.usedQuery = usedQ ? TensorAddress(*usedQ) : 0;
    a.usedKv = currentK ? TensorAddress(*currentK) : 0;
    a.checked = a.workspace + checkedOffset;
    a.context = c;
    a.inputs = {(tnd ? CheckTndTensorLayout(q) : CheckBsndTensorLayout(q)),
                qv ? (tnd ? CheckTndTensorLayout(*qv) : CheckBsndTensorLayout(*qv)) : MlaBsndTensorLayout{},
                (cuK ? CheckTndTensorLayout(k) : CheckBsndTensorLayout(k)),
                (cuK ? CheckTndTensorLayout(v) : CheckBsndTensorLayout(v))};
    a.cache = cache;
    a.scale = scale;
    a.softcap = softcap;
    FAMetadataArgs args{};
    FillMlaArgs(args, c, uint32_t(q.size(-1)), TensorAddress(metadata), a.cuQuery, a.cuKv, a.usedQuery, a.usedKv,
                a.checked, 0, !supplied, cache);
    RunMlaLaunch("ascendc_mla", [a, args, supplied, q, qv, k, v, out, lse, accum, accumLse, workspace, metadata, usedQ,
                                 currentK, cuQ, cuK, pageTable]() mutable {
        ComputeMlaMetadata<<<1, nullptr, a.stream>>>(&args, sizeof(args));
        const auto error = q.scalar_type() == at::kHalf ? LaunchMla<half>(a) : LaunchMla<bfloat16_t>(a);
        TORCH_CHECK(error == ACL_SUCCESS, "MLA: BATCH_MODE launch failed (", error, ")");
        return 0;
    });
    return {out, lse, accum, accumLse};
}

at::Tensor SchedulerMetadata(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t dim,
                             int64_t valueDim, at::ScalarType dtype, const at::Tensor& lengths,
                             const std::optional<at::Tensor>& cuQ, std::optional<int64_t> page, int64_t left,
                             int64_t right, bool causal, bool hasSoftcap, int64_t splits, std::optional<bool> pack,
                             int64_t margin)
{
    TORCH_CHECK(batch >= 0 && batch <= UINT32_MAX && sq >= 0 && sq <= INT32_MAX && sk >= 0 && sk <= INT32_MAX &&
                    heads > 0 && heads <= INT32_MAX && kvHeads > 0 && kvHeads <= INT32_MAX && heads % kvHeads == 0 &&
                    dim > 0 && dim <= INT32_MAX && valueDim > 0 && valueDim <= INT32_MAX,
                "MLA: metadata shape bounds exceeded; KV heads must divide query heads");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "MLA: qkv_dtype must be FP16 or BF16, not INT8");
    TORCH_CHECK(!page || (*page > 0 && *page <= INT32_MAX), "MLA: page_size must be positive and fit int32");
    CheckCacheLengths(lengths, batch, lengths.device());
    if (cuQ)
        CheckMetadataQueryPrefix(*cuQ, batch, lengths.device());
    const c10::OptionalDeviceGuard device_guard(device_of(lengths));
    TORCH_CHECK((dim == 64 && valueDim == 512 && kvHeads == 1) || (dim == 192 && valueDim == 128),
                "MLA: expected compressed 64/512 or prefill 192/128 dimensions");
    const auto platform = lengths.is_meta() ? ShapePlatform() : DevicePlatform(lengths.device().index(), false);
    MlaLayout l{};
    const auto c =
        MakeContext(batch, sq, sk, heads, kvHeads, valueDim, dtype, splits, left, right, causal, 2, platform,
                    page ? (cuQ ? 4 : 3) : (cuQ ? 2 : 0), 0, 0, page.value_or(0), hasSoftcap, pack, margin, l);
    auto metadata = at::empty({int64_t(MlaUnifiedBytes(l))}, lengths.options().dtype(at::kByte));
    if (lengths.is_meta())
        return metadata;
    FAMetadataArgs args{};
    FillMlaArgs(args, c, uint32_t(dim), TensorAddress(metadata), cuQ ? TensorAddress(*cuQ) : 0, 0, 0,
                TensorAddress(lengths), 0, uint32_t(bool(cuQ)), true, {});
    auto stream = c10_npu::getCurrentNPUStream();
    RecordInput(lengths, stream);
    if (cuQ)
        RecordInput(*cuQ, stream);
    at_npu::native::OpCommand::RunOpApiV2("ascendc_mla_metadata", [metadata, lengths, cuQ, args, stream]() mutable {
        ComputeMlaMetadata<<<1, nullptr, stream.stream(false)>>>(&args, sizeof(args));
        return 0;
    });
    return metadata;
}

// Adapt the existing public inference arguments without changing ordinary FA3.
std::optional<std::vector<at::Tensor>>
mla_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const std::optional<at::Tensor>& k_new_,
           const std::optional<at::Tensor>& v_new_, const std::optional<at::Tensor>& q_v_,
           const std::optional<at::Tensor>& out_, const std::optional<at::Tensor>& cu_seqlens_q_,
           const std::optional<at::Tensor>& cu_seqlens_k_, const std::optional<at::Tensor>& cu_seqlens_k_new_,
           const std::optional<at::Tensor>& seqused_q_, const std::optional<at::Tensor>& seqused_k_,
           std::optional<int64_t> max_seqlen_q_, std::optional<int64_t> max_seqlen_k_,
           const std::optional<at::Tensor>& page_table_, const std::optional<at::Tensor>& kv_batch_idx_,
           const std::optional<at::Tensor>& leftpad_k_, const std::optional<at::Tensor>& rotary_cos_,
           const std::optional<at::Tensor>& rotary_sin_, const std::optional<at::Tensor>& seqlens_rotary_,
           const std::optional<at::Tensor>& q_descale_, const std::optional<at::Tensor>& k_descale_,
           const std::optional<at::Tensor>& v_descale_, std::optional<double> softmax_scale_, bool is_causal,
           int64_t window_size_left, int64_t window_size_right, int64_t attention_chunk, double softcap_option,
           bool is_rotary_interleaved, const std::optional<at::Tensor>& scheduler_metadata_, int64_t num_splits,
           std::optional<bool> pack_gqa_, int64_t sm_margin, int64_t layout)
{
    const float softcap = static_cast<float>(softcap_option);
    const bool isMla = flash_attn::mla::CheckForwardBoundary(q, k, v, q_v_, cu_seqlens_q_.has_value(),
                                                             cu_seqlens_k_.has_value(), page_table_.has_value());
    if (!isMla)
        return std::nullopt;
    TORCH_CHECK(layout == 0 || layout == 1, "MLA: layout must be TND or BSND");
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    TORCH_CHECK(std::isfinite(softcap_option) && std::isfinite(softcap) && softcap_option >= 0 &&
                    (softcap_option == 0 || softcap > 0),
                "MLA: softcap must be finite and non-negative in FP32 without positive underflow");
    TORCH_CHECK(!k_new_ && !v_new_ && !out_ && !cu_seqlens_k_new_ && (cu_seqlens_q_ || !max_seqlen_q_) &&
                    (cu_seqlens_k_ || !max_seqlen_k_) && !kv_batch_idx_ && !leftpad_k_ && !rotary_cos_ &&
                    !rotary_sin_ && !seqlens_rotary_ && is_rotary_interleaved,
                "MLA: k/v, out, cu_seqlens_k_new, cache_batch_idx, cache_leftpad, rotary_cos, rotary_sin, "
                "rotary_seqlens and length/rotary options are unsupported");
    TORCH_CHECK(attention_chunk == 0 &&
                    (num_splits == 1 || (num_splits >= 0 && num_splits <= 256 && (!cu_seqlens_k_ || cu_seqlens_q_))),
                "MLA compressed forward is not implemented yet for mask, split, softcap or scheduling options");
    TORCH_CHECK(q.scalar_type() != at::kChar && v.scalar_type() != at::kChar &&
                    (!q_v_ || q_v_->scalar_type() != at::kChar),
                "MLA: INT8 quantization is not supported");
    TORCH_CHECK(!q_descale_ && !k_descale_ && !v_descale_, "MLA: q_descale/k_descale/v_descale require INT8 content");
    TORCH_CHECK(!cu_seqlens_q_ || (max_seqlen_q_ && (!cu_seqlens_k_ || max_seqlen_k_)),
                "MLA: TND requires max_seqlen_q/k");
    const float scale = softmax_scale_.value_or(1.0 / std::sqrt(q_v_ ? 576.0 : 192.0));
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    auto result = Forward(q, k, v, q_v_, num_splits, window_size_left, window_size_right, is_causal, scale, seqused_q_,
                          seqused_k_, scheduler_metadata_, cu_seqlens_q_, cu_seqlens_k_, max_seqlen_q_.value_or(0),
                          max_seqlen_k_.value_or(0), page_table_, softcap, pack_gqa_, sm_margin);
    // Eager and Meta must both return tensors for the public accumulator slots.
    for (size_t i = 2; i < result.size(); ++i) {
        if (!result[i].defined())
            result[i] = at::empty({0}, at::TensorOptions().device(q.device()));
    }
    return result;
}

} // namespace flash_attn::mla
