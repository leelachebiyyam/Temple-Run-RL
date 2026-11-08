#import "metal_platform.h"
#import <QuartzCore/CABase.h>  // CACurrentMediaTime
#import <dispatch/dispatch.h>
#include "metal_shader_src.h"

#include <algorithm>
#include <cassert>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>

static MetalContext g_ctx = {};
static std::mutex g_pipeline_mutex;
// Protects both the pointer registry and all dynamic residency-set mutations.
static std::mutex g_buffer_mutex;
static std::mutex g_stats_mutex;
static std::mutex g_streams_mutex;
static std::vector<MetalStream *> g_dynamic_streams;

static void add_resident_buffer(id<MTLBuffer> buffer) {
  std::lock_guard<std::mutex> lock(g_buffer_mutex);
  [g_ctx.residency_set addAllocation:buffer];
  [g_ctx.residency_set commit];
  [g_ctx.residency_set requestResidency];
}

static void remove_resident_buffer(id<MTLBuffer> buffer) {
  std::lock_guard<std::mutex> lock(g_buffer_mutex);
  [g_ctx.residency_set removeAllocation:buffer];
  [g_ctx.residency_set commit];
}

static void register_buffer(char *base, int64_t size, id<MTLBuffer> buffer) {
  std::lock_guard<std::mutex> lock(g_buffer_mutex);
  [g_ctx.residency_set addAllocation:buffer];
  [g_ctx.residency_set commit];
  [g_ctx.residency_set requestResidency];
  g_ctx.buffers.push_back({base, size, buffer});
}

// The owner must first wait for every command that references this buffer.
static void unregister_buffer(void *base) {
  std::lock_guard<std::mutex> lock(g_buffer_mutex);
  for (auto it = g_ctx.buffers.begin(); it != g_ctx.buffers.end(); ++it) {
    if (it->base == base) {
      [g_ctx.residency_set removeAllocation:it->buffer];
      [g_ctx.residency_set commit];
      g_ctx.buffers.erase(it);
      return;
    }
  }
  assert(false && "scratch buffer missing from registry");
}

void *mtl_stream_scratch(MetalStream *stream, MetalScratchKind kind,
                         int64_t bytes) {
  assert(bytes > 0);
  MetalScratch &scratch = stream->scratch[(int)kind];
  if (bytes > scratch.capacity) {
    mtl_ensure_stream_synced((cudaStream_t)stream);
    if (scratch.ptr) unregister_buffer(scratch.ptr);
    id<MTLBuffer> buffer = [g_ctx.device newBufferWithLength:(NSUInteger)bytes
                                    options:MTLResourceStorageModeShared];
    assert(buffer && "Failed to allocate Metal scratch buffer");
    scratch.ptr = buffer.contents;
    scratch.capacity = bytes;
    register_buffer((char *)scratch.ptr, bytes, buffer);
  } else {
    // Previous dispatches on this stream may still access the same scratch.
    if (stream->flushed) stream->wait_completed();
    mtl_barrier(stream);
  }
  return scratch.ptr;
}

static void release_stream_scratch(MetalStream *stream) {
  for (auto &scratch : stream->scratch) {
    if (scratch.ptr) unregister_buffer(scratch.ptr);
    scratch = {};
  }
}

// ============================================================================
// Metal 4 tensor_ops GEMM — MSL source for JIT compilation.
// Separate library (needs metal_tensor + MetalPerformancePrimitives includes).
// ============================================================================

static const char *get_tensor_ops_shader_source() {
  return R"METAL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;

// C(M,N) = A(M,K) @ B(N,K)^T — float32, tensor_inline with device memory.
// Tile: 64 rows (M) x 32 cols (N), dynamic K.
// N MUST be a multiple of 32, M MUST be a multiple of 64.
kernel void tensor_ops_gemm_nt_f32(
    device float* A_buf [[buffer(0)]],
    device float* B_buf [[buffer(1)]],
    device float* C_buf [[buffer(2)]],
    constant uint& M    [[buffer(3)]],
    constant uint& N    [[buffer(4)]],
    constant uint& K    [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(K, M));
    auto B = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(K, N));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    constexpr auto desc = matmul2d_descriptor(
        64, 32,
        static_cast<int>(dynamic_extent),
        false, true, false
    );
    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(0, tgid.x * 32);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);

    op.run(mA, mB, mC);
}

// C(M,N) = A(M,K) @ B(K,N) — float32, tensor_inline with device memory.
// Row-major NN maps to col-major: C_cm(N,M) = B_cm(N,K) @ A_cm(K,M).
// Tiling follows NT convention: tgid.y tiles M (stride 64), tgid.x tiles N (stride 32).
// M % 64 == 0 and N % 32 == 0 required (caller falls back to steel_gemm).
kernel void tensor_ops_gemm_nn_f32(
    device float* A_buf [[buffer(0)]],
    device float* B_buf [[buffer(1)]],
    device float* C_buf [[buffer(2)]],
    constant uint& M    [[buffer(3)]],
    constant uint& N    [[buffer(4)]],
    constant uint& K    [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Row-major A(M,K) in memory == col-major A_cm(K,M)
    auto A_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(K, M));
    // Row-major B(K,N) in memory == col-major B_cm(N,K)
    auto B_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(N, K));
    // Row-major C(M,N) in memory == col-major C_cm(N,M)
    auto C_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    // Col-major NN: C_cm(N,M) = B_cm(N,K) @ A_cm(K,M)
    // op.run convention: result = second @ first (same as NT kernel)
    // first=A_cm, second=B_cm, no transposes
    constexpr auto desc = matmul2d_descriptor(
        64, 32,
        static_cast<int>(dynamic_extent),
        false, false, false
    );
    matmul2d<desc, execution_simdgroups<4>> op;

    // tgid.y tiles M at stride 64 (tile_M), tgid.x tiles N at stride 32 (tile_N)
    // Same convention as NT kernel
    auto mFirst  = A_cm.slice(0, tgid.y * 64);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto mResult = C_cm.slice(tgid.x * 32, tgid.y * 64);

    op.run(mFirst, mSecond, mResult);
}

// ---- fp16 variants: half inputs/outputs, float accumulation inside matmul2d ----

// C(M,N) = A(M,K) @ B(N,K)^T — half precision, tensor_inline with device memory.
kernel void tensor_ops_gemm_nt_f16(
    device half* A_buf [[buffer(0)]],
    device half* B_buf [[buffer(1)]],
    device half* C_buf [[buffer(2)]],
    constant uint& M    [[buffer(3)]],
    constant uint& N    [[buffer(4)]],
    constant uint& K    [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(K, M));
    auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(K, N));
    auto C = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    constexpr auto desc = matmul2d_descriptor(
        64, 32,
        static_cast<int>(dynamic_extent),
        false, true, false
    );
    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(0, tgid.x * 32);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);

    op.run(mA, mB, mC);
}

// C(M,N) = A(M,K) @ B(K,N) — half precision NN variant.
kernel void tensor_ops_gemm_nn_f16(
    device half* A_buf [[buffer(0)]],
    device half* B_buf [[buffer(1)]],
    device half* C_buf [[buffer(2)]],
    constant uint& M    [[buffer(3)]],
    constant uint& N    [[buffer(4)]],
    constant uint& K    [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A_cm = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(K, M));
    auto B_cm = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(N, K));
    auto C_cm = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    constexpr auto desc = matmul2d_descriptor(
        64, 32,
        static_cast<int>(dynamic_extent),
        false, false, false
    );
    matmul2d<desc, execution_simdgroups<4>> op;

    auto mFirst  = A_cm.slice(0, tgid.y * 64);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto mResult = C_cm.slice(tgid.x * 32, tgid.y * 64);

    op.run(mFirst, mSecond, mResult);
}

// C(M,N) = A(K,M)^T @ B(K,N) — float32, tensor_inline with device memory.
// TN: backward weight gradient. Row-major A(K,M) = col-major (M,K).
// matmul2d: result(N,M) = second(N,K) @ transpose(first(M,K))
// M % 64 == 0 and N % 32 == 0 required (caller falls back to steel_gemm).
kernel void tensor_ops_gemm_tn_f32(
    device float* A_buf [[buffer(0)]],
    device float* B_buf [[buffer(1)]],
    device float* C_buf [[buffer(2)]],
    constant uint& M    [[buffer(3)]],
    constant uint& N    [[buffer(4)]],
    constant uint& K    [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Row-major A(K,M) in memory == col-major tensor(M, K)
    auto A_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(M, K));
    // Row-major B(K,N) in memory == col-major tensor(N, K)
    auto B_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(N, K));
    // Row-major C(M,N) in memory == col-major tensor(N, M)
    auto C_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    // result(N,M) = second(N,K) @ transpose(first(M,K))
    // transpose_first=true: matmul sees first as (K,M)
    // (N,K) @ (K,M) = (N,M) = result
    constexpr auto desc = matmul2d_descriptor(
        64, 32,
        static_cast<int>(dynamic_extent),
        true, false, false
    );
    matmul2d<desc, execution_simdgroups<4>> op;

    // tgid.y tiles M at stride 64, tgid.x tiles N at stride 32
    auto mFirst  = A_cm.slice(tgid.y * 64, 0);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto mResult = C_cm.slice(tgid.x * 32, tgid.y * 64);

    op.run(mFirst, mSecond, mResult);
}

// C(M,N) = A(K,M)^T @ B(K,N) — half precision TN variant.
kernel void tensor_ops_gemm_tn_f16(
    device half* A_buf [[buffer(0)]],
    device half* B_buf [[buffer(1)]],
    device half* C_buf [[buffer(2)]],
    constant uint& M    [[buffer(3)]],
    constant uint& N    [[buffer(4)]],
    constant uint& K    [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A_cm = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(M, K));
    auto B_cm = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(N, K));
    auto C_cm = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    constexpr auto desc = matmul2d_descriptor(
        64, 32,
        static_cast<int>(dynamic_extent),
        true, false, false
    );
    matmul2d<desc, execution_simdgroups<4>> op;

    auto mFirst  = A_cm.slice(tgid.y * 64, 0);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto mResult = C_cm.slice(tgid.x * 32, tgid.y * 64);

    op.run(mFirst, mSecond, mResult);
}
)METAL";
}

void MetalStream::begin() {
  enc = nil;
  enc_active = false;
  pending_work = false;
  flushed = false;
  memset(bound_addresses, 0, sizeof(bound_addresses));
  [allocator reset];
  [cmd beginCommandBufferWithAllocator:allocator];
  {
    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    [cmd useResidencySet:mtl_ctx()->residency_set];
  }
  const_ring_offset = 0;
}

void MetalStream::compute_encoder() {
  if (flushed) wait_completed();
  if (!enc_active) {
    enc = [cmd computeCommandEncoder];
    [enc setArgumentTable:arg_table];
    enc_active = true;
    pending_work = true;
  }
}

void MetalStream::end_compute() {
  if (enc_active) {
    [enc endEncoding];
    enc = nil;
    enc_active = false;
  }
}

// Sync profiling globals
static int g_sync_count = 0;
static double g_sync_total_ns = 0.0;

// GPU timing diagnostic — actual kernel execution vs scheduling delay.
// Only active when g_gpu_timing_enabled is true (set via mtl_enable_gpu_timing).
// The MTL4CommitOptions + feedback handler allocation adds ObjC overhead that
// causes measurable scheduling jitter when sampled unconditionally.
static bool g_gpu_timing_enabled = false;
static double g_gpu_exec_ns = 0.0;
static double g_sched_wait_ns = 0.0;
static constexpr NSUInteger kMetalSyncTimeoutMs = 300000;

static void mtl_abort_sync_timeout(const char *where) {
  std::fprintf(stderr, "Metal sync timeout in %s\n", where);
  std::abort();
}

static double mach_to_ns(uint64_t ticks) {
  static const mach_timebase_info_data_t timebase = [] {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return info;
  }();
  return (double)ticks * timebase.numer / timebase.denom;
}

void MetalStream::sync() {
  if (flushed) {
    wait_completed();
    return;
  }
  end_compute();
  uint64_t t0 = mach_absolute_time();
  [cmd endCommandBuffer];
  uint64_t val = ++sync_event_value;
  id<MTL4CommandBuffer> bufs[] = { cmd };
  MetalContext *ctx = mtl_ctx();
  id<MTL4CommandQueue> q =
      (this == &ctx->train_stream) ? ctx->train_queue : ctx->queue;
  // Sample GPU timing every 32nd sync, only when profiling is enabled.
  // The MTL4CommitOptions + ObjC feedback handler block allocation adds
  // measurable jitter (~50-200us) that contributes to SPS variance.
  bool sample_timing;
  {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    sample_timing = g_gpu_timing_enabled && (g_sync_count % 32 == 0);
  }
  if (sample_timing) {
    CFTimeInterval cpu_commit = CACurrentMediaTime();
    MTL4CommitOptions *opts = [MTL4CommitOptions new];
    __block CFTimeInterval gpu_start = 0, gpu_end = 0;
    dispatch_semaphore_t feedback_done = dispatch_semaphore_create(0);
    [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
      gpu_start = fb.GPUStartTime;
      gpu_end = fb.GPUEndTime;
      dispatch_semaphore_signal(feedback_done);
    }];
    [q commit:bufs count:1 options:opts];
    [q signalEvent:sync_event value:val];
    BOOL signaled = [sync_event waitUntilSignaledValue:val timeoutMS:kMetalSyncTimeoutMs];
    if (!signaled) {
      mtl_abort_sync_timeout("MetalStream::sync");
    }
    if (dispatch_semaphore_wait(feedback_done,
        dispatch_time(DISPATCH_TIME_NOW, (int64_t)kMetalSyncTimeoutMs * NSEC_PER_MSEC))) {
      mtl_abort_sync_timeout("MetalStream::sync feedback");
    }
    if (gpu_start > 0 && gpu_end > 0) {
      std::lock_guard<std::mutex> lock(g_stats_mutex);
      g_gpu_exec_ns += (gpu_end - gpu_start) * 1e9;
      g_sched_wait_ns += (gpu_start - cpu_commit) * 1e9;
    }
  } else {
    [q commit:bufs count:1];
    [q signalEvent:sync_event value:val];
    BOOL signaled = [sync_event waitUntilSignaledValue:val timeoutMS:kMetalSyncTimeoutMs];
    if (!signaled) {
      mtl_abort_sync_timeout("MetalStream::sync");
    }
  }
  uint64_t t1 = mach_absolute_time();
  {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_sync_count++;
    g_sync_total_ns += mach_to_ns(t1 - t0);
  }
  begin();
}

void MetalStream::flush() {
  if (flushed) return;
  end_compute();
  if (pending_work) {
    [cmd endCommandBuffer];
    id<MTL4CommandBuffer> bufs[] = { cmd };
    MetalContext *ctx = mtl_ctx();
    id<MTL4CommandQueue> q =
        (this == &ctx->train_stream) ? ctx->train_queue : ctx->queue;
    // Signal event value for wait_completed().
    flush_event_val = ++sync_event_value;
    [q commit:bufs count:1];
    [q signalEvent:sync_event value:flush_event_val];
    flushed = true;
    pending_work = false;
  }
}

void MetalStream::commit_chunk() {
  if (flushed) {
    wait_completed();
    return;
  }
  end_compute();
  if (!pending_work) return;

  MetalContext *ctx = mtl_ctx();
  [cmd endCommandBuffer];
  id<MTL4CommandBuffer> bufs[] = { cmd };
  id<MTL4CommandQueue> q =
      (this == &ctx->train_stream) ? ctx->train_queue : ctx->queue;
  uint64_t t0 = mach_absolute_time();
  uint64_t val = ++sync_event_value;
  [q commit:bufs count:1];
  [q signalEvent:sync_event value:val];
  BOOL signaled = [sync_event waitUntilSignaledValue:val timeoutMS:kMetalSyncTimeoutMs];
  if (!signaled) {
    mtl_abort_sync_timeout("MetalStream::commit_chunk");
  }
  uint64_t t1 = mach_absolute_time();
  {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_sync_count++;
    g_sync_total_ns += mach_to_ns(t1 - t0);
  }
  pending_work = false;
  flushed = false;

  cmd = [ctx->device newCommandBuffer];
  assert(cmd && "Failed to allocate Metal command buffer for chunked training");
  begin();
}

void MetalStream::wait_completed() {
  if (flushed) {
    uint64_t t0 = mach_absolute_time();
    BOOL signaled = [sync_event waitUntilSignaledValue:flush_event_val timeoutMS:kMetalSyncTimeoutMs];
    if (!signaled) {
      mtl_abort_sync_timeout("MetalStream::wait_completed");
    }
    uint64_t t1 = mach_absolute_time();
    {
      std::lock_guard<std::mutex> lock(g_stats_mutex);
      g_sync_count++;
      g_sync_total_ns += mach_to_ns(t1 - t0);
    }
    flushed = false;
    begin();
  }
}

void mtl_sync_stats(int *out_count, double *out_total_ms) {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  *out_count = g_sync_count;
  *out_total_ms = g_sync_total_ns / 1e6;
  g_sync_count = 0;
  g_sync_total_ns = 0.0;
}

void mtl_enable_gpu_timing(bool enable) {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  g_gpu_timing_enabled = enable;
}

void mtl_gpu_timing_stats(double *gpu_exec_ms, double *sched_wait_ms) {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  *gpu_exec_ms = g_gpu_exec_ns / 1e6;
  *sched_wait_ms = g_sched_wait_ns / 1e6;
  g_gpu_exec_ns = 0.0;
  g_sched_wait_ns = 0.0;
}

static std::atomic_int g_gemm_dispatch_count{0};

void mtl_gemm_stats(int *tensor_ops_count) {
  *tensor_ops_count = g_gemm_dispatch_count.exchange(0, std::memory_order_relaxed);
}

void mtl_init() {
  @autoreleasepool {
    g_ctx.device = MTLCreateSystemDefaultDevice();
    assert(g_ctx.device && "No Metal device found");

    // JIT-compile all MSL shaders from the embedded source string
    NSError *error = nil;
    NSString *src =
        [NSString stringWithUTF8String:get_all_metal_shader_source()];
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion4_0;
    g_ctx.library =
        [g_ctx.device newLibraryWithSource:src options:opts error:&error];
    if (!g_ctx.library)
      throw std::runtime_error(error ? error.localizedDescription.UTF8String
                                     : "MSL compilation failed");

    g_ctx.pipelines = [NSMutableDictionary new];

    // Tensor operations are optional: older Apple GPUs use the steel kernels.
    {
      NSString *tensor_src = [NSString stringWithUTF8String:get_tensor_ops_shader_source()];
      MTLCompileOptions *tensor_opts = [[MTLCompileOptions alloc] init];
      tensor_opts.mathMode = MTLMathModeFast;
      tensor_opts.languageVersion = MTLLanguageVersion4_0;
      NSError *tensor_err = nil;
      id<MTLLibrary> tensor_lib = [g_ctx.device newLibraryWithSource:tensor_src
                                                             options:tensor_opts
                                                               error:&tensor_err];
      // A missing tensor_ops library or pipeline is an unsupported fast path;
      // dispatch checks for nil and uses the general GEMM implementation.
      auto compile_pso = [&](const char *name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [tensor_lib newFunctionWithName:
            [NSString stringWithUTF8String:name]];
        if (!fn) return nil;
        MTLComputePipelineDescriptor *pd = [[MTLComputePipelineDescriptor alloc] init];
        pd.computeFunction = fn;
        pd.maxTotalThreadsPerThreadgroup = 128;
        NSError *err = nil;
        id<MTLComputePipelineState> pso =
            [g_ctx.device newComputePipelineStateWithDescriptor:pd
                                                       options:0
                                                    reflection:nil
                                                         error:&err];
        if (!pso) return nil;
        return pso;
      };

      if (tensor_lib) {
        g_ctx.tensor_ops_gemm_nt_f32  = compile_pso("tensor_ops_gemm_nt_f32");
        g_ctx.tensor_ops_gemm_nn_f32  = compile_pso("tensor_ops_gemm_nn_f32");
        g_ctx.tensor_ops_gemm_tn_f32  = compile_pso("tensor_ops_gemm_tn_f32");
        g_ctx.tensor_ops_gemm_nt_f16  = compile_pso("tensor_ops_gemm_nt_f16");
        g_ctx.tensor_ops_gemm_nn_f16  = compile_pso("tensor_ops_gemm_nn_f16");
        g_ctx.tensor_ops_gemm_tn_f16  = compile_pso("tensor_ops_gemm_tn_f16");
      }

    }

    // Metal 4 reusable command buffer infrastructure
    g_ctx.queue = [g_ctx.device newMTL4CommandQueue];
    assert(g_ctx.queue && "Metal 4 required — device must support newMTL4CommandQueue");
    g_ctx.train_queue = [g_ctx.device newMTL4CommandQueue];

    // Command allocators + reusable command buffers
    g_ctx.stream.allocator = [g_ctx.device newCommandAllocator];
    g_ctx.stream.cmd = [g_ctx.device newCommandBuffer];
    g_ctx.stream.sync_event = [g_ctx.device newSharedEvent];
    g_ctx.stream.sync_event_value = 0;
    g_ctx.train_stream.allocator = [g_ctx.device newCommandAllocator];
    g_ctx.train_stream.cmd = [g_ctx.device newCommandBuffer];
    g_ctx.train_stream.sync_event = [g_ctx.device newSharedEvent];
    g_ctx.train_stream.sync_event_value = 0;

    // Argument tables — PPO kernel uses slots 0-19, Metal 4 max is 31
    MTL4ArgumentTableDescriptor *atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = 31;
    NSError *at_err = nil;
    g_ctx.stream.arg_table =
        [g_ctx.device newArgumentTableWithDescriptor:atd error:&at_err];
    assert(g_ctx.stream.arg_table && "Failed to create argument table");
    g_ctx.train_stream.arg_table =
        [g_ctx.device newArgumentTableWithDescriptor:atd error:&at_err];
    assert(g_ctx.train_stream.arg_table &&
           "Failed to create train argument table");

    // Per-stream constants ring buffers (64KB each, replaces setBytes)
    g_ctx.stream.const_ring =
        [g_ctx.device newBufferWithLength:MTL_CONST_RING_SIZE
                                  options:MTLResourceStorageModeShared];
    g_ctx.train_stream.const_ring =
        [g_ctx.device newBufferWithLength:MTL_CONST_RING_SIZE
                                  options:MTLResourceStorageModeShared];

    // Residency set — populated by mtl_wrap_allocator as buffers arrive
    MTLResidencySetDescriptor *rsd = [MTLResidencySetDescriptor new];
    rsd.initialCapacity = 16;
    NSError *rs_err = nil;
    g_ctx.residency_set =
        [g_ctx.device newResidencySetWithDescriptor:rsd error:&rs_err];
    assert(g_ctx.residency_set && "Failed to create residency set");
    [g_ctx.residency_set addAllocation:g_ctx.stream.const_ring];
    [g_ctx.residency_set addAllocation:g_ctx.train_stream.const_ring];
    [g_ctx.residency_set commit];
    [g_ctx.residency_set requestResidency];
    [g_ctx.queue addResidencySet:g_ctx.residency_set];
    [g_ctx.train_queue addResidencySet:g_ctx.residency_set];

    // Start the default stream (rollout) and training stream
    g_ctx.stream.begin();
    g_ctx.train_stream.begin();

  }
}

MetalContext *mtl_ctx() { return &g_ctx; }


void *mtl_stream() { return &g_ctx.stream; }

void *mtl_train_stream() { return &g_ctx.train_stream; }

void *mtl_create_stream() {
  MetalStream *ms = new MetalStream{};
  ms->allocator = [g_ctx.device newCommandAllocator];
  ms->cmd = [g_ctx.device newCommandBuffer];
  ms->sync_event = [g_ctx.device newSharedEvent];
  ms->sync_event_value = 0;
  assert(ms->allocator && ms->cmd && "Failed to create Metal stream allocator/cmd");

  MTL4ArgumentTableDescriptor *atd = [MTL4ArgumentTableDescriptor new];
  atd.maxBufferBindCount = 31;
  NSError *at_err = nil;
  ms->arg_table = [g_ctx.device newArgumentTableWithDescriptor:atd error:&at_err];
  assert(ms->arg_table && "Failed to create Metal stream argument table");

  ms->const_ring = [g_ctx.device newBufferWithLength:MTL_CONST_RING_SIZE
                                             options:MTLResourceStorageModeShared];
  assert(ms->const_ring && "Failed to create Metal stream constants ring");
  add_resident_buffer(ms->const_ring);

  ms->begin();
  {
    std::lock_guard<std::mutex> lock(g_streams_mutex);
    g_dynamic_streams.push_back(ms);
  }
  return ms;
}

void mtl_destroy_stream(void *stream) {
  assert(stream && "mtl_destroy_stream: null stream");
  MetalStream *ms = (MetalStream *)stream;
  assert(ms != &g_ctx.stream && ms != &g_ctx.train_stream);
  {
    std::lock_guard<std::mutex> lock(g_streams_mutex);
    auto it = std::find(g_dynamic_streams.begin(), g_dynamic_streams.end(), ms);
    assert(it != g_dynamic_streams.end() && "Unknown Metal stream");
    g_dynamic_streams.erase(it);
  }
  if (ms->flushed) {
    ms->wait_completed();
  } else if (ms->enc_active || ms->pending_work) {
    ms->sync();
  }
  release_stream_scratch(ms);
  remove_resident_buffer(ms->const_ring);
  ms->arg_table = nil;
  ms->cmd = nil;
  ms->allocator = nil;
  ms->enc = nil;
  ms->sync_event = nil;
  ms->const_ring = nil;
  delete ms;
}

void mtl_destroy() {
  // Context teardown requires all stream owners to have stopped. Drain dynamic
  // streams too, before releasing any buffer or the shared residency set.
  for (;;) {
    MetalStream *ms;
    {
      std::lock_guard<std::mutex> lock(g_streams_mutex);
      if (g_dynamic_streams.empty()) break;
      ms = g_dynamic_streams.back();
    }
    mtl_destroy_stream(ms);
  }
  mtl_ensure_stream_synced((cudaStream_t)&g_ctx.stream);
  mtl_ensure_stream_synced((cudaStream_t)&g_ctx.train_stream);
  release_stream_scratch(&g_ctx.stream);
  release_stream_scratch(&g_ctx.train_stream);

  // 3. Release all Metal objects inside @autoreleasepool to force immediate
  //    deallocation. Device released LAST — MTLBuffers/pipelines reference it.
  @autoreleasepool {
    g_ctx.stream.arg_table = nil;
    g_ctx.stream.cmd = nil;
    g_ctx.stream.allocator = nil;
    g_ctx.stream.enc = nil;
    g_ctx.stream.sync_event = nil;
    g_ctx.train_stream.arg_table = nil;
    g_ctx.train_stream.cmd = nil;
    g_ctx.train_stream.allocator = nil;
    g_ctx.train_stream.enc = nil;
    g_ctx.train_stream.sync_event = nil;
    g_ctx.stream.const_ring = nil;
    g_ctx.train_stream.const_ring = nil;
    g_ctx.residency_set = nil;
    g_ctx.queue = nil;
    g_ctx.train_queue = nil;
    g_ctx.buffers.clear();
    g_ctx.pipelines = nil;
    g_ctx.library = nil;
  }
  g_ctx.device = nil;
}

id<MTLBuffer> mtl_wrap_allocator(Allocator *alloc) {
  assert(alloc->mem && "Allocator::create() not called");

  // Find the highest byte offset used by any registered tensor
  int64_t max_end = 0;
  for (auto &e : alloc->regs) {
    int64_t end =
        ((char *)*e.data_ptr - (char *)alloc->mem) + puf_numel(e.shape) * e.elem_size;
    if (end > max_end)
      max_end = end;
  }

  // The backing allocation is page-rounded for Metal, while the registry
  // retains the logical range used for pointer lookup.
  int64_t size = max_end;
  int64_t buffer_size = (size + 16383) & ~int64_t(16383);

  // Zero-copy wrap: StorageModeShared on Apple Silicon means CPU and GPU
  // access the same physical memory pages. No deallocator — Allocator
  // owns the memory and frees it in destroy().
  id<MTLBuffer> buf =
      [g_ctx.device newBufferWithBytesNoCopy:alloc->mem
                                      length:buffer_size
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
  assert(buf && "newBufferWithBytesNoCopy failed — is Allocator::mem "
                "page-aligned? (requires WITH_METAL)");

  register_buffer((char *)alloc->mem, size, buf);

  return buf;
}

id<MTLBuffer> mtl_buffer_for(const PufTensor &t, NSUInteger *out_offset) {
  return mtl_buffer_for_ptr(t.bytes, out_offset);
}

// Typed tensor variant: look up buffer by raw pointer.
id<MTLBuffer> mtl_buffer_for_ptr(const void *ptr, NSUInteger *out_offset) {
  std::lock_guard<std::mutex> lock(g_buffer_mutex);
  uintptr_t p = (uintptr_t)ptr;
  for (auto &wb : g_ctx.buffers) {
    uintptr_t base = (uintptr_t)wb.base;
    if (p >= base && p - base < (uint64_t)wb.size) {
      *out_offset = (NSUInteger)(p - base);
      return wb.buffer;
    }
  }
  assert(false && "pointer not in any wrapped allocator buffer");
  __builtin_unreachable();
}
std::mutex &mtl_pipeline_mutex() { return g_pipeline_mutex; }


id<MTLComputePipelineState> mtl_pipeline(const char *name) {
  std::lock_guard<std::mutex> lock(g_pipeline_mutex);
  NSString *key = [NSString stringWithUTF8String:name];
  id<MTLComputePipelineState> pso = g_ctx.pipelines[key];
  if (pso)
    return pso;

  id<MTLFunction> fn = [g_ctx.library newFunctionWithName:key];
  assert(fn && "Kernel function not found in MSL library");

  NSError *error = nil;
  pso = [g_ctx.device newComputePipelineStateWithFunction:fn error:&error];
  if (!pso)
    throw std::runtime_error(error ? error.localizedDescription.UTF8String
                                   : "Pipeline creation failed");

  g_ctx.pipelines[key] = pso;
  return pso;
}

// ============================================================================
// GEMM dispatch — tensor_ops (aligned) or steel_gemm (unaligned) on GPU.
// All GEMM goes through Metal compute shaders (no CPU cblas path).
//
// Matches cuBLAS calling conventions in models.cu (row-major data,
// column-major API trick: swap A/B and transpose flags).
// ============================================================================

// All stream owners must be quiescent when changing this process-wide mode.
// Drain their prior GPU work before allowing subsequent dispatches in the new mode.
static void mtl_drain_streams() {
  if (!g_ctx.device) return;
  mtl_ensure_stream_synced((cudaStream_t)&g_ctx.stream);
  mtl_ensure_stream_synced((cudaStream_t)&g_ctx.train_stream);
  std::lock_guard<std::mutex> lock(g_streams_mutex);
  for (MetalStream *stream : g_dynamic_streams)
    mtl_ensure_stream_synced((cudaStream_t)stream);
}

static std::atomic_bool g_gpu_training = false;
void puf_set_gpu_training(bool val) {
  if (g_gpu_training.load(std::memory_order_acquire) == val) return;
  mtl_drain_streams();
  g_gpu_training.store(val, std::memory_order_release);
}
bool puf_is_gpu_training() { return g_gpu_training.load(std::memory_order_acquire); }

bool puf_stream_has_encoder(cudaStream_t stream) {
  MetalStream *ms = mtl_resolve_stream(stream);
  return ms->enc_active;
}

// Find MTLBuffer containing a raw pointer and compute byte offset.
static id<MTLBuffer> buffer_for_ptr(const void *ptr, NSUInteger *out_offset) {
  return mtl_buffer_for_ptr(ptr, out_offset);
}

// Bind a pre-resolved MTLBuffer+offset to a stream binding slot (GEMM helpers).
static inline void bind_buf(MetalStream *ms, id<MTLBuffer> buf,
                             NSUInteger offset, uint32_t index) {
  uint64_t addr = buf.gpuAddress + offset;
  [ms->arg_table setAddress:addr atIndex:index];
  ms->bound_addresses[index] = addr;
}

// ============================================================================
// Metal compute GEMM: uses simdgroup_matrix hardware instructions (M3+).
// Stays on the compute encoder (no encoder transitions).
// Used for fp32 GEMMs (rollout inference + Muon optimizer).
// ============================================================================

// Must match MSL GemmParams layout exactly (10 x 4 bytes = 40 bytes).
struct HostGemmParams {
  int M, N, K, lda, ldb, ldc;
  float alpha, beta;
  int trans_a, trans_b;
};

// steel_gemm dispatch: C(M,N) = alpha * op(A) @ op(B) + beta * C.
// 64x64 output tile per threadgroup, 128 threads (4 simdgroups).
static void steel_gemm_dispatch(const char *kernel_name,
                                 const void *A, const void *B, void *C,
                                 int M, int N, int K,
                                 bool trans_a, bool trans_b,
                                 int lda, int ldb, int ldc,
                                 float alpha, float beta,
                                 cudaStream_t stream) {
  MetalStream *ms = mtl_resolve_stream(stream);
  ms->compute_encoder();
  id<MTLComputePipelineState> pso = mtl_pipeline(kernel_name);
  mtl_set_pso(ms, pso);

  NSUInteger off_a, off_b, off_c;
  id<MTLBuffer> buf_a = buffer_for_ptr(A, &off_a);
  id<MTLBuffer> buf_b = buffer_for_ptr(B, &off_b);
  id<MTLBuffer> buf_c = buffer_for_ptr(C, &off_c);
  bind_buf(ms, buf_a, off_a, 0);
  bind_buf(ms, buf_b, off_b, 1);
  bind_buf(ms, buf_c, off_c, 2);

  HostGemmParams params = {M, N, K, lda, ldb, ldc, alpha, beta,
                            trans_a ? 1 : 0, trans_b ? 1 : 0};
  mtl_set_params(ms, params, 3);

  [ms->enc dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

  ms->pending_work = true;
  g_gemm_dispatch_count++;
}

static void compute_gemm(const float *A, const float *B, float *C,
                          int M, int N, int K, bool trans_a, bool trans_b,
                          int lda, int ldb, int ldc, float alpha, float beta,
                          cudaStream_t stream) {
  steel_gemm_dispatch("steel_gemm", A, B, C, M, N, K, trans_a, trans_b,
                       lda, ldb, ldc, alpha, beta, stream);
}

static void compute_gemm_f16(const void *A, const void *B, void *C,
                              int M, int N, int K, bool trans_a, bool trans_b,
                              int lda, int ldb, int ldc, float alpha, float beta,
                              cudaStream_t stream) {
  steel_gemm_dispatch("steel_gemm_f16", A, B, C, M, N, K, trans_a, trans_b,
                       lda, ldb, ldc, alpha, beta, stream);
}

// ============================================================================
// tensor_ops GEMM dispatch. All variants (NT/NN/TN x fp32/fp16) use identical
// dispatch: bind A/B/C buffers, set M/N/K params, dispatch 64x32 tile groups.
// Returns false if the PSO is nil (compilation failed).
// ============================================================================

static bool tensor_ops_dispatch(id<MTLComputePipelineState> pso,
                                const void *A, const void *B, void *C,
                                int M, int N, int K, cudaStream_t stream) {
  if (!pso) return false;
  g_gemm_dispatch_count++;

  MetalStream *ms = mtl_resolve_stream(stream);
  ms->compute_encoder();
  mtl_set_pso(ms, pso);

  NSUInteger off_a, off_b, off_c;
  id<MTLBuffer> buf_a = buffer_for_ptr(A, &off_a);
  id<MTLBuffer> buf_b = buffer_for_ptr(B, &off_b);
  id<MTLBuffer> buf_c = buffer_for_ptr(C, &off_c);

  bind_buf(ms, buf_a, off_a, 0);
  bind_buf(ms, buf_b, off_b, 1);
  bind_buf(ms, buf_c, off_c, 2);

  uint32_t mM = (uint32_t)M, mN = (uint32_t)N, mK = (uint32_t)K;
  mtl_set_params(ms, mM, 3);
  mtl_set_params(ms, mN, 4);
  mtl_set_params(ms, mK, 5);

  int groups_m = (M + 63) / 64;
  int groups_n = (N + 31) / 32;
  [ms->enc dispatchThreadgroups:MTLSizeMake(groups_n, groups_m, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

  ms->pending_work = true;
  return true;
}

// Typed wrappers for callers that pass float* (fp32 variants).
static bool tensor_ops_gemm_nt(const float *A, const float *B, float *C,
                                int M, int N, int K, cudaStream_t s) {
  return tensor_ops_dispatch(g_ctx.tensor_ops_gemm_nt_f32, A, B, C, M, N, K, s);
}
static bool tensor_ops_gemm_nn(const float *A, const float *B, float *C,
                                int M, int N, int K, cudaStream_t s) {
  return tensor_ops_dispatch(g_ctx.tensor_ops_gemm_nn_f32, A, B, C, M, N, K, s);
}
static bool tensor_ops_gemm_tn(const float *A, const float *B, float *C,
                                int M, int N, int K, cudaStream_t s) {
  return tensor_ops_dispatch(g_ctx.tensor_ops_gemm_tn_f32, A, B, C, M, N, K, s);
}
static bool tensor_ops_gemm_nt_f16(const void *A, const void *B, void *C,
                                    int M, int N, int K, cudaStream_t s) {
  return tensor_ops_dispatch(g_ctx.tensor_ops_gemm_nt_f16, A, B, C, M, N, K, s);
}
static bool tensor_ops_gemm_nn_f16(const void *A, const void *B, void *C,
                                    int M, int N, int K, cudaStream_t s) {
  return tensor_ops_dispatch(g_ctx.tensor_ops_gemm_nn_f16, A, B, C, M, N, K, s);
}
static bool tensor_ops_gemm_tn_f16(const void *A, const void *B, void *C,
                                    int M, int N, int K, cudaStream_t s) {
  return tensor_ops_dispatch(g_ctx.tensor_ops_gemm_tn_f16, A, B, C, M, N, K, s);
}

// K-split TN GEMM for small-M, small-N, large-K reductions (wgrad).
// Partitions K across Z threadgroups, each writing partial sums. Then reduces.

static void compute_gemm_ksplit_tn(const float *A, const float *B, float *C,
                                    int M, int N, int K,
                                    int lda, int ldb, int ldc,
                                    cudaStream_t stream) {
  int tile_groups_m = (M + 31) / 32;
  int tile_groups_n = (N + 31) / 32;
  int tile_groups = tile_groups_m * tile_groups_n;

  // Target ~32 threadgroups per GPU core (16 cores on M4 Pro).
  int target_tgs = 16 * 32;
  int num_splits = (target_tgs + tile_groups - 1) / tile_groups;
  num_splits = std::max(2, std::min(num_splits, K / 32));
  int k_per_split = (K + num_splits - 1) / num_splits;

  MetalStream *ms = mtl_resolve_stream(stream);
  int64_t partials_count = (int64_t)num_splits * M * N;
  float *ksplit_ptr = (float *)mtl_stream_scratch(
      ms, MetalScratchKind::KSplit, partials_count * sizeof(float));

  // Step 1: K-split GEMM — write partials.
  ms->compute_encoder();
  auto pso_ksplit = mtl_pipeline("sgemm_ksplit");
  mtl_set_pso(ms, pso_ksplit);

  NSUInteger off_a, off_b, off_p;
  id<MTLBuffer> buf_a = buffer_for_ptr(A, &off_a);
  id<MTLBuffer> buf_b = buffer_for_ptr(B, &off_b);
  id<MTLBuffer> buf_p = buffer_for_ptr(ksplit_ptr, &off_p);

  bind_buf(ms, buf_a, off_a, 0);
  bind_buf(ms, buf_b, off_b, 1);
  bind_buf(ms, buf_p, off_p, 2);

  HostGemmParams params = {M, N, K, lda, ldb, ldc, 1.0f, 0.0f, 1, 0}; // trans_a=TN
  mtl_set_params(ms, params, 3);
  int kps = k_per_split;
  mtl_set_params(ms, kps, 4);

  // sgemm_ksplit uses 2D threadgroups: (BN/TN, BM/TM) = (8, 8) = 64 threads.
  [ms->enc dispatchThreadgroups:MTLSizeMake(tile_groups_n, tile_groups_m, num_splits)
      threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  ms->pending_work = true;

  // Barrier before reduce.
  mtl_barrier(ms);

  // Step 2: Reduce partials → C.
  ms->compute_encoder();
  auto pso_reduce = mtl_pipeline("reduce_ksplit");
  mtl_set_pso(ms, pso_reduce);

  NSUInteger off_c;
  id<MTLBuffer> buf_c = buffer_for_ptr(C, &off_c);
  bind_buf(ms, buf_p, off_p, 0);
  bind_buf(ms, buf_c, off_c, 1);

  struct { int MN, num_splits; float alpha, beta; }
    rp = {M * N, num_splits, 1.0f, 0.0f};
  mtl_set_params(ms, rp, 2);

  mtl_dispatch_1d(ms, pso_reduce, M * N);
}

// Small compute-encoder GEMM for unaligned N.
// C(M,N) = A(M,K) @ B(N,K)^T. Each threadgroup covers several output rows;
// threads partition (row, column). Efficient for small N (e.g. decoder output
// N=15) where 64x64 tile waste dominates.
static void small_gemm_nt_dispatch(const float *A, const float *B, float *C,
                                    int M, int N, int K,
                                    cudaStream_t stream) {
  MetalStream *ms = mtl_resolve_stream(stream);
  ms->compute_encoder();
  id<MTLComputePipelineState> pso = mtl_pipeline("small_gemm_nt_f32");
  mtl_set_pso(ms, pso);

  NSUInteger off_a, off_b, off_c;
  id<MTLBuffer> buf_a = buffer_for_ptr(A, &off_a);
  id<MTLBuffer> buf_b = buffer_for_ptr(B, &off_b);
  id<MTLBuffer> buf_c = buffer_for_ptr(C, &off_c);

  bind_buf(ms, buf_a, off_a, 0);
  bind_buf(ms, buf_b, off_b, 1);
  bind_buf(ms, buf_c, off_c, 2);

  // Threadgroup size: block several output rows so the grid stays small enough
  // for the GPU to schedule efficiently (one row per threadgroup starves it).
  int max_threads = (int)pso.maxTotalThreadsPerThreadgroup;
  int rows = 1;
  if (N > 0 && N <= max_threads) {
    rows = MIN(8, max_threads / N);
    if (rows < 1) rows = 1;
  }
  int tg_size = ((rows * N + 31) / 32) * 32;
  tg_size = MIN(tg_size, max_threads);

  struct { uint32_t M, N, K, rows; } params = {
      (uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)rows};
  mtl_set_params(ms, params, 3);

  [ms->enc dispatchThreadgroups:MTLSizeMake((M + rows - 1) / rows, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

  ms->pending_work = true;
  g_gemm_dispatch_count++;
}

// out(...,N) = a(...,K) @ b(N,K)^T — leading dims folded into M
void puf_mm(PufTensor &a, PufTensor &b, PufTensor &out,
            cudaStream_t stream) {
  int na = a.ndim(), nb = b.ndim();
  int M = (int)(a.batch_size() * a.shape[na - 2]);
  int K = (int)a.shape[na - 1];
  int N = (int)b.shape[nb - 2];
  bool aligned = (N % 32 == 0) && (M % 64 == 0);
  float *a_f32 = (float *)a.bytes;
  float *b_f32 = (float *)b.bytes;

  if (a.dtype_size == 2) {
    if (aligned &&
        tensor_ops_gemm_nt_f16(a.bytes, b.bytes, out.bytes, M, N, K, stream)) {
      // fp16 tensor_ops (aligned)
    } else {
      // fp16 unaligned NT: steel_gemm_f16 (stays on compute encoder)
      compute_gemm_f16(a.bytes, b.bytes, out.bytes, M, N, K,
                       /*trans_a=*/false, /*trans_b=*/true,
                       K, K, N, 1.0f, 0.0f, stream);
    }
  } else if (aligned &&
             tensor_ops_gemm_nt(a_f32, b_f32,
                                (float *)out.bytes, M, N, K, stream)) {
    // fp32 tensor_ops (aligned)
  } else if (a.dtype_size == 4 && N < 128) {
    // Small unaligned N: 1-row-per-threadgroup kernel (avoids 64x64 tile waste)
    small_gemm_nt_dispatch(a_f32, b_f32,
                           (float *)out.bytes, M, N, K, stream);
  } else {
    // f32 unaligned: steel_gemm NT (stays on compute encoder)
    compute_gemm(a_f32, b_f32,
                 (float *)out.bytes, M, N, K,
                 /*trans_a=*/false, /*trans_b=*/true,
                 K, K, N, 1.0f, 0.0f, stream);
  }
}

// out(M,N) = a(...,M)^T @ b(...,N) — leading dims folded into K
void puf_mm_tn(PufTensor &a, PufTensor &b, PufTensor &out,
               cudaStream_t stream) {
  int na = a.ndim(), nb = b.ndim();
  int K = (int)(a.batch_size() * a.shape[na - 2]);
  int M = (int)a.shape[na - 1];
  int N = (int)b.shape[nb - 1];

  bool aligned = (M % 64 == 0) && (N % 32 == 0);
  float *a_f32 = (float *)a.bytes;
  float *b_f32 = (float *)b.bytes;

  if (a.dtype_size == 2) {
    if (aligned &&
        tensor_ops_gemm_tn_f16(a.bytes, b.bytes, out.bytes, M, N, K, stream)) {
      // fp16 tensor_ops TN on compute encoder
    } else {
      // fp16 unaligned TN: steel_gemm_f16 (stays on compute encoder)
      compute_gemm_f16(a.bytes, b.bytes, out.bytes, M, N, K,
                       /*trans_a=*/true, /*trans_b=*/false,
                       M, N, N, 1.0f, 0.0f, stream);
    }
  } else if (a.dtype_size == 4 && K > 4096 &&
             ((M + 31) / 32) * ((N + 31) / 32) < 32) {
    // K-split TN for small output (few tiles) + large K reduction.
    compute_gemm_ksplit_tn(a_f32, b_f32,
                           (float *)out.bytes, M, N, K,
                           M, N, N, stream);
  } else if (aligned &&
             tensor_ops_gemm_tn(a_f32, b_f32,
                                (float *)out.bytes, M, N, K, stream)) {
    // fp32 tensor_ops TN on compute encoder
  } else {
    // fp32 steel_gemm TN fallback (stays on compute encoder)
    compute_gemm(a_f32, b_f32,
                 (float *)out.bytes, M, N, K,
                 /*trans_a=*/true, /*trans_b=*/false,
                 M, N, N, 1.0f, 0.0f, stream);
  }
}

// out(...,N) = a(...,K) @ b(K,N) — leading dims folded into M
void puf_mm_nn(PufTensor &a, PufTensor &b, PufTensor &out,
               cudaStream_t stream) {
  int na = a.ndim(), nb = b.ndim();
  int M = (int)(a.batch_size() * a.shape[na - 2]);
  int K = (int)a.shape[na - 1];
  int N = (int)b.shape[nb - 1];

  bool aligned = (M % 64 == 0) && (N % 32 == 0);
  float *a_f32 = (float *)a.bytes;
  float *b_f32 = (float *)b.bytes;

  if (a.dtype_size == 2) {
    if (aligned &&
        tensor_ops_gemm_nn_f16(a.bytes, b.bytes, out.bytes, M, N, K, stream)) {
      // fp16 tensor_ops NN on compute encoder
    } else {
      // fp16 unaligned NN: steel_gemm_f16 (stays on compute encoder)
      compute_gemm_f16(a.bytes, b.bytes, out.bytes, M, N, K,
                       /*trans_a=*/false, /*trans_b=*/false,
                       K, N, N, 1.0f, 0.0f, stream);
    }
  } else if (aligned &&
             tensor_ops_gemm_nn(a_f32, b_f32,
                                (float *)out.bytes, M, N, K, stream)) {
    // fp32 tensor_ops NN (aligned)
  } else {
    // f32 unaligned: steel_gemm NN (stays on compute encoder)
    compute_gemm(a_f32, b_f32,
                 (float *)out.bytes, M, N, K,
                 /*trans_a=*/false, /*trans_b=*/false,
                 K, N, N, 1.0f, 0.0f, stream);
  }
}

// ============================================================================
// addmm temp buffer — lazily allocated for tensor_ops addmm decomposition.
// Only used for aligned Muon Newton-Schulz GEMMs.
// ============================================================================

static float *addmm_temp_buf(int count, cudaStream_t stream) {
  return (float *)mtl_stream_scratch(mtl_resolve_stream(stream),
      MetalScratchKind::AddMM, (int64_t)count * sizeof(float));
}

// out(...,N) = beta*out + alpha * a(...,K) @ b(K,N)
// Only called from Muon Newton-Schulz (small fp32 GEMMs: 512×512).
// When tensor_ops NN is available and dimensions align, compute the matrix
// product into scratch, then combine it with out in one pass.
void puf_addmm_nn(PufTensor &a, PufTensor &b, PufTensor &out, float alpha,
                   float beta, cudaStream_t stream) {
  int na = a.ndim(), nb = b.ndim();
  int M = (int)(a.batch_size() * a.shape[na - 2]);
  int K = (int)a.shape[na - 1];
  int N = (int)b.shape[nb - 1];

  const float *a_f32 = (const float *)a.bytes;
  const float *b_f32 = (const float *)b.bytes;

  if ((M % 64 == 0) && (N % 32 == 0) && g_ctx.tensor_ops_gemm_nn_f32) {
    // Decompose: out = beta*out + alpha*(a@b)
    // Step 1: temp = a @ b via tensor_ops NN (compute encoder)
    float *temp = addmm_temp_buf(M * N, stream);
    tensor_ops_gemm_nn(a_f32, b_f32, temp, M, N, K, stream);
    // Metal 4: make the scratch writes visible to the accumulation kernel.
    MetalStream *ms = mtl_resolve_stream(stream);
    mtl_barrier(ms);
    ms->compute_encoder();
    auto pso = mtl_pipeline("axpby_f32");
    mtl_set_pso(ms, pso);
    NSUInteger off_out, off_temp;
    id<MTLBuffer> out_buf = buffer_for_ptr(out.bytes, &off_out);
    id<MTLBuffer> temp_buf = buffer_for_ptr(temp, &off_temp);
    bind_buf(ms, out_buf, off_out, 0);
    bind_buf(ms, temp_buf, off_temp, 1);
    struct { float alpha, beta; int n; } params = {alpha, beta, M * N};
    mtl_set_params(ms, params, 2);
    mtl_dispatch_1d(ms, pso, M * N);
    ms->pending_work = true;
  } else {
    // Unaligned addmm: steel_gemm with alpha/beta (stays on compute encoder)
    compute_gemm(a_f32, b_f32, (float *)out.bytes, M, N, K,
                 /*trans_a=*/false, /*trans_b=*/false,
                 K, N, N, alpha, beta, stream);
  }
}

// ============================================================================
// CUDA compatibility functions for vecenv.h. Memory is shared on Apple
// Silicon, but host copies must wait for preceding GPU access. Async copies
// therefore block on their stream, then copy on the host; callers must also
// synchronize any other stream accessing the same memory. Synchronous copies,
// memset and frees drain all streams, so host owners must first be quiescent.
// ============================================================================

extern "C" {

int cudaHostAlloc(void **ptr, size_t size, unsigned int /*flags*/) {
  *ptr = calloc(1, size);
  return 0;
}

int cudaMalloc(void **ptr, size_t size) {
  *ptr = calloc(1, size);
  return 0;
}

int cudaMemcpy(void *dst, const void *src, size_t size, int /*kind*/) {
  mtl_drain_streams();
  memcpy(dst, src, size);
  return 0;
}

int cudaMemcpyAsync(void *dst, const void *src, size_t size, int /*kind*/,
                    void *stream) {
  if (g_ctx.device) mtl_ensure_stream_synced((cudaStream_t)stream);
  memcpy(dst, src, size);
  return 0;
}

int cudaMemset(void *ptr, int value, size_t size) {
  mtl_drain_streams();
  memset(ptr, value, size);
  return 0;
}

int cudaFree(void *ptr) {
  mtl_drain_streams();
  free(ptr);
  return 0;
}

int cudaFreeHost(void *ptr) {
  mtl_drain_streams();
  free(ptr);
  return 0;
}

int cudaSetDevice(int /*device*/) { return 0; }

int cudaDeviceSynchronize(void) {
  mtl_drain_streams();
  return 0;
}

int cudaStreamSynchronize(void *stream) {
  mtl_ensure_stream_synced((cudaStream_t)stream);
  return 0;
}

int cudaStreamCreateWithFlags(void **stream, unsigned int /*flags*/) {
  assert(stream && "cudaStreamCreateWithFlags expects a non-null stream pointer");
  *stream = mtl_create_stream();
  return 0;
}

int cudaStreamQuery(void *stream) {
  MetalStream *ms = mtl_resolve_stream((cudaStream_t)stream);
  // CUDA's cudaErrorNotReady. Unsubmitted encoded work is not complete.
  constexpr int not_ready = 600;
  if (ms->enc_active || ms->pending_work) return not_ready;
  if (ms->flushed && ms->sync_event.signaledValue < ms->flush_event_val)
    return not_ready;
  return 0;
}

const char *cudaGetErrorString(int error) {
  if (error == 0) return "no error";
  if (error == 600) return "not ready";
  return "unknown Metal compatibility error";
}

} // extern "C"
