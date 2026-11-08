#include "metal_craftax.h"

// The embedding gradient is gathered per table entry instead of scattered:
// this avoids requiring 64-bit atomics while retaining CUDA's fixed-point sum.
static const char *craftax_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct CXParams { int batch; int hidden; };
constant int cx_offsets[8] = {0, 64, 72, 74, 90, 106, 122, 138};

kernel void cx_concat(device float *concat [[buffer(0)]],
                      device const float *obs [[buffer(1)]],
                      device const float *embed [[buffer(2)]],
                      constant CXParams &p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= uint(p.batch * 1635)) return;
  int b = i / 1635, j = i % 1635;
  if (j >= 1584) {
    concat[i] = obs[b * 843 + 792 + j - 1584];
    return;
  }
  int cell = j / 16, d = j % 16;
  float sum = 0.0f;
  for (int ch = 0; ch < 8; ++ch) {
    int row = cx_offsets[ch] + int(obs[b * 843 + cell * 8 + ch]);
    sum += embed[row * 16 + d];
  }
  concat[i] = sum;
}

kernel void cx_embed_grad(device float *dw [[buffer(0)]],
                          device const float *dc [[buffer(1)]],
                          device const float *obs [[buffer(2)]],
                          constant CXParams &p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= 2464) return;
  int row = i / 16, d = i % 16;
  int ch = 7;
  for (int k = 1; k < 8; ++k) {
    if (row < cx_offsets[k]) { ch = k - 1; break; }
  }
  int id = row - cx_offsets[ch];
  long sum = 0;
  for (int b = 0; b < p.batch; ++b) {
    for (int cell = 0; cell < 99; ++cell) {
      if (int(obs[b * 843 + cell * 8 + ch]) == id) {
        float scaled = dc[b * 1635 + cell * 16 + d] * 16777216.0f;
        sum += long(rint(scaled));
      }
    }
  }
  dw[i] = float(sum) * (1.0f / 16777216.0f);
}
)MSL";

struct CXParams { int batch, hidden; };

static id<MTLComputePipelineState> cx_pipeline(const char *name) {
  std::lock_guard<std::mutex> lock(mtl_pipeline_mutex());
  MetalContext *ctx = mtl_ctx();
  NSString *key = [NSString stringWithUTF8String:name];
  id<MTLComputePipelineState> pso = ctx->pipelines[key];
  if (pso) return pso;
  NSError *error = nil;
  id<MTLLibrary> lib = [ctx->device newLibraryWithSource:
      [NSString stringWithUTF8String:craftax_source] options:nil error:&error];
  if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Craftax Metal compilation failed");
  id<MTLFunction> fn = [lib newFunctionWithName:key];
  if (!fn) throw std::runtime_error("Craftax Metal kernel not found");
  pso = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
  if (!pso) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Craftax Metal pipeline creation failed");
  ctx->pipelines[key] = pso;
  return pso;
}

static void cx_check_tensor(const PufTensor &t, int64_t count) {
  if (!t.bytes || t.dtype_size != 4 || t.numel() != count)
    throw std::invalid_argument("Craftax Metal encoder: invalid float32 tensor");
}

static void cx_check_matrix(const PufTensor &t, int rows, int cols) {
  cx_check_tensor(t, (int64_t)rows * cols);
  if (t.ndim() != 2 || t.shape[0] != rows || t.shape[1] != cols)
    throw std::invalid_argument("Craftax Metal encoder: invalid matrix shape");
}

static void cx_check_common(const MetalCraftaxEncoder &e, const PufTensor &obs) {
  if (e.batch <= 0 || e.hidden <= 0 ||
      (int64_t)e.batch * 1635 > INT32_MAX ||
      (int64_t)e.batch * e.hidden > INT32_MAX ||
      (int64_t)e.hidden * 1635 > INT32_MAX ||
      (int64_t)e.batch * 843 > INT32_MAX)
    throw std::invalid_argument("Craftax Metal encoder: invalid dimensions");
  cx_check_matrix(obs, e.batch, 843);
  cx_check_matrix(e.embed_w, 154, 16);
  cx_check_matrix(e.proj_w, e.hidden, 1635);
  cx_check_matrix(e.concat, e.batch, 1635);
  cx_check_matrix(e.output, e.batch, e.hidden);
}

void metal_craftax_forward(MetalCraftaxEncoder &e, const PufTensor &obs,
                           cudaStream_t stream) {
  cx_check_common(e, obs);
  MetalStream *ms = mtl_resolve_stream(stream);
  CXParams p{e.batch, e.hidden};
  ms->compute_encoder();
  auto pso = cx_pipeline("cx_concat");
  mtl_set_pso(ms, pso);
  mtl_set_tensor(ms, e.concat, 0);
  mtl_set_tensor(ms, obs, 1);
  mtl_set_tensor(ms, e.embed_w, 2);
  mtl_set_params(ms, p, 3);
  mtl_dispatch_1d(ms, pso, e.batch * 1635);
  mtl_barrier(ms);
  puf_mm(e.concat, e.proj_w, e.output, stream);
  mtl_barrier(ms);
}

void metal_craftax_backward(MetalCraftaxEncoder &e, const PufTensor &obs,
                            const PufTensor &grad, cudaStream_t stream) {
  cx_check_common(e, obs);
  cx_check_matrix(grad, e.batch, e.hidden);
  cx_check_matrix(e.embed_wgrad, 154, 16);
  cx_check_matrix(e.proj_wgrad, e.hidden, 1635);
  MetalStream *ms = mtl_resolve_stream(stream);
  PufTensor grad_view = grad;
  puf_mm_tn(grad_view, e.concat, e.proj_wgrad, stream);
  mtl_barrier(ms);
  puf_mm_nn(grad_view, e.proj_w, e.concat, stream);
  mtl_barrier(ms);
  ms->compute_encoder();
  auto pso = cx_pipeline("cx_embed_grad");
  mtl_set_pso(ms, pso);
  mtl_set_tensor(ms, e.embed_wgrad, 0);
  mtl_set_tensor(ms, e.concat, 1);
  mtl_set_tensor(ms, obs, 2);
  CXParams p{e.batch, e.hidden};
  mtl_set_params(ms, p, 3);
  mtl_dispatch_1d(ms, pso, 2464);
  mtl_barrier(ms);
}
