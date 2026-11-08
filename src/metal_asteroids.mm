#include "metal_asteroids.h"

// Compiled separately from the shared kernel library so the custom encoder can
// be linked independently. All intermediates use the policy precision, as in
// the CUDA encoder; reductions accumulate in float.
static const char *asteroids_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct AEParams { int batch; int hidden; };

template<typename T> void points_impl(device T *points, device const T *obs,
                                      constant AEParams &p, uint i) {
  if (i >= uint(p.batch * 26 * 9)) return;
  int d = i % 9, point = (i / 9) % 26, b = i / (26 * 9);
  points[i] = obs[b * 134 + (d < 4 ? d : 4 + point * 5 + d - 4)];
}
template<typename T> void max_impl(device T *out, device int *arg,
    device const T *entity, device const T *w, constant AEParams &p, uint i) {
  if (i >= uint(p.batch * p.hidden)) return;
  int b = i / p.hidden, h = i % p.hidden;
  float best = -3.402823466e38f; int best_point = 0;
  for (int q = 0; q < 26; ++q) {
    float s = 0.0f;
    for (int k = 0; k < 16; ++k)
      s += float(w[h * 16 + k]) * max(0.0f, float(entity[(b * 26 + q) * 16 + k]));
    if (s > best) { best = s; best_point = q; }
  }
  out[i] = T(best); arg[i] = best_point;
}
template<typename T> void output_grad_impl(device T *dw,
    device const T *grad, device const T *entity, device const int *arg,
    constant AEParams &p, uint i) {
  if (i >= uint(p.hidden * 16)) return;
  int h = i / 16, k = i % 16; float sum = 0.0f;
  for (int b = 0; b < p.batch; ++b) {
    int j = b * p.hidden + h;
    sum += float(grad[j]) * max(0.0f, float(entity[(b * 26 + arg[j]) * 16 + k]));
  }
  dw[i] = T(sum);
}
template<typename T> void entity_grad_impl(device T *de,
    device const T *grad, device const T *w, device const T *entity,
    device const int *arg, constant AEParams &p, uint i) {
  if (i >= uint(p.batch * 26 * 16)) return;
  int k = i % 16, q = (i / 16) % 26, b = i / (26 * 16);
  float sum = 0.0f;
  if (float(entity[i]) > 0.0f)
    for (int h = 0; h < p.hidden; ++h)
      if (arg[b * p.hidden + h] == q)
        sum += float(grad[b * p.hidden + h]) * float(w[h * 16 + k]);
  de[i] = T(sum);
}
#define AE_KERNELS(T, SUFFIX) \
kernel void ae_points_##SUFFIX(device T *a [[buffer(0)]], device const T *b [[buffer(1)]], constant AEParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) { points_impl(a,b,p,i); } \
kernel void ae_max_##SUFFIX(device T *a [[buffer(0)]], device int *b [[buffer(1)]], device const T *c [[buffer(2)]], device const T *d [[buffer(3)]], constant AEParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) { max_impl(a,b,c,d,p,i); } \
kernel void ae_output_grad_##SUFFIX(device T *a [[buffer(0)]], device const T *b [[buffer(1)]], device const T *c [[buffer(2)]], device const int *d [[buffer(3)]], constant AEParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) { output_grad_impl(a,b,c,d,p,i); } \
kernel void ae_entity_grad_##SUFFIX(device T *a [[buffer(0)]], device const T *b [[buffer(1)]], device const T *c [[buffer(2)]], device const T *d [[buffer(3)]], device const int *e [[buffer(4)]], constant AEParams &p [[buffer(5)]], uint i [[thread_position_in_grid]]) { entity_grad_impl(a,b,c,d,e,p,i); }
AE_KERNELS(float, f32)
AE_KERNELS(half, f16)
)MSL";

struct AEParams { int batch, hidden; };

static id<MTLComputePipelineState> ae_pipeline(const char *base, int dtype) {
  std::lock_guard<std::mutex> lock(mtl_pipeline_mutex());
  MetalContext *ctx = mtl_ctx();
  NSString *key = [NSString stringWithFormat:@"%s_%s", base, dtype == 2 ? "f16" : "f32"];
  id<MTLComputePipelineState> pso = ctx->pipelines[key];
  if (pso) return pso;
  NSError *error = nil;
  NSString *source = [NSString stringWithUTF8String:asteroids_source];
  MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
  options.languageVersion = MTLLanguageVersion4_0;
  id<MTLLibrary> library = [ctx->device newLibraryWithSource:source options:options error:&error];
  if (!library) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Asteroids Metal compilation failed");
  id<MTLFunction> function = [library newFunctionWithName:key];
  if (!function) throw std::runtime_error("Asteroids Metal kernel not found");
  pso = [ctx->device newComputePipelineStateWithFunction:function error:&error];
  if (!pso) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Asteroids Metal pipeline creation failed");
  ctx->pipelines[key] = pso;
  return pso;
}

static void ae_check_tensor(const PufTensor &t, int dtype, int64_t count) {
  if (!t.bytes || t.dtype_size != dtype || t.numel() != count)
    throw std::invalid_argument("Asteroids Metal encoder: invalid tensor");
}
static void ae_check_shape(const PufTensor &t, int64_t a, int64_t b,
                           int64_t c = 0) {
  if (t.shape[0] != a || t.shape[1] != b || t.shape[2] != c ||
      t.shape[3] != 0)
    throw std::invalid_argument("Asteroids Metal encoder: invalid tensor shape");
}


static void ae_check_common(const MetalAsteroidsEncoder &e) {
  if (e.batch <= 0 || e.hidden <= 0 ||
      (int64_t)e.batch * 26 * 16 > INT32_MAX ||
      (int64_t)e.batch * e.hidden > INT32_MAX ||
      (int64_t)e.hidden * 16 > INT32_MAX ||
      (e.input_w.dtype_size != 2 && e.input_w.dtype_size != 4))
    throw std::invalid_argument("Asteroids Metal encoder: invalid dimensions or precision");
  int d = e.input_w.dtype_size;
  ae_check_tensor(e.input_w, d, 16 * 9);
  ae_check_tensor(e.output_w, d, (int64_t)e.hidden * 16);
  ae_check_tensor(e.points, d, (int64_t)e.batch * 26 * 9);
  ae_check_tensor(e.entity, d, (int64_t)e.batch * 26 * 16);
  ae_check_tensor(e.output, d, (int64_t)e.batch * e.hidden);
  ae_check_tensor(e.argmax, 4, (int64_t)e.batch * e.hidden);
  ae_check_shape(e.input_w, 16, 9);
  ae_check_shape(e.output_w, e.hidden, 16);
  ae_check_shape(e.points, e.batch, 26, 9);
  ae_check_shape(e.entity, e.batch, 26, 16);
  ae_check_shape(e.output, e.batch, e.hidden);
  ae_check_shape(e.argmax, e.batch, e.hidden);
}

void metal_asteroids_forward(MetalAsteroidsEncoder &e, const PufTensor &obs, cudaStream_t stream) {
  ae_check_common(e);
  int d = e.input_w.dtype_size;
  ae_check_tensor(obs, d, (int64_t)e.batch * 134);
  ae_check_shape(obs, e.batch, 134);
  MetalStream *ms = mtl_resolve_stream(stream);
  AEParams p{e.batch, e.hidden};
  ms->compute_encoder();
  auto points = ae_pipeline("ae_points", d);
  mtl_set_pso(ms, points);
  mtl_set_tensor(ms, e.points, 0); mtl_set_tensor(ms, obs, 1);
  mtl_set_params(ms, p, 2);
  mtl_dispatch_1d(ms, points, e.batch * 26 * 9);
  mtl_barrier(ms);
  puf_mm(e.points, e.input_w, e.entity, stream);
  mtl_barrier(ms);
  ms->compute_encoder();
  auto max = ae_pipeline("ae_max", d);
  mtl_set_pso(ms, max);
  mtl_set_tensor(ms, e.output, 0); mtl_set_tensor(ms, e.argmax, 1);
  mtl_set_tensor(ms, e.entity, 2); mtl_set_tensor(ms, e.output_w, 3);
  mtl_set_params(ms, p, 4);
  mtl_dispatch_1d(ms, max, e.batch * e.hidden);
  mtl_barrier(ms);
}

void metal_asteroids_backward(MetalAsteroidsEncoder &e, const PufTensor &grad, cudaStream_t stream) {
  ae_check_common(e);
  int d = e.input_w.dtype_size;
  ae_check_tensor(grad, d, (int64_t)e.batch * e.hidden);
  ae_check_tensor(e.grad_entity, d, (int64_t)e.batch * 26 * 16);
  ae_check_tensor(e.input_wgrad, d, 16 * 9);
  ae_check_tensor(e.output_wgrad, d, (int64_t)e.hidden * 16);
  ae_check_shape(grad, e.batch, e.hidden);
  ae_check_shape(e.grad_entity, e.batch, 26, 16);
  ae_check_shape(e.input_wgrad, 16, 9);
  ae_check_shape(e.output_wgrad, e.hidden, 16);
  MetalStream *ms = mtl_resolve_stream(stream);
  AEParams p{e.batch, e.hidden};
  ms->compute_encoder();
  auto output_grad = ae_pipeline("ae_output_grad", d);
  mtl_set_pso(ms, output_grad);
  mtl_set_tensor(ms, e.output_wgrad, 0); mtl_set_tensor(ms, grad, 1);
  mtl_set_tensor(ms, e.entity, 2); mtl_set_tensor(ms, e.argmax, 3);
  mtl_set_params(ms, p, 4);
  mtl_dispatch_1d(ms, output_grad, e.hidden * 16);
  auto entity_grad = ae_pipeline("ae_entity_grad", d);
  mtl_set_pso(ms, entity_grad);
  mtl_set_tensor(ms, e.grad_entity, 0); mtl_set_tensor(ms, grad, 1);
  mtl_set_tensor(ms, e.output_w, 2); mtl_set_tensor(ms, e.entity, 3);
  mtl_set_tensor(ms, e.argmax, 4); mtl_set_params(ms, p, 5);
  mtl_dispatch_1d(ms, entity_grad, e.batch * 26 * 16);
  mtl_barrier(ms);
  puf_mm_tn(e.grad_entity, e.points, e.input_wgrad, stream);
  mtl_barrier(ms);
}
