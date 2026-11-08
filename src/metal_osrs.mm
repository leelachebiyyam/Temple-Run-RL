#include "metal_osrs.h"
#include "../ocean/osrs/osrs_item_obs_generated.h"
#include <initializer_list>
#include <mutex>

#define OSRS_STRINGIFY_(...) #__VA_ARGS__
#define OSRS_STRINGIFY(...) OSRS_STRINGIFY_(__VA_ARGS__)
#define OSRS_METAL_ITEM_ROW(code, item, idx, id, slot, click, kind, doses, next, style, ...) \
  "{" OSRS_STRINGIFY(__VA_ARGS__) "},"
static const char *osrs_items = OSRS_ITEM_CONTENT_ROWS(OSRS_METAL_ITEM_ROW);
#undef OSRS_METAL_ITEM_ROW
#undef OSRS_STRINGIFY
#undef OSRS_STRINGIFY_

// The implementation deliberately uses independent output threads and no
// atomics: ties select the first record and gradients have deterministic order.
static const char *osrs_source = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params { int B, O, H, R, F, start, obs_features, onehot, scale, item; };
)METAL";
static const char *osrs_kernels = R"METAL(
inline float gelu(float x) { return 0.5f*x*(1.0f+tanh(0.7978845608028654f*(x+0.044715f*x*x*x))); }
inline float dgelu(float x) { float t=tanh(0.7978845608028654f*(x+0.044715f*x*x*x)); return 0.5f*(1.0f+t)+0.5f*x*(1.0f-t*t)*0.7978845608028654f*(1.0f+3.0f*0.044715f*x*x); }
template<typename T> void global_f(device T *out, device const T *obs, device const T *w, constant Params &p, uint i) {
 if(i>=uint(p.B*p.H)) return; int b=i/p.H,h=i%p.H; float s=0;
 for(int j=0;j<p.O;j++) s+=float(obs[b*p.O+j])*float(w[h*p.O+j]); out[i]=T(s);
}
template<typename T> void gather_f(device T *flat, device const T *obs, constant Params &p, uint i) {
 if(i>=uint(p.B*p.R*p.F)) return; int b=i/(p.R*p.F), rem=i%(p.R*p.F), r=rem/p.F, f=rem%p.F;
 int off=b*p.O+p.start+r*p.obs_features; int code=int(rint(float(obs[off])*p.scale));
 if(p.item) flat[i]=T(items[code][f]);
 else if(f<p.onehot) flat[i]=T(code==f+1 ? 1.0f : 0.0f);
 else flat[i]=obs[off+1+f-p.onehot];
}
template<typename T> void layer_f(device T *z, device T *a, device const T *x, device const T *w, constant Params &p, uint i) {
 if(i>=uint(p.B*p.R*16)) return; int rec=i/16,k=i%16; float s=0;
 for(int f=0;f<p.F;f++) s+=float(x[rec*p.F+f])*float(w[k*p.F+f]);
 z[i]=T(s); a[i]=T(gelu(float(z[i])));
}
template<typename T> void pool_f(device T *out, device int *arg, device const T *z, device const T *x, device const T *w, constant Params &p, uint i) {
 if(i>=uint(p.B*p.H)) return; int b=i/p.H,h=i%p.H, best_r=-1; float best=-3.4028234663852886e38f;
 for(int r=0;r<p.R;r++) { int base=(b*p.R+r)*p.F; float active=0;
  for(int f=0;f<(p.item ? 1 : p.onehot);f++) active+=float(x[base+f]);
  if(active<=0) continue; float s=0; for(int k=0;k<16;k++) s+=gelu(float(z[(b*p.R+r)*16+k]))*float(w[h*16+k]);
  if(s>best) {best=s;best_r=r;} }
 out[i]=T(float(out[i])+(best_r<0 ? 0.0f : best)); arg[i]=best_r;
}
template<typename T> void global_b(device T *dw, device const T *obs, device const T *g, constant Params &p, uint i) {
 if(i>=uint(p.H*p.O)) return; int h=i/p.O,f=i%p.O; float s=0; for(int b=0;b<p.B;b++) s+=float(g[b*p.H+h])*float(obs[b*p.O+f]); dw[i]=T(s);
}
template<typename T> void l2_b(device T *dw, device const T *g, device const T *a, device const int *arg, constant Params &p, uint i) {
 if(i>=uint(p.H*16)) return; int h=i/16,k=i%16; float s=0; for(int b=0;b<p.B;b++) { int r=arg[b*p.H+h]; if(r>=0) s+=float(g[b*p.H+h])*float(a[(b*p.R+r)*16+k]); } dw[i]=T(s);
}
template<typename T> void dz_b(device T *dz, device const T *g, device const T *w, device const T *z, device const int *arg, constant Params &p, uint i) {
 if(i>=uint(p.B*p.R*16)) return; int b=i/(p.R*16),r=(i/16)%p.R,k=i%16; float s=0;
 for(int h=0;h<p.H;h++) if(arg[b*p.H+h]==r) s+=float(g[b*p.H+h])*float(w[h*16+k]); dz[i]=T(s*dgelu(float(z[i])));
}
template<typename T> void l1_b(device T *dw, device const T *dz, device const T *x, constant Params &p, uint i) {
 if(i>=uint(16*p.F)) return; int k=i/p.F,f=i%p.F; float s=0; for(int r=0;r<p.B*p.R;r++) s+=float(dz[r*16+k])*float(x[r*p.F+f]); dw[i]=T(s);
}
#define MAKE(T,S) \
kernel void osrs_global_##S(device T *a [[buffer(0)]],device const T *b [[buffer(1)]],device const T *c [[buffer(2)]],constant Params &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {global_f(a,b,c,p,i);} \
kernel void osrs_gather_##S(device T *a [[buffer(0)]],device const T *b [[buffer(1)]],constant Params &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {gather_f(a,b,p,i);} \
kernel void osrs_layer_##S(device T *a [[buffer(0)]],device T *b [[buffer(1)]],device const T *c [[buffer(2)]],device const T *d [[buffer(3)]],constant Params &p [[buffer(4)]],uint i [[thread_position_in_grid]]) {layer_f(a,b,c,d,p,i);} \
kernel void osrs_pool_##S(device T *a [[buffer(0)]],device int *b [[buffer(1)]],device const T *c [[buffer(2)]],device const T *d [[buffer(3)]],device const T *e [[buffer(4)]],constant Params &p [[buffer(5)]],uint i [[thread_position_in_grid]]) {pool_f(a,b,c,d,e,p,i);} \
kernel void osrs_global_b_##S(device T *a [[buffer(0)]],device const T *b [[buffer(1)]],device const T *c [[buffer(2)]],constant Params &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {global_b(a,b,c,p,i);} \
kernel void osrs_l2_b_##S(device T *a [[buffer(0)]],device const T *b [[buffer(1)]],device const T *c [[buffer(2)]],device const int *d [[buffer(3)]],constant Params &p [[buffer(4)]],uint i [[thread_position_in_grid]]) {l2_b(a,b,c,d,p,i);} \
kernel void osrs_dz_b_##S(device T *a [[buffer(0)]],device const T *b [[buffer(1)]],device const T *c [[buffer(2)]],device const T *d [[buffer(3)]],device const int *e [[buffer(4)]],constant Params &p [[buffer(5)]],uint i [[thread_position_in_grid]]) {dz_b(a,b,c,d,e,p,i);} \
kernel void osrs_l1_b_##S(device T *a [[buffer(0)]],device const T *b [[buffer(1)]],device const T *c [[buffer(2)]],constant Params &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {l1_b(a,b,c,p,i);}
MAKE(float,f32)
MAKE(half,f16)
)METAL";

struct OsrsParams { int B, O, H, R, F, start, obs_features, onehot, scale, item; };

static id<MTLComputePipelineState> osrs_pipeline(const char *name, int dtype) {
  MetalContext *ctx = mtl_ctx();
  NSString *key = [NSString stringWithFormat:@"osrs_%s_%s", name, dtype == 2 ? "f16" : "f32"];
  std::lock_guard<std::mutex> lock(mtl_pipeline_mutex());
  id<MTLComputePipelineState> pso = ctx->pipelines[key];
  if (pso) return pso;
  NSString *source = [NSString stringWithFormat:@"%s\nconstant float items[201][14]={%s};\n%s", osrs_source, osrs_items, osrs_kernels];
  NSError *error = nil;
  id<MTLLibrary> library = [ctx->device newLibraryWithSource:source options:nil error:&error];
  if (!library) throw std::runtime_error(error.localizedDescription.UTF8String ?: "OSRS Metal compilation failed");
  id<MTLFunction> fn = [library newFunctionWithName:key];
  if (!fn) throw std::runtime_error("OSRS Metal kernel not found");
  pso = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
  if (!pso) throw std::runtime_error(error.localizedDescription.UTF8String ?: "OSRS Metal pipeline creation failed");
  ctx->pipelines[key] = pso;
  return pso;
}

static void tensor_check(const PufTensor &t, int64_t n, int dtype) {
  if (!t.bytes || t.numel() != n || t.dtype_size != dtype)
    throw std::invalid_argument("OSRS Metal tensor has invalid storage, size or dtype");
}
static void dispatch(MetalStream *ms, const char *name, int dtype, const OsrsParams &p,
                     int count, std::initializer_list<const PufTensor *> tensors) {
  mtl_barrier(ms);
  ms->compute_encoder();
  auto pipeline = osrs_pipeline(name, dtype);
  mtl_set_pso(ms, pipeline);
  unsigned index = 0;
  for (const PufTensor *t : tensors) mtl_set_tensor(ms, *t, index++);
  mtl_set_params(ms, p, index);
  mtl_dispatch_1d(ms, pipeline, count);
}
static void validate_branch(const MetalOsrsEncoder &e, const MetalOsrsBranch &b,
                            const OsrsParams &p, int dtype, bool backward) {
  tensor_check(b.l1_w, 16*p.F, dtype); tensor_check(b.l2_w, e.hidden*16, dtype);
  tensor_check(b.flat, (int64_t)e.batch*p.R*p.F, dtype);
  tensor_check(b.z1, (int64_t)e.batch*p.R*16, dtype);
  tensor_check(b.h1, (int64_t)e.batch*p.R*16, dtype);
  tensor_check(b.argmax, (int64_t)e.batch*e.hidden, 4);
  if (backward) {
    tensor_check(b.grad_z1, (int64_t)e.batch*p.R*16, dtype);
    tensor_check(b.l1_wgrad, 16*p.F, dtype); tensor_check(b.l2_wgrad, e.hidden*16, dtype);
  }
}
static OsrsParams params(const MetalOsrsEncoder &e, int r, int f, int start,
                         int obs_features, int onehot, int scale, bool item) {
  return {e.batch,e.obs_size,e.hidden,r,f,start,obs_features,onehot,scale,item};
}
static void branch_forward(MetalStream *ms, MetalOsrsEncoder &e, MetalOsrsBranch &b,
                           const PufTensor &obs, const OsrsParams &p, int dtype) {
  validate_branch(e,b,p,dtype,false);
  dispatch(ms,"gather",dtype,p,p.B*p.R*p.F,{&b.flat,&obs});
  dispatch(ms,"layer",dtype,p,p.B*p.R*16,{&b.z1,&b.h1,&b.flat,&b.l1_w});
  dispatch(ms,"pool",dtype,p,p.B*p.H,{&e.output,&b.argmax,&b.z1,&b.flat,&b.l2_w});
}
static void branch_backward(MetalStream *ms, MetalOsrsEncoder &e, MetalOsrsBranch &b,
                            const PufTensor &grad, const OsrsParams &p, int dtype) {
  validate_branch(e,b,p,dtype,true);
  dispatch(ms,"l2_b",dtype,p,p.H*16,{&b.l2_wgrad,&grad,&b.h1,&b.argmax});
  dispatch(ms,"dz_b",dtype,p,p.B*p.R*16,{&b.grad_z1,&grad,&b.l2_w,&b.z1,&b.argmax});
  dispatch(ms,"l1_b",dtype,p,16*p.F,{&b.l1_wgrad,&b.grad_z1,&b.flat});
}
static void validate(const MetalOsrsEncoder &e, const PufTensor &obs) {
  int min_obs = e.encounter == METAL_OSRS_COLOSSEUM ? 653 : e.encounter == METAL_OSRS_INFERNO ? 306 : 91;
  if (e.encounter < METAL_OSRS_COLOSSEUM || e.encounter > METAL_OSRS_PVP ||
      e.batch <= 0 || e.hidden <= 0 || e.obs_size < min_obs ||
      (obs.dtype_size != 2 && obs.dtype_size != 4))
    throw std::invalid_argument("Invalid OSRS Metal encoder dimensions or encounter");
  tensor_check(obs,(int64_t)e.batch*e.obs_size,obs.dtype_size);
  tensor_check(e.global_w,(int64_t)e.hidden*e.obs_size,obs.dtype_size);
  tensor_check(e.output,(int64_t)e.batch*e.hidden,obs.dtype_size);
}
void metal_osrs_forward(MetalOsrsEncoder &e, const PufTensor &obs, cudaStream_t stream) {
  validate(e,obs); int dtype=obs.dtype_size; MetalStream *ms=mtl_resolve_stream(stream);
  OsrsParams p=params(e,0,0,0,0,0,0,false);
  dispatch(ms,"global",dtype,p,p.B*p.H,{&e.output,&obs,&e.global_w});
  branch_forward(ms,e,e.inventory,obs,params(e,28,14,52,1,0,256,true),dtype);
  branch_forward(ms,e,e.equipment,obs,params(e,11,14,80,1,0,256,true),dtype);
  if(e.encounter==METAL_OSRS_COLOSSEUM) branch_forward(ms,e,e.npc,obs,params(e,24,34,101,23,12,1,false),dtype);
  if(e.encounter==METAL_OSRS_INFERNO) branch_forward(ms,e,e.npc,obs,params(e,14,26,124,13,14,16,false),dtype);
}
void metal_osrs_backward(MetalOsrsEncoder &e, const PufTensor &obs, const PufTensor &grad, cudaStream_t stream) {
  validate(e,obs); int dtype=obs.dtype_size; MetalStream *ms=mtl_resolve_stream(stream);
  tensor_check(grad,(int64_t)e.batch*e.hidden,dtype);
  tensor_check(e.global_wgrad,(int64_t)e.hidden*e.obs_size,dtype);
  OsrsParams p=params(e,0,0,0,0,0,0,false);
  dispatch(ms,"global_b",dtype,p,p.H*p.O,{&e.global_wgrad,&obs,&grad});
  branch_backward(ms,e,e.inventory,grad,params(e,28,14,52,1,0,256,true),dtype);
  branch_backward(ms,e,e.equipment,grad,params(e,11,14,80,1,0,256,true),dtype);
  if(e.encounter==METAL_OSRS_COLOSSEUM) branch_backward(ms,e,e.npc,grad,params(e,24,34,101,23,12,1,false),dtype);
  if(e.encounter==METAL_OSRS_INFERNO) branch_backward(ms,e,e.npc,grad,params(e,14,26,124,13,14,16,false),dtype);
}
