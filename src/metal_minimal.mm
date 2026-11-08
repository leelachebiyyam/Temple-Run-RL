#include "metal_minimal.h"
#include <initializer_list>

static const char *minimal_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Params { int batch, hidden; };
kernel void min_points(device float *points [[buffer(0)]], device const float *obs [[buffer(1)]], constant Params &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= uint(p.batch * 16 * 6)) return;
  int f = i % 6, n = (i / 6) % 16, b = i / 96;
  points[i] = obs[b * 66 + (f < 2 ? f : 2 + n * 4 + f - 2)];
}
kernel void min_max(device float *out [[buffer(0)]], device int *arg [[buffer(1)]], device const float *x [[buffer(2)]], device const float *w [[buffer(3)]], constant Params &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= uint(p.batch * p.hidden)) return;
  int b = i / p.hidden, h = i % p.hidden, bestn = 0;
  float best = -3.402823466e38f;
  for (int n = 0; n < 16; ++n) {
    float s = 0; for (int d = 0; d < 16; ++d) s += w[h*16+d] * max(0.0f, x[(b*16+n)*16+d]);
    if (s > best) { best = s; bestn = n; }
  }
  out[i] = best; arg[i] = bestn;
}
kernel void min_owgrad(device float *dw [[buffer(0)]], device const float *g [[buffer(1)]], device const float *x [[buffer(2)]], device const int *arg [[buffer(3)]], constant Params &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= uint(p.hidden*16)) return;
  int h=i/16, d=i%16; float s=0;
  for (int b=0;b<p.batch;++b) { int j=b*p.hidden+h; s+=g[j]*max(0.0f,x[(b*16+arg[j])*16+d]); }
  dw[i]=s;
}
kernel void min_entitygrad(device float *dx [[buffer(0)]], device const float *g [[buffer(1)]], device const float *w [[buffer(2)]], device const float *x [[buffer(3)]], device const int *arg [[buffer(4)]], constant Params &p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  if (i >= uint(p.batch*256)) return;
  int d=i%16,n=(i/16)%16,b=i/256; float s=0;
  if (x[i]>0) for (int h=0;h<p.hidden;++h) if (arg[b*p.hidden+h]==n) s+=g[b*p.hidden+h]*w[h*16+d];
  dx[i]=s;
}
kernel void min_relu(device float *x [[buffer(0)]], constant Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i < uint(p.batch*256)) x[i]=max(0.0f,x[i]);
}
kernel void min_attn_pool(device float *prob [[buffer(0)]], device float *pool [[buffer(1)]], device const float *x [[buffer(2)]], device const float *q [[buffer(3)]], device const float *k [[buffer(4)]], device const float *v [[buffer(5)]], constant Params &p [[buffer(6)]], uint b [[thread_position_in_grid]]) {
  if (b >= uint(p.batch)) return;
  float pooled[16]; for (int d=0;d<16;++d) pooled[d]=0;
  for (int i=0;i<16;++i) {
    float scores[16], m=-3.402823466e38f, sum=0;
    for (int j=0;j<16;++j) { float s=0; for (int d=0;d<16;++d) s+=q[(b*16+i)*16+d]*k[(b*16+j)*16+d]; scores[j]=s*0.25f; m=max(m,scores[j]); }
    for (int j=0;j<16;++j) { scores[j]=exp(scores[j]-m); sum+=scores[j]; }
    for (int j=0;j<16;++j) { float a=scores[j]/sum; prob[(b*16+i)*16+j]=a; for (int d=0;d<16;++d) pooled[d]+=a*v[(b*16+j)*16+d]; }
    for (int d=0;d<16;++d) pooled[d]+=x[(b*16+i)*16+d];
  }
  for (int d=0;d<16;++d) pool[b*16+d]=pooled[d]/16.0f;
}
kernel void min_attn_ds(device float *ds [[buffer(0)]], device const float *prob [[buffer(1)]], device const float *v [[buffer(2)]], device const float *dp [[buffer(3)]], constant Params &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= uint(p.batch*256)) return;
  int j=i%16, row=i/16, b=row/16; float dots[16], mean=0;
  for (int l=0;l<16;++l) { float s=0; for (int d=0;d<16;++d) s+=dp[b*16+d]*v[(b*16+l)*16+d]/16.0f; dots[l]=s; mean+=prob[row*16+l]*s; }
  // Match the CUDA backward, which does not apply the forward score scale.
  ds[i]=prob[i]*(dots[j]-mean);
}
kernel void min_attn_qkvgrad(device float *dq [[buffer(0)]], device float *dk [[buffer(1)]], device float *dv [[buffer(2)]], device const float *ds [[buffer(3)]], device const float *prob [[buffer(4)]], device const float *q [[buffer(5)]], device const float *k [[buffer(6)]], device const float *dp [[buffer(7)]], constant Params &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
  if (i >= uint(p.batch*256)) return;
  int d=i%16,n=(i/16)%16,b=i/256; float a=0,c=0,t=0;
  for (int j=0;j<16;++j) { a+=ds[(b*16+n)*16+j]*k[(b*16+j)*16+d]; c+=ds[(b*16+j)*16+n]*q[(b*16+j)*16+d]; t+=prob[(b*16+j)*16+n]*dp[b*16+d]/16.0f; }
  dq[i]=a; dk[i]=c; dv[i]=t;
}
kernel void min_attn_dx(device float *dx [[buffer(0)]], device const float *x [[buffer(1)]], device const float *dq [[buffer(2)]], device const float *dk [[buffer(3)]], device const float *dv [[buffer(4)]], device const float *wq [[buffer(5)]], device const float *wk [[buffer(6)]], device const float *wv [[buffer(7)]], device const float *dp [[buffer(8)]], constant Params &p [[buffer(9)]], uint i [[thread_position_in_grid]]) {
  if (i >= uint(p.batch*256)) return;
  int d=i%16,row=i/16,b=row/16; float s=dp[b*16+d]/16.0f;
  for (int j=0;j<16;++j) s+=dq[row*16+j]*wq[j*16+d]+dk[row*16+j]*wk[j*16+d]+dv[row*16+j]*wv[j*16+d];
  dx[i]=x[i]>0 ? s : 0;
}
)MSL";

struct MinimalParams { int batch, hidden; };
static id<MTLComputePipelineState> minimal_pipeline(const char *name) {
  std::lock_guard<std::mutex> lock(mtl_pipeline_mutex());
  MetalContext *ctx=mtl_ctx();
  NSString *key=[NSString stringWithUTF8String:name];
  id<MTLComputePipelineState> pso=ctx->pipelines[key];
  if (pso) return pso;
  NSError *error=nil;
  id<MTLLibrary> lib=[ctx->device newLibraryWithSource:[NSString stringWithUTF8String:minimal_source] options:nil error:&error];
  if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Minimal Metal compilation failed");
  id<MTLFunction> fn=[lib newFunctionWithName:key];
  if (!fn) throw std::runtime_error("Minimal Metal kernel not found");
  pso=[ctx->device newComputePipelineStateWithFunction:fn error:&error];
  if (!pso) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Minimal Metal pipeline creation failed");
  ctx->pipelines[key]=pso;
  return pso;
}
static void check(const PufTensor &t, std::initializer_list<int64_t> shape) {
  if (!t.bytes || t.dtype_size != 4 || t.ndim() != (int)shape.size()) throw std::invalid_argument("Minimal Metal encoder: invalid tensor");
  int i=0; for (int64_t n:shape) if (t.shape[i++] != n) throw std::invalid_argument("Minimal Metal encoder: invalid tensor shape");
}
static void dims(int b,int h) {
  if (b<=0 || h<=0 || (int64_t)b*256>INT32_MAX || (int64_t)b*h>INT32_MAX || (int64_t)h*16>INT32_MAX) throw std::invalid_argument("Minimal Metal encoder: invalid dimensions");
}
static void points(MetalStream *ms, PufTensor &dst, const PufTensor &obs, MinimalParams p) {
  ms->compute_encoder(); auto s=minimal_pipeline("min_points"); mtl_set_pso(ms,s);
  mtl_set_tensor(ms,dst,0); mtl_set_tensor(ms,obs,1); mtl_set_params(ms,p,2); mtl_dispatch_1d(ms,s,p.batch*96); mtl_barrier(ms);
}
static void common(MetalMinimalEncoder &e) {
  dims(e.batch,e.hidden); int b=e.batch,h=e.hidden;
  check(e.input_w,{16,6}); check(e.output_w,{h,16}); check(e.points,{b,16,6}); check(e.entity,{b,16,16}); check(e.output,{b,h}); check(e.argmax,{b,h});
}
void metal_minimal_forward(MetalMinimalEncoder &e,const PufTensor &obs,cudaStream_t stream) {
  common(e); check(obs,{e.batch,66}); MinimalParams p{e.batch,e.hidden}; MetalStream *ms=mtl_resolve_stream(stream);
  points(ms,e.points,obs,p); puf_mm(e.points,e.input_w,e.entity,stream); mtl_barrier(ms);
  ms->compute_encoder(); auto s=minimal_pipeline("min_max"); mtl_set_pso(ms,s);
  mtl_set_tensor(ms,e.output,0); mtl_set_tensor(ms,e.argmax,1); mtl_set_tensor(ms,e.entity,2); mtl_set_tensor(ms,e.output_w,3); mtl_set_params(ms,p,4); mtl_dispatch_1d(ms,s,e.batch*e.hidden); mtl_barrier(ms);
}
void metal_minimal_backward(MetalMinimalEncoder &e,const PufTensor &grad,cudaStream_t stream) {
  common(e); check(grad,{e.batch,e.hidden}); check(e.grad_entity,{e.batch,16,16}); check(e.input_wgrad,{16,6}); check(e.output_wgrad,{e.hidden,16});
  MinimalParams p{e.batch,e.hidden}; MetalStream *ms=mtl_resolve_stream(stream); ms->compute_encoder();
  auto s=minimal_pipeline("min_owgrad"); mtl_set_pso(ms,s); mtl_set_tensor(ms,e.output_wgrad,0); mtl_set_tensor(ms,grad,1); mtl_set_tensor(ms,e.entity,2); mtl_set_tensor(ms,e.argmax,3); mtl_set_params(ms,p,4); mtl_dispatch_1d(ms,s,e.hidden*16);
  s=minimal_pipeline("min_entitygrad"); mtl_set_pso(ms,s); mtl_set_tensor(ms,e.grad_entity,0); mtl_set_tensor(ms,grad,1); mtl_set_tensor(ms,e.output_w,2); mtl_set_tensor(ms,e.entity,3); mtl_set_tensor(ms,e.argmax,4); mtl_set_params(ms,p,5); mtl_dispatch_1d(ms,s,e.batch*256); mtl_barrier(ms);
  puf_mm_tn(e.grad_entity,e.points,e.input_wgrad,stream); mtl_barrier(ms);
}
static void attention_common(MetalMinimalAttentionEncoder &e) {
  dims(e.batch,e.hidden); int b=e.batch,h=e.hidden;
  check(e.win,{16,6}); check(e.wq,{16,16}); check(e.wk,{16,16}); check(e.wv,{16,16}); check(e.wout,{h,16});
  check(e.points,{b,16,6}); check(e.x,{b,16,16}); check(e.q,{b,16,16}); check(e.k,{b,16,16}); check(e.v,{b,16,16}); check(e.prob,{b,16,16}); check(e.pooled,{b,16}); check(e.output,{b,h});
}
void metal_minimal_attention_forward(MetalMinimalAttentionEncoder &e,const PufTensor &obs,cudaStream_t stream) {
  attention_common(e); check(obs,{e.batch,66}); MinimalParams p{e.batch,e.hidden}; MetalStream *ms=mtl_resolve_stream(stream);
  points(ms,e.points,obs,p); puf_mm(e.points,e.win,e.x,stream); mtl_barrier(ms);
  ms->compute_encoder(); auto s=minimal_pipeline("min_relu"); mtl_set_pso(ms,s); mtl_set_tensor(ms,e.x,0); mtl_set_params(ms,p,1); mtl_dispatch_1d(ms,s,e.batch*256); mtl_barrier(ms);
  puf_mm(e.x,e.wq,e.q,stream); puf_mm(e.x,e.wk,e.k,stream); puf_mm(e.x,e.wv,e.v,stream); mtl_barrier(ms);
  ms->compute_encoder(); s=minimal_pipeline("min_attn_pool"); mtl_set_pso(ms,s); mtl_set_tensor(ms,e.prob,0); mtl_set_tensor(ms,e.pooled,1); mtl_set_tensor(ms,e.x,2); mtl_set_tensor(ms,e.q,3); mtl_set_tensor(ms,e.k,4); mtl_set_tensor(ms,e.v,5); mtl_set_params(ms,p,6); mtl_dispatch_1d(ms,s,e.batch); mtl_barrier(ms);
  puf_mm(e.pooled,e.wout,e.output,stream); mtl_barrier(ms);
}
void metal_minimal_attention_backward(MetalMinimalAttentionEncoder &e,const PufTensor &grad,cudaStream_t stream) {
  attention_common(e); int b=e.batch,h=e.hidden; check(grad,{b,h}); check(e.dpool,{b,16}); check(e.ds,{b,16,16}); check(e.dq,{b,16,16}); check(e.dk,{b,16,16}); check(e.dv,{b,16,16}); check(e.dx,{b,16,16});
  check(e.win_g,{16,6}); check(e.wq_g,{16,16}); check(e.wk_g,{16,16}); check(e.wv_g,{16,16}); check(e.wout_g,{h,16});
  MinimalParams p{b,h}; MetalStream *ms=mtl_resolve_stream(stream); PufTensor g=grad;
  puf_mm_tn(g,e.pooled,e.wout_g,stream); puf_mm_nn(g,e.wout,e.dpool,stream); mtl_barrier(ms);
  ms->compute_encoder(); auto s=minimal_pipeline("min_attn_ds"); mtl_set_pso(ms,s); mtl_set_tensor(ms,e.ds,0); mtl_set_tensor(ms,e.prob,1); mtl_set_tensor(ms,e.v,2); mtl_set_tensor(ms,e.dpool,3); mtl_set_params(ms,p,4); mtl_dispatch_1d(ms,s,b*256); mtl_barrier(ms);
  s=minimal_pipeline("min_attn_qkvgrad"); mtl_set_pso(ms,s); mtl_set_tensor(ms,e.dq,0); mtl_set_tensor(ms,e.dk,1); mtl_set_tensor(ms,e.dv,2); mtl_set_tensor(ms,e.ds,3); mtl_set_tensor(ms,e.prob,4); mtl_set_tensor(ms,e.q,5); mtl_set_tensor(ms,e.k,6); mtl_set_tensor(ms,e.dpool,7); mtl_set_params(ms,p,8); mtl_dispatch_1d(ms,s,b*256); mtl_barrier(ms);
  s=minimal_pipeline("min_attn_dx"); mtl_set_pso(ms,s); mtl_set_tensor(ms,e.dx,0); mtl_set_tensor(ms,e.x,1); mtl_set_tensor(ms,e.dq,2); mtl_set_tensor(ms,e.dk,3); mtl_set_tensor(ms,e.dv,4); mtl_set_tensor(ms,e.wq,5); mtl_set_tensor(ms,e.wk,6); mtl_set_tensor(ms,e.wv,7); mtl_set_tensor(ms,e.dpool,8); mtl_set_params(ms,p,9); mtl_dispatch_1d(ms,s,b*256); mtl_barrier(ms);
  puf_mm_tn(e.dq,e.x,e.wq_g,stream); puf_mm_tn(e.dk,e.x,e.wk_g,stream); puf_mm_tn(e.dv,e.x,e.wv_g,stream); puf_mm_tn(e.dx,e.points,e.win_g,stream); mtl_barrier(ms);
}
