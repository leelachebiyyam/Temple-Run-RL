#include "metal_nmmo3.h"

#include <initializer_list>
#include <mutex>

namespace {
// One thread owns each output or parameter gradient, avoiding floating-point
// atomics and making repeated embedding indices deterministic.
const char *source = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct P { int B, O, H, C, E; };
constant int offsets[10] = {0,4,8,25,30,33,38,43,48,55};
inline int feature(int c) { int f=0; for(int i=1;i<10;i++) if(c>=offsets[i]) f=i; return f; }
kernel void n3_c1(device const float *obs [[buffer(0)]], device const float *w [[buffer(1)]], device float *out [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i >= uint(p.B*1536)) return;
 int b=i/1536, oc=(i/12)%128, pos=i%12, y=pos/4, x=pos%4; float v=0;
 for(int ky=0;ky<5;ky++) for(int kx=0;kx<5;kx++) for(int f=0;f<10;f++) {
  int code=int(obs[b*p.O+((y*3+ky)*15+x*3+kx)*10+f]);
  int c=offsets[f]+code, end=f==9?59:offsets[f+1];
  if(c>=offsets[f] && c<end) v+=w[oc*1475+c*25+ky*5+kx];
 }
 out[i]=max(v,0.0f);
}
kernel void n3_c2(device const float *in [[buffer(0)]], device const float *w [[buffer(1)]], device float *out [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.B*256)) return; int b=i/256, oc=(i/2)%128, x=i%2; float v=0;
 for(int c=0;c<128;c++) for(int y=0;y<3;y++) for(int k=0;k<3;k++) v+=in[b*1536+c*12+y*4+x+k]*w[oc*1152+c*9+y*3+k];
 out[i]=v;
}
kernel void n3_concat(device const float *obs [[buffer(0)]], device const float *conv [[buffer(1)]], device const float *embed [[buffer(2)]], device float *out [[buffer(3)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.B*1817)) return; int b=i/1817, c=i%1817; float v;
 if(c<256) v=conv[b*256+c];
 else if(c<1760) { int q=c-256, code=int(obs[b*p.O+1650+q/32]); v=embed[code*32+q%32]; }
 else if(c<1807) v=obs[b*p.O+1650+c-1760];
 else v=obs[b*p.O+p.O-10+c-1807]; out[i]=v;
}
kernel void n3_proj(device const float *in [[buffer(0)]], device const float *w [[buffer(1)]], device float *out [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.B*p.H)) return; int b=i/p.H, h=i%p.H; float v=0; for(int c=0;c<p.C;c++) v+=in[b*p.C+c]*w[h*p.C+c]; out[i]=max(v,0.0f);
}
kernel void n3_dproj(device const float *concat [[buffer(0)]], device const float *out [[buffer(1)]], device const float *dout [[buffer(2)]], device float *dw [[buffer(3)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.H*p.C)) return; int h=i/p.C,c=i%p.C; float v=0; for(int b=0;b<p.B;b++) if(out[b*p.H+h]>0) v+=dout[b*p.H+h]*concat[b*p.C+c]; dw[i]=v;
}
kernel void n3_dconcat(device const float *w [[buffer(0)]], device const float *out [[buffer(1)]], device const float *dout [[buffer(2)]], device float *dc [[buffer(3)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.B*p.C)) return; int b=i/p.C,c=i%p.C; float v=0; for(int h=0;h<p.H;h++) if(out[b*p.H+h]>0) v+=dout[b*p.H+h]*w[h*p.C+c]; dc[i]=v;
}
kernel void n3_dembed(device const float *obs [[buffer(0)]], device const float *dc [[buffer(1)]], device float *dw [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=4096) return; int code=i/32,d=i%32; float v=0; for(int b=0;b<p.B;b++) for(int f=0;f<47;f++) if(int(obs[b*p.O+1650+f])==code) v+=dc[b*p.C+p.E+f*32+d]; dw[i]=v;
}
kernel void n3_dw2(device const float *c1 [[buffer(0)]], device const float *dc [[buffer(1)]], device float *dw [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=128*1152) return; int oc=i/1152, k=i%1152, ic=k/9, y=(k%9)/3, x=k%3; float v=0; for(int b=0;b<p.B;b++) for(int s=0;s<2;s++) v+=dc[b*p.C+oc*2+s]*c1[b*1536+ic*12+y*4+s+x]; dw[i]=v;
}
kernel void n3_dc1(device const float *c1 [[buffer(0)]], device const float *w [[buffer(1)]], device const float *dc [[buffer(2)]], device float *di [[buffer(3)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.B*1536)) return; int b=i/1536, ic=(i/12)%128, y=(i%12)/4,x=i%4; float v=0;
 if(c1[i]>0) for(int oc=0;oc<128;oc++) for(int s=0;s<2;s++) { int k=x-s; if(k>=0 && k<3) v+=dc[b*p.C+oc*2+s]*w[oc*1152+ic*9+y*3+k]; } di[i]=v;
}
kernel void n3_dw1(device const float *obs [[buffer(0)]], device const float *di [[buffer(1)]], device float *dw [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=128*1475) return; int oc=i/1475,k=i%1475,ic=k/25,ky=(k%25)/5,kx=k%5,f=feature(ic),code=ic-offsets[f]; float v=0;
 for(int b=0;b<p.B;b++) for(int s=0;s<12;s++) { int y=s/4,x=s%4; if(int(obs[b*p.O+((y*3+ky)*15+x*3+kx)*10+f])==code) v+=di[b*1536+oc*12+s]; } dw[i]=v;
}
inline int tokrow(device const float *e, int j) {
 int kind=clamp(int(e[1]),0,1), typ=int(e[4]), extra=int(e[5]);
 if(j==0) return kind;
 if(j==1) return 38+clamp(int(e[2]),-7,7)+7;
 if(j==2) return 53+clamp(int(e[3]),-5,5)+5;
 if(j==3) return kind ? 19+clamp(typ,0,3) : 2+clamp(typ,0,16);
 if(j==4) return kind ? 23+clamp(extra,0,4) : 64+clamp(extra,0,5);
 if(j==5 && kind) return 28+clamp(int(e[6]),0,4);
 if(j==6 && kind) return 33+clamp(int(e[7]),0,4);
 return -1;
}
kernel void n3a_pack(device const float *obs [[buffer(0)]], device float *cand [[buffer(1)]], constant P &p [[buffer(8)]], uint b [[thread_position_in_grid]]) {
 if(b>=uint(p.B)) return; device const float *map=obs+b*p.O; device float *e=cand+b*128;
 bool ue[165], ui[165]; for(int t=0;t<165;t++){ue[t]=false;ui[t]=false;}
 ue[82]=true; for(int j=0;j<128;j++) e[j]=0;
 for(int n=0;n<16;n++) {
  int best=-1, dist=100000, kind=1;
  if(n==0) best=82;
  else {
   for(int pass=0;pass<2 && best<0;pass++) {
    kind=1-pass;
    for(int t=0;t<165;t++) {
     int y=t/15-5,x=t%15-7,d=x*x+y*y;
     bool used=kind?ue[t]:ui[t]; int code=int(map[t*10+(kind?4:2)]);
     if(!used && code!=0 && d<dist) {best=t;dist=d;}
    }
   }
  }
  if(best<0) break; if(kind) ue[best]=true; else ui[best]=true;
  device float *s=e+n*8; s[0]=1; s[1]=kind; s[2]=best%15-7; s[3]=best/15-5;
  if(kind) for(int j=0;j<4;j++) s[4+j]=map[best*10+4+j];
  else {s[4]=map[best*10+2];s[5]=map[best*10+3];}
 }
}
kernel void n3a_tokens(device const float *cand [[buffer(0)]], device const float *tok [[buffer(1)]], device float *x [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.B*256)) return; int b=i/256,n=(i/16)%16,d=i%16; device const float *e=cand+b*128+n*8; float a=0;
 if(e[0]>0.5f) for(int j=0;j<7;j++){int r=tokrow(e,j);if(r>=0)a+=tok[r*16+d];} x[i]=a;
}
kernel void n3a_linear(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.B*256)) return; int row=i/16,d=i%16; float a=0; for(int k=0;k<16;k++)a+=x[row*16+k]*w[d*16+k]; y[i]=a;
}
kernel void n3a_readout(device const float *cand [[buffer(0)]], device const float *x [[buffer(1)]], device const float *q [[buffer(2)]], device const float *k [[buffer(3)]], device const float *v [[buffer(4)]], device float *prob [[buffer(5)]], device float *out [[buffer(6)]], constant P &p [[buffer(8)]], uint b [[thread_position_in_grid]]) {
 if(b>=uint(p.B)) return; device const float *e=cand+b*128; device float *o=out+b*48; device float *a=prob+b*16;
 int N=0; for(int n=0;n<16;n++) if(e[n*8]>0.5f) N++;
 float mx=-INFINITY; for(int n=0;n<N;n++){float z=0;for(int d=0;d<16;d++)z+=q[b*256+d]*k[b*256+n*16+d];a[n]=z*0.25f;mx=max(mx,a[n]);}
 float sum=0;for(int n=0;n<N;n++){a[n]=exp(a[n]-mx);sum+=a[n];}for(int n=0;n<16;n++)a[n]=n<N?a[n]/sum:0;
 for(int j=0;j<48;j++)o[j]=0; float melee=0,sword=0,bow=0;
 for(int n=0;n<N;n++){device const float *s=e+n*8;int dx=int(s[2]),dy=int(s[3]),ax=abs(dx),ay=abs(dy);
  if(int(s[1]) && int(s[4])==2){if(ax+ay==1)melee=1;if(max(ax,ay)==1)sword=1;
   if((dx==0 && ay>=1 && ay<=4)||(dy==0 && ax>=1 && ax<=4))bow=1;
   if(dx==0 && (ay==3 || ay==4))o[24+(ay==4?4:0)+(dy>0?0:1)]=1;
   if(dy==0 && (ax==3 || ax==4))o[24+(ax==4?4:0)+(dx>0?2:3)]=1;
  }
 }
 for(int r=0;r<8;r++){o[r*3]=melee;o[r*3+1]=sword;o[r*3+2]=bow;}
 for(int d=0;d<16;d++){float z=x[b*256+d];for(int n=0;n<N;n++)z+=a[n]*v[b*256+n*16+d];o[32+d]=z;}
}
kernel void n3a_concat(device const float *obs [[buffer(0)]], device const float *conv [[buffer(1)]], device const float *embed [[buffer(2)]], device const float *att [[buffer(3)]], device float *out [[buffer(4)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.B*1865))return; int b=i/1865,c=i%1865;float v;
 if(c<256)v=conv[b*256+c];else if(c<304)v=att[b*48+c-256];
 else if(c<1808){int q=c-304;v=embed[int(obs[b*p.O+1650+q/32])*32+q%32];}
 else if(c<1855)v=obs[b*p.O+1650+c-1808];else v=obs[b*p.O+p.O-10+c-1855];out[i]=v;
}
// Match 5.0 CUDA's backward, which omits the forward QK scale here.
kernel void n3a_datt(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]], device const float *prob [[buffer(3)]], device const float *dc [[buffer(4)]], device float *dq [[buffer(5)]], device float *dk [[buffer(6)]], device float *dv [[buffer(7)]], constant P &p [[buffer(8)]], uint b [[thread_position_in_grid]]) {
 if(b>=uint(p.B))return; float dots[16],mean=0;for(int n=0;n<16;n++){dots[n]=0;for(int d=0;d<16;d++)dots[n]+=dc[b*p.C+288+d]*v[b*256+n*16+d];mean+=prob[b*16+n]*dots[n];}
 for(int n=0;n<16;n++){float ds=prob[b*16+n]*(dots[n]-mean);for(int d=0;d<16;d++){dk[b*256+n*16+d]=ds*q[b*256+d];dv[b*256+n*16+d]=prob[b*16+n]*dc[b*p.C+288+d];}}
 for(int n=0;n<16;n++)for(int d=0;d<16;d++){float z=0;if(n==0)for(int t=0;t<16;t++)z+=prob[b*16+t]*(dots[t]-mean)*k[b*256+t*16+d];dq[b*256+n*16+d]=z;}
}
kernel void n3a_dx(device const float *dq [[buffer(0)]], device const float *dk [[buffer(1)]], device const float *dv [[buffer(2)]], device const float *wq [[buffer(3)]], device const float *wk [[buffer(4)]], device const float *wv [[buffer(5)]], device const float *dc [[buffer(6)]], device float *dx [[buffer(7)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.B*256))return;int b=i/256,n=(i/16)%16,k=i%16;float z=n==0?dc[b*p.C+288+k]:0;
 for(int d=0;d<16;d++)z+=dq[b*256+n*16+d]*wq[d*16+k]+dk[b*256+n*16+d]*wk[d*16+k]+dv[b*256+n*16+d]*wv[d*16+k];dx[i]=z;
}
kernel void n3a_dw(device const float *d [[buffer(0)]], device const float *x [[buffer(1)]], device float *dw [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=256)return;int o=i/16,k=i%16;float z=0;for(int b=0;b<p.B;b++)for(int n=0;n<16;n++)z+=d[b*256+n*16+o]*x[b*256+n*16+k];dw[i]=z;
}
kernel void n3a_dtoken(device const float *cand [[buffer(0)]], device const float *dx [[buffer(1)]], device float *dw [[buffer(2)]], constant P &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
 if(i>=1120)return;int r=i/16,d=i%16;float z=0;for(int b=0;b<p.B;b++)for(int n=0;n<16;n++){device const float *e=cand+b*128+n*8;if(e[0]>0.5f)for(int j=0;j<7;j++)if(tokrow(e,j)==r)z+=dx[b*256+n*16+d];}dw[i]=z;
}
)METAL";
struct Params { int B, O, H, C, E; };

id<MTLLibrary> library() {
  static id<MTLLibrary> lib;
  static std::once_flag once;
  std::call_once(once, [] {
    NSError *error = nil;
    lib = [mtl_ctx()->device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                          options:nil error:&error];
    if (!lib) throw std::runtime_error([[error localizedDescription] UTF8String]);
  });
  return lib;
}
id<MTLComputePipelineState> pipeline(const char *name) {
  // mtl_pipeline only searches the platform library, so these custom kernels
  // retain their own pipelines.
  static std::mutex mutex;
  static NSMutableDictionary *cache = [NSMutableDictionary new];
  std::lock_guard<std::mutex> lock(mutex);
  NSString *key = [NSString stringWithUTF8String:name];
  id<MTLComputePipelineState> pso = cache[key];
  if (!pso) {
    id<MTLFunction> fn = [library() newFunctionWithName:key];
    NSError *error = nil;
    pso = [mtl_ctx()->device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) throw std::runtime_error([[error localizedDescription] UTF8String]);
    cache[key] = pso;
  }
  return pso;
}
void run(MetalStream *ms, const char *name, int count, Params p,
         std::initializer_list<const float *> args) {
  ms->compute_encoder();
  auto ps = pipeline(name);
  mtl_set_pso(ms, ps);
  int index = 0;
  for (const float *ptr : args) {
    NSUInteger offset;
    id<MTLBuffer> buf = mtl_buffer_for_ptr(ptr, &offset);
    uint64_t addr = buf.gpuAddress + offset;
    if (ms->bound_addresses[index] != addr) {
      [ms->arg_table setAddress:addr atIndex:index];
      ms->bound_addresses[index] = addr;
    }
    index++;
  }
  mtl_set_params(ms, p, 8);
  mtl_dispatch_1d(ms, ps, count);
  mtl_barrier(ms);
}
void validate(int b, int o, int h) {
  if (b <= 0 || o < 1707 || h <= 0)
    throw std::invalid_argument("NMMO3 requires positive batch/hidden and obs_size >= 1707");
}
}

void metal_nmmo3_forward(const MetalNMMO3Weights &w, MetalNMMO3Scratch &s,
                         const float *obs, int b, int o, int h, cudaStream_t stream) {
  validate(b,o,h); auto ms=mtl_resolve_stream(stream); Params p{b,o,h,1817,256};
  run(ms,"n3_c1",b*1536,p,{obs,w.conv1,s.conv1});
  run(ms,"n3_c2",b*256,p,{s.conv1,w.conv2,s.conv2});
  run(ms,"n3_concat",b*1817,p,{obs,s.conv2,w.embedding,s.concat});
  run(ms,"n3_proj",b*h,p,{s.concat,w.projection,s.out});
}
void metal_nmmo3_backward(const MetalNMMO3Weights &w, MetalNMMO3Scratch &s,
                          const float *obs, const float *dout, const MetalNMMO3Weights &dw,
                          int b, int o, int h, cudaStream_t stream) {
  validate(b,o,h); auto ms=mtl_resolve_stream(stream); Params p{b,o,h,1817,256};
  run(ms,"n3_dproj",h*1817,p,{s.concat,s.out,dout,dw.projection});
  run(ms,"n3_dconcat",b*1817,p,{w.projection,s.out,dout,s.dconcat});
  run(ms,"n3_dembed",4096,p,{obs,s.dconcat,dw.embedding});
  run(ms,"n3_dw2",128*1152,p,{s.conv1,s.dconcat,dw.conv2});
  run(ms,"n3_dc1",b*1536,p,{s.conv1,w.conv2,s.dconcat,s.dconv1});
  run(ms,"n3_dw1",128*1475,p,{obs,s.dconv1,dw.conv1});
}

void metal_nmmo3_attention_forward(const MetalNMMO3AttentionWeights &w,
    MetalNMMO3AttentionScratch &s, const float *obs, int b, int o, int h,
    cudaStream_t stream) {
  validate(b,o,h); auto ms=mtl_resolve_stream(stream); Params p{b,o,h,1865,304};
  auto &c=s.common;
  run(ms,"n3_c1",b*1536,p,{obs,w.conv1,c.conv1});
  run(ms,"n3_c2",b*256,p,{c.conv1,w.conv2,c.conv2});
  run(ms,"n3a_pack",b,p,{obs,s.candidates});
  run(ms,"n3a_tokens",b*256,p,{s.candidates,w.token,s.x});
  run(ms,"n3a_linear",b*256,p,{s.x,w.query,s.q});
  run(ms,"n3a_linear",b*256,p,{s.x,w.key,s.k});
  run(ms,"n3a_linear",b*256,p,{s.x,w.value,s.v});
  run(ms,"n3a_readout",b,p,{s.candidates,s.x,s.q,s.k,s.v,s.prob,s.readout});
  run(ms,"n3a_concat",b*1865,p,{obs,c.conv2,w.embedding,s.readout,c.concat});
  run(ms,"n3_proj",b*h,p,{c.concat,w.projection,c.out});
}
void metal_nmmo3_attention_backward(const MetalNMMO3AttentionWeights &w,
    MetalNMMO3AttentionScratch &s, const float *obs, const float *dout,
    const MetalNMMO3AttentionWeights &dw, int b, int o, int h,
    cudaStream_t stream) {
  validate(b,o,h); auto ms=mtl_resolve_stream(stream); Params p{b,o,h,1865,304};
  auto &c=s.common;
  run(ms,"n3_dproj",h*1865,p,{c.concat,c.out,dout,dw.projection});
  run(ms,"n3_dconcat",b*1865,p,{w.projection,c.out,dout,c.dconcat});
  run(ms,"n3_dembed",4096,p,{obs,c.dconcat,dw.embedding});
  run(ms,"n3_dw2",128*1152,p,{c.conv1,c.dconcat,dw.conv2});
  run(ms,"n3_dc1",b*1536,p,{c.conv1,w.conv2,c.dconcat,c.dconv1});
  run(ms,"n3_dw1",128*1475,p,{obs,c.dconv1,dw.conv1});
  run(ms,"n3a_datt",b,p,{s.q,s.k,s.v,s.prob,c.dconcat,s.dq,s.dk,s.dv});
  run(ms,"n3a_dx",b*256,p,{s.dq,s.dk,s.dv,w.query,w.key,w.value,c.dconcat,s.dx});
  run(ms,"n3a_dw",256,p,{s.dq,s.x,dw.query});
  run(ms,"n3a_dw",256,p,{s.dk,s.x,dw.key});
  run(ms,"n3a_dw",256,p,{s.dv,s.x,dw.value});
  run(ms,"n3a_dtoken",1120,p,{s.candidates,s.dx,dw.token});
}
