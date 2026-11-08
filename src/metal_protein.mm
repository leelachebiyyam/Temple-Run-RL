#include "metal_protein.h"
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <random>
#define PROTEIN_EPSILON 1e-6f
#define PROTEIN_NUM_COST_RATIOS 6
#define PROTEIN_ACQ_MAX_CAP 65536
#define PROTEIN_COST_QUANTILE 0.97f
#define PROTEIN_MIN_OBS_NO_FAIL 100
#define PROTEIN_COST_GROWTH 1.01f
#define PROTEIN_CLF_ITERS 100
#define PROTEIN_THRESHOLD_COST_CAP 1.2f
#define PROTEIN_THRESHOLD_FALLBACK 0.9f
#define KD_LEAF 16
#define SP_LB 1e-4f
#define SF_IDX(np) ((np)-2)
#define OFF_IDX(np) ((np)-1)
using std::vector;
static float softplus(float x) { return (x > 20 ? x : log1pf(expf(x))) + SP_LB; }
static float inv_softplus(float x) { float v=x-SP_LB; return v>20?v:logf(expm1f(v)); }
static float softplus_grad(float x) { return 1/(1+expf(-x)); }
static inline float space_fwd(SpaceType type, float v) {
    if (type == SPACE_LOG) return log10f(v);
    if (type == SPACE_POW2) return log2f(v);
    if (type == SPACE_LOGIT) return log10f(1.0f - v);
    return v;
}

extern "C" float space_normalize(const Space *s, float value) {
    if (s->type == SPACE_LOGIT) {
        value = fmaxf(s->min, fminf(value, s->max));
    }
    float t = space_fwd(s->type, value);
    float lo = space_fwd(s->type, s->min);
    float hi = space_fwd(s->type, s->max);
    return 2.0f * ((t - lo) / (hi - lo)) - 1.0f;
}

extern "C" float space_unnormalize(const Space *s, float norm) {
    float u = (norm + 1.0f) * 0.5f;
    float lo = space_fwd(s->type, s->min);
    float hi = space_fwd(s->type, s->max);
    float t = u * (hi - lo) + lo;
    if (s->type == SPACE_POW2) {
        t = roundf(t);
    }
    float val = t;
    if (s->type == SPACE_LOG) val = powf(10.0f, t);
    if (s->type == SPACE_POW2) val = powf(2.0f, t);
    if (s->type == SPACE_LOGIT) val = 1.0f - powf(10.0f, t);
    if (s->is_integer) {
        val = roundf(val);
    }
    return val;
}


struct GaussianProcess {
    int dim, n;
    vector<float> raw, X, y, L, alpha, m, v, vmax;
    float noise, mean;
    int t;
    explicit GaussianProcess(int d): dim(d), n(0), raw(d+2, inv_softplus(1)), m(d+4,0), v(d+4,0), vmax(d+4,0), noise(inv_softplus(.01f)), mean(0), t(0) {}
};
static float kernel(const GaussianProcess &g, const float *a, const float *b) {
    float dot=0,r2=0;
    for(int i=0;i<g.dim;i++) { dot+=a[i]*b[i]; float z=(a[i]-b[i])/softplus(g.raw[i]); r2+=z*z; }
    float u=sqrtf(3*r2);
    return softplus(g.raw[g.dim])*(dot+softplus(g.raw[g.dim+1])+(1+u)*expf(-u));
}
static bool recompute(GaussianProcess &g) {
    int n=g.n; g.L.resize(n*n); g.alpha.resize(n);
    for(int retry=0;retry<2;retry++) {
        for(int j=0;j<n;j++) for(int i=0;i<n;i++) g.L[j*n+i]=kernel(g,&g.X[i*g.dim],&g.X[j*g.dim])+(i==j ? softplus(g.noise)+(retry?1e-8f:0):0);
        char lower='L'; __LAPACK_int ln=n, info; spotrf_(&lower,&ln,g.L.data(),&ln,&info);
        if(!info) { for(int i=0;i<n;i++) g.alpha[i]=g.y[i]-g.mean; __LAPACK_int one=1; spotrs_(&lower,&ln,&one,g.L.data(),&ln,g.alpha.data(),&ln,&info); return info==0; }
    }
    return false;
}
static float gp_train(GaussianProcess *p, const float *X, const float *y, int n, int iters, float lr) {
    GaussianProcess &g=*p; int d=g.dim, np=d+2; g.n=n;
    g.X.assign(X,X+n*d); g.y.assign(y,y+n);
    if(!recompute(g)) return 0;
    float loss=0;
    for(int iter=0;iter<iters;iter++) {
        if(!recompute(g)) break;
        vector<float> inv(n*n,0), grad(np,0); for(int i=0;i<n;i++) inv[i*n+i]=1;
        char lower='L'; __LAPACK_int ln=n, info; spotrs_(&lower,&ln,&ln,g.L.data(),&ln,inv.data(),&ln,&info); if(info) break;
        float fit=0, sum=0, trace=0, alpha2=0, logdet=0;
        for(int i=0;i<n;i++) { fit+=(g.y[i]-g.mean)*g.alpha[i]; sum+=g.alpha[i]; trace+=inv[i*n+i]; alpha2+=g.alpha[i]*g.alpha[i]; logdet+=logf(g.L[i*n+i]); }
        float sf=softplus(g.raw[d]), off=softplus(g.raw[d+1]);
        for(int j=0;j<n;j++) for(int i=0;i<n;i++) {
            const float *a=&g.X[i*d], *b=&g.X[j*d]; float dot=0,r2=0;
            for(int k=0;k<d;k++) { dot+=a[k]*b[k]; float z=(a[k]-b[k])/softplus(g.raw[k]); r2+=z*z; }
            float u=sqrtf(3*r2), e=expf(-u), w=g.alpha[i]*g.alpha[j]-inv[j*n+i];
            for(int k=0;k<d;k++) { float dif=a[k]-b[k], iv=1/softplus(g.raw[k]); grad[k]+=w*sf*3*dif*dif*iv*iv*iv*e; }
            grad[d]+=w*(dot+off+(1+u)*e); grad[d+1]+=w*sf;
        }
        for(int k=0;k<np;k++) grad[k]*=0.5f/n*softplus_grad(g.raw[k]);
        float nv=softplus(g.noise), sig=softplus_grad(g.noise), z=(logf(nv)+4.60517f)/.5f;
        float ng=(((-z/.5f)-1)*sig/nv)/n;
        float mll=(-.5f*fit-logdet-.5f*n*logf(2*M_PI))/n;
        loss=-mll-(-.5f*z*z-logf(nv)-logf(.5f))/n;
        g.t++; float bc1=1-powf(.9f,g.t),bc2=1-powf(.999f,g.t);
        for(int k=0;k<np+2;k++) {
            float v=k<np?-grad[k]:k==np?(-.5f*(alpha2-trace)*sig/n-ng):-sum/n;
            g.m[k]=.9f*g.m[k]+.1f*v; g.v[k]=.999f*g.v[k]+.001f*v*v;
            g.vmax[k]=fmaxf(g.vmax[k],g.v[k]/bc2);
            float step=lr*(g.m[k]/bc1)/(sqrtf(g.vmax[k])+1e-8f);
            if(k<np) g.raw[k]-=step; else if(k==np) g.noise-=step; else g.mean-=step;
        }
    }
    recompute(g); return loss;
}
static float gp_predict(const GaussianProcess &g,const float *x) {
    float value=g.mean; for(int i=0;i<g.n;i++) value+=kernel(g,&g.X[i*g.dim],x)*g.alpha[i]; return value;
}
struct KDNode { int lo, hi, left, right, split_dim; float split_val; };
struct SweepState : ProteinSweep {
    float *succ_params,*succ_scores,*succ_costs,*fail_params,*fail_scores,*fail_costs;
    int succ_n,fail_n,n_top,dim,cost_dim,*top_idx;
    float min_score,max_score,log_c_min,log_c_max;
    int acq_cap; float cm_A,cm_B,cm_max_score,cm_upper; int cm_min_samples,cm_fitted;
    float *clf_w,clf_bias; int clf_fitted;
    GaussianProcess gp_score,gp_cost;
    int suggestion_idx; float upper_cost_threshold,ratio_pool[6]; int pool_remaining;
    float *gp_train_params,*gp_train_y,*gp_train_c,*centers_buf,*log_c_buf,*ext_buf,*h_cands,*h_pred;
    int *pareto_buf,*pruned_buf,*kept_buf,*kd_idx,*kd_keep; KDNode *kd_nodes;
    std::mt19937_64 rng; unsigned long long quasi_index;
    vector<uint32_t> sobol_directions, sobol_scramble;
    explicit SweepState(ProteinSweep c): ProteinSweep(c), gp_score(c.space->num), gp_cost(c.space->num), rng(c.rng_seed), quasi_index(c.rng_seed) {}
};
static float uniform(SweepState *s) { return std::generate_canonical<float,24>(s->rng); }
static float sample_normal(SweepState *s) { float a=fmaxf(uniform(s),1e-9f),b=uniform(s); return .1f*sqrtf(-2*logf(a))*cosf(2*M_PI*b); }
// Generate valid Sobol direction numbers from primitive binary polynomials.
// The digital shift differs from cuRAND's fixed scrambling tables.
static void init_sobol(SweepState *sw) {
    sw->sobol_directions.resize(sw->dim * 32);
    sw->sobol_scramble.resize(sw->dim);
    std::mt19937_64 shift(sw->rng_seed);
    for (int d=0; d<sw->dim; d++) sw->sobol_scramble[d]=(uint32_t)shift();
    for (int j=0; j<32; j++) sw->sobol_directions[j]=uint32_t(1) << (31-j);
    int d=1;
    for (int degree=1; d<sw->dim && degree<31; degree++) {
        uint32_t high=uint32_t(1)<<degree, order=high-1;
        for (uint32_t poly=high|1; poly<(high<<1) && d<sw->dim; poly+=2) {
            uint32_t x=1, period=0;
            do {
                x <<= 1;
                if (x & high) x ^= poly;
                ++period;
            } while (x != 1 && period <= order);
            if (period != order) continue;
            uint32_t *v=&sw->sobol_directions[d*32];
            for (int j=0; j<degree; j++) v[j]=uint32_t(1) << (31-j);
            for (int j=degree; j<32; j++) {
                v[j]=v[j-degree] ^ (v[j-degree]>>degree);
                for (int k=1; k<degree; k++)
                    if ((poly>>(degree-k))&1) v[j]^=v[j-k];
            }
            ++d;
        }
    }
}


extern "C" ProteinSweep *protein_sweep_create(ProteinSweep c) {
    if(!c.space || !c.space->spaces || c.space->num<=0 || c.space->cost_idx < -1 || c.space->cost_idx>=c.space->num || c.success_cap<=0 || c.failure_cap<0 || c.top_k<=0 || c.gp_max_obs<=0 || c.suggestions_per_pareto<=0 || c.infer_batch_size<=0) return nullptr;
    SweepState *sw=new SweepState(c); int dim=c.space->num;
    sw->dim=dim; sw->cost_dim=c.space->cost_idx; sw->succ_n=sw->fail_n=sw->n_top=sw->suggestion_idx=0;
    init_sobol(sw);
    sw->cm_fitted=sw->clf_fitted=sw->pool_remaining=0; sw->upper_cost_threshold=-FLT_MAX;
    sw->cm_min_samples=30; sw->min_score=sw->log_c_min=FLT_MAX; sw->max_score=sw->log_c_max=-FLT_MAX;
    sw->clf_bias=sw->cm_A=sw->cm_B=sw->cm_max_score=sw->cm_upper=0;
    float ratios[]={.16f,.32f,.48f,.64f,.8f,1}; memcpy(sw->ratio_pool,ratios,sizeof ratios);
    sw->succ_params=(float*)calloc(c.success_cap*(dim+2),sizeof(float)); sw->succ_scores=sw->succ_params+c.success_cap*dim; sw->succ_costs=sw->succ_scores+c.success_cap;
    sw->fail_params=(float*)calloc(std::max(c.failure_cap,1)*(dim+2),sizeof(float)); sw->fail_scores=sw->fail_params+c.failure_cap*dim; sw->fail_costs=sw->fail_scores+c.failure_cap;
    sw->top_idx=(int*)malloc(c.top_k*sizeof(int));
    int centers=c.success_cap+3*c.top_k; sw->acq_cap=std::min(PROTEIN_ACQ_MAX_CAP,std::max(c.infer_batch_size,centers*c.suggestions_per_pareto));
    int ext=c.success_cap+c.failure_cap, kd=std::max(ext,sw->acq_cap);
    sw->clf_w=(float*)calloc(dim,sizeof(float));
    sw->gp_train_params=(float*)malloc(c.gp_max_obs*(dim+2)*sizeof(float)); sw->gp_train_y=sw->gp_train_params+c.gp_max_obs*dim; sw->gp_train_c=sw->gp_train_y+c.gp_max_obs;
    sw->pareto_buf=(int*)malloc(2*c.success_cap*sizeof(int)); sw->pruned_buf=sw->pareto_buf+c.success_cap;
    sw->centers_buf=(float*)malloc(centers*dim*sizeof(float)); sw->log_c_buf=(float*)malloc(c.success_cap*sizeof(float));
    sw->ext_buf=(float*)malloc(ext*(dim+2)*sizeof(float)); sw->kept_buf=(int*)malloc(kd*sizeof(int));
    sw->h_cands=(float*)malloc(sw->acq_cap*dim*sizeof(float)); sw->h_pred=(float*)malloc(2*sw->acq_cap*sizeof(float));
    sw->kd_idx=(int*)malloc(kd*sizeof(int)); sw->kd_nodes=(KDNode*)malloc(2*kd*sizeof(KDNode)); sw->kd_keep=(int*)malloc(kd*sizeof(int));
    return sw;
}
extern "C" void protein_sweep_destroy(ProteinSweep *s) {
    if(!s) return; SweepState *sw=static_cast<SweepState*>(s);
    free(sw->succ_params); free(sw->fail_params); free(sw->top_idx); free(sw->clf_w); free(sw->gp_train_params);
    free(sw->pareto_buf); free(sw->centers_buf); free(sw->log_c_buf); free(sw->ext_buf); free(sw->kept_buf);
    free(sw->h_cands); free(sw->h_pred); free(sw->kd_idx); free(sw->kd_nodes); free(sw->kd_keep); delete sw;
}
static void sobol_fallback(SweepState *sw,float *out,int fixed,float fixed_cost) {
    uint32_t n=(uint32_t)sw->quasi_index++;
    uint32_t gray=n^(n>>1);
    for (int d=0; d<sw->dim; d++) {
        uint32_t bits=sw->sobol_scramble[d];
        for (int j=0; j<32; j++)
            if ((gray>>j)&1) bits^=sw->sobol_directions[d*32+j];
        out[d]=2.0f*((bits+0.5)/4294967296.0)-1.0f;
    }
    if(sw->cost_dim>=0) out[sw->cost_dim]=fixed?fixed_cost:fmaxf(-1,fminf(1,sw->cost_random_suggestion+sample_normal(sw)));
}
static float logit_transform(float value) {
    value = fmaxf(1e-9f, fminf(1.0f - 1e-9f, value));
    return fmaxf(-5.0f, logf(value / (1.0f - value)));
}

static void sweep_observe(SweepState *sw, const float *norm_params,
        float score, float cost, int is_failure) {
    if (sw->use_logit) {
        score = logit_transform(score);
    }
    int dim = sw->dim;
    if (is_failure || !isfinite(score)) {
        if (sw->fail_n >= sw->failure_cap) {
            return;
        }
        int i = sw->fail_n++;
        memcpy(&sw->fail_params[i * dim], norm_params, dim * sizeof(float));
        sw->fail_scores[i] = score;
        sw->fail_costs[i] = cost;
        return;
    }
    for (int i = 0; i < sw->succ_n; i++) {
        float dist2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = norm_params[d] - sw->succ_params[i * dim + d];
            dist2 += diff * diff;
        }
        if (dist2 < PROTEIN_EPSILON * PROTEIN_EPSILON) {
            memcpy(&sw->succ_params[i * dim], norm_params, dim * sizeof(float));
            sw->succ_scores[i] = score;
            sw->succ_costs[i] = cost;
            return;
        }
    }
    if (sw->cost_dim >= 0 && norm_params[sw->cost_dim] <= -1.0f) {
        return;
    }
    if (sw->succ_n >= sw->success_cap) {
        return;
    }
    int idx = sw->succ_n++;
    memcpy(&sw->succ_params[idx * dim], norm_params, dim * sizeof(float));
    sw->succ_scores[idx] = score;
    sw->succ_costs[idx] = cost;
    if (sw->n_top >= sw->top_k
            && score <= sw->succ_scores[sw->top_idx[sw->n_top - 1]]) {
        return;
    }
    if (sw->n_top < sw->top_k) {
        sw->top_idx[sw->n_top++] = idx;
    } else {
        sw->top_idx[sw->n_top - 1] = idx;
    }
    for (int j = sw->n_top - 1; j > 0; j--) {
        if (sw->succ_scores[sw->top_idx[j]] <= sw->succ_scores[sw->top_idx[j - 1]]) {
            break;
        }
        int tmp = sw->top_idx[j];
        sw->top_idx[j] = sw->top_idx[j - 1];
        sw->top_idx[j - 1] = tmp;
    }
}

static int sweep_should_stop(const SweepState *sw, float score, float cost) {
    if (sw->use_logit) {
        score = logit_transform(score);
    }
    if (!sw->cm_fitted) {
        return 0;
    }
    float min_allowed = sw->cm_upper * 0.3f + 10.0f;
    if (cost < min_allowed) {
        return 0;
    }
    float threshold;
    if (cost > PROTEIN_THRESHOLD_COST_CAP * sw->cm_upper) {
        threshold = PROTEIN_THRESHOLD_FALLBACK * sw->cm_max_score;
    } else {
        threshold = sw->cm_A + sw->cm_B * logf(cost);
    }
    return score < threshold;
}

// KD Tree near-duplicate filter
typedef struct {
    const float *X;
    int dim, n_nodes;
    int *idx;
    KDNode *nodes;
} KDTree;

static int g_kd_split_dim, g_kd_d;
static const float *g_kd_X;

static int kd_cmp_fn(const void *a, const void *b) {
    float va = g_kd_X[*(const int *)a * g_kd_d + g_kd_split_dim];
    float vb = g_kd_X[*(const int *)b * g_kd_d + g_kd_split_dim];
    return (va > vb) - (va < vb);
}

static void kd_build(KDTree *t, int node, int lo, int hi) {
    KDNode *nd = &t->nodes[node];
    nd->lo = lo;
    nd->hi = hi;
    nd->left = nd->right = -1;
    nd->split_dim = -1;
    if (hi - lo <= KD_LEAF) {
        return;
    }
    int best = 0;
    float spread = -1.0;
    for (int k = 0; k < t->dim; k++) {
        float mn = t->X[t->idx[lo] * t->dim + k], mx = mn;
        for (int i = lo + 1; i < hi; i++) {
            float v = t->X[t->idx[i] * t->dim + k];
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        if (mx - mn > spread) {
            spread = mx - mn;
            best = k;
        }
    }
    int mid = (lo + hi) / 2;
    g_kd_split_dim = best;
    g_kd_d = t->dim;
    g_kd_X = t->X;
    qsort(t->idx + lo, hi - lo, sizeof(int), kd_cmp_fn);
    nd->split_dim = best;
    nd->split_val = t->X[t->idx[mid] * t->dim + best];
    int ln = t->n_nodes++, rn = t->n_nodes++;
    nd->left = ln;
    nd->right = rn;
    kd_build(t, ln, lo, mid);
    kd_build(t, rn, mid, hi);
}

static void kd_remove_near(const KDTree *t, int node, const float *q, float r,
        float r2, int self, int *keep) {
    const KDNode *nd = &t->nodes[node];
    if (nd->split_dim < 0) {
        for (int i = nd->lo; i < nd->hi; i++) {
            int p = t->idx[i];
            float sq = 0.0;
            for (int k = 0; k < t->dim; k++) {
                float df = q[k] - t->X[p * t->dim + k];
                sq += df * df;
            }
            if (p != self && sq <= r2) {
                keep[p] = 0;
            }
        }
        return;
    }
    float dv = q[nd->split_dim] - nd->split_val;
    if (dv <= r) {
        kd_remove_near(t, nd->left, q, r, r2, self, keep);
    }
    if (dv >= -r) {
        kd_remove_near(t, nd->right, q, r, r2, self, keep);
    }
}

static int filter_near_duplicates(SweepState *sw, const float *X,
        int n, int dim, float threshold, int *kept_indices) {
    KDTree tree = {
        .X = X, .dim = dim, .n_nodes = 1, .idx = sw->kd_idx, .nodes = sw->kd_nodes};
    for (int i = 0; i < n; i++) {
        sw->kd_idx[i] = i;
    }
    kd_build(&tree, 0, 0, n);
    float r2 = threshold * threshold;
    for (int i = 0; i < n; i++) {
        sw->kd_keep[i] = 1;
    }
    for (int i = n - 1; i >= 0; i--) {
        if (sw->kd_keep[i]) {
            kd_remove_near(&tree, 0, &X[i * dim], threshold, r2, i, sw->kd_keep);
        }
    }
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (sw->kd_keep[i]) {
            kept_indices[count++] = i;
        }
    }
    return count;
}

static int float_cmp(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float pinball(float a, float b, const float *x,
        const float *y, int n, float q) {
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float r = y[i] - (a + b * x[i]);
        loss += (r > 0.0f) ? q * r : (q - 1.0f) * r;
    }
    return loss;
}

static const float *g_pareto_costs;
static int pareto_cmp(const void *a, const void *b) {
    float ca = g_pareto_costs[*(const int *)a];
    float cb = g_pareto_costs[*(const int *)b];
    return (ca > cb) - (ca < cb);
}

static ProteinSweepInfo sweep_suggest(SweepState *sw,
        float *out, float fixed_cost_norm) {
    ProteinSweepInfo info = {0};
    sw->suggestion_idx++;
    int dim = sw->dim, cost_dim = sw->cost_dim;
    int is_fixed_cost = !isnan(fixed_cost_norm);

    if (sw->suggestion_idx <= sw->num_random_samples || sw->succ_n == 0) {
        sobol_fallback(sw, out, is_fixed_cost, fixed_cost_norm);
        info.is_random = 1;
        return info;
    }

    // sample observations for GP
    sw->min_score = sw->max_score = sw->succ_scores[0];
    sw->log_c_min = logf(fmaxf(sw->succ_costs[0], PROTEIN_EPSILON));
    sw->log_c_buf[0] = sw->log_c_min;
    for (int i = 1; i < sw->succ_n; i++) {
        float s = sw->succ_scores[i];
        sw->min_score = fminf(sw->min_score, s);
        sw->max_score = fmaxf(sw->max_score, s);
        sw->log_c_buf[i] = logf(fmaxf(sw->succ_costs[i], PROTEIN_EPSILON));
        sw->log_c_min = fminf(sw->log_c_min, sw->log_c_buf[i]);
    }
    qsort(sw->log_c_buf, sw->succ_n, sizeof(float), float_cmp);
    float qidx = PROTEIN_COST_QUANTILE * (sw->succ_n - 1);
    int qlo = (int)qidx, qhi = qlo + 1 < sw->succ_n ? qlo + 1 : sw->succ_n - 1;
    float qfrac = qidx - qlo;
    sw->log_c_max = sw->log_c_buf[qlo] * (1.0f - qfrac) + sw->log_c_buf[qhi] * qfrac;

    int use_failures = (sw->succ_n < PROTEIN_MIN_OBS_NO_FAIL && sw->fail_n > 0);
    int combined_n = use_failures ? sw->fail_n + sw->succ_n : sw->succ_n;
    int ext_dim = dim + 2;
    int ci = 0;
    if (use_failures) {
        for (int i = 0; i < sw->fail_n; i++) {
            memcpy(&sw->ext_buf[ci * ext_dim], &sw->fail_params[i * dim],
                dim * sizeof(float));
            sw->ext_buf[ci * ext_dim + dim] = sw->min_score;
            sw->ext_buf[ci * ext_dim + dim + 1] = sw->fail_costs[i];
            ci++;
        }
    }
    for (int i = 0; i < sw->succ_n; i++) {
        memcpy(&sw->ext_buf[ci * ext_dim], &sw->succ_params[i * dim],
            dim * sizeof(float));
        sw->ext_buf[ci * ext_dim + dim] = sw->succ_scores[i];
        sw->ext_buf[ci * ext_dim + dim + 1] = sw->succ_costs[i];
        ci++;
    }
    int n_kept = filter_near_duplicates(
        sw, sw->ext_buf, combined_n, ext_dim, PROTEIN_EPSILON, sw->kept_buf);
    if (n_kept > sw->gp_max_obs) {
        int recent_size = (int)(0.5f * sw->gp_max_obs);
        int older_size = n_kept - recent_size;
        int num_sample = sw->gp_max_obs - recent_size;
        for (int i = 0; i < num_sample; i++) {
            int j = i + (int)(sw->rng() % (older_size - i));
            int tmp = sw->kept_buf[i];
            sw->kept_buf[i] = sw->kept_buf[j];
            sw->kept_buf[j] = tmp;
        }
        memmove(sw->kept_buf + num_sample, sw->kept_buf + older_size,
            recent_size * sizeof(int));
        n_kept = sw->gp_max_obs;
    }
    for (int i = 0; i < n_kept; i++) {
        int idx = sw->kept_buf[i];
        memcpy(&sw->gp_train_params[i * dim], &sw->ext_buf[idx * ext_dim],
            dim * sizeof(float));
        sw->gp_train_y[i] = sw->ext_buf[idx * ext_dim + dim];
        sw->gp_train_c[i] = sw->ext_buf[idx * ext_dim + dim + 1];
    }
    int n_gp = n_kept;

    float min_score = sw->min_score, max_score = sw->max_score;
    float log_c_min = sw->log_c_min, log_c_max = sw->log_c_max;
    float score_range = fabsf(max_score - min_score) + PROTEIN_EPSILON;
    float cost_range = log_c_max - log_c_min + PROTEIN_EPSILON;
    for (int i = 0; i < n_gp; i++) {
        sw->gp_train_y[i] = (sw->gp_train_y[i] - min_score) / score_range;
        float lc = logf(fmaxf(sw->gp_train_c[i], PROTEIN_EPSILON));
        sw->gp_train_c[i] = (lc - log_c_min) / cost_range;
    }

    float score_loss=gp_train(&sw->gp_score,sw->gp_train_params,sw->gp_train_y,n_gp,sw->gp_training_iter,sw->gp_learning_rate);
    float cost_loss=gp_train(&sw->gp_cost,sw->gp_train_params,sw->gp_train_c,n_gp,sw->gp_training_iter,sw->gp_learning_rate);
    if(sw->optimizer_reset_frequency>0 && sw->suggestion_idx % sw->optimizer_reset_frequency==0) {
        for(GaussianProcess *g : {&sw->gp_score,&sw->gp_cost}) { std::fill(g->m.begin(),g->m.end(),0); std::fill(g->v.begin(),g->v.end(),0); std::fill(g->vmax.begin(),g->vmax.end(),0); g->t=0; }
    }

    // Pareto front + prune
    for (int i = 0; i < sw->succ_n; i++) {
        sw->pruned_buf[i] = i;
    }
    g_pareto_costs = sw->succ_costs;
    qsort(sw->pruned_buf, sw->succ_n, sizeof(int), pareto_cmp);
    int n_pareto = 0;
    float max_ps = -FLT_MAX;
    for (int i = 0; i < sw->succ_n; i++) {
        int idx = sw->pruned_buf[i];
        if (sw->succ_scores[idx] > max_ps + PROTEIN_EPSILON) {
            sw->pareto_buf[n_pareto++] = idx;
            max_ps = sw->succ_scores[idx];
        }
    }
    memcpy(sw->pruned_buf, sw->pareto_buf, n_pareto * sizeof(int));
    int n_pruned = n_pareto;
    if (n_pruned >= 2) {
        float s_min = sw->succ_scores[sw->pruned_buf[0]], s_max = s_min;
        float c_min = sw->succ_costs[sw->pruned_buf[0]], c_max = c_min;
        for (int i = 1; i < n_pruned; i++) {
            float ss = sw->succ_scores[sw->pruned_buf[i]];
            float cc = sw->succ_costs[sw->pruned_buf[i]];
            s_min = fminf(s_min, ss);
            s_max = fmaxf(s_max, ss);
            c_min = fminf(c_min, cc);
            c_max = fmaxf(c_max, cc);
        }
        float s_range = fmaxf(s_max - s_min, PROTEIN_EPSILON);
        float c_range2 = fmaxf(c_max - c_min, PROTEIN_EPSILON);
        float max_pareto_s = sw->succ_scores[sw->pruned_buf[n_pruned - 1]];
        for (int i = n_pruned - 1; i > 1; i--) {
            if (sw->succ_scores[sw->pruned_buf[i - 1]] < 0.98f * max_pareto_s) {
                break;
            }
            float ng = (sw->succ_scores[sw->pruned_buf[i]]
                - sw->succ_scores[sw->pruned_buf[i - 1]]) / s_range;
            float nc = (sw->succ_costs[sw->pruned_buf[i]]
                - sw->succ_costs[sw->pruned_buf[i - 1]]) / c_range2;
            if (ng / (nc + PROTEIN_EPSILON) >= 0.5f) {
                break;
            }
            n_pruned--;
        }
    }

    if (n_pruned > 0) {
        float pruned_max_cost = sw->succ_costs[sw->pruned_buf[n_pruned - 1]];
        if (sw->upper_cost_threshold < 0) {
            sw->upper_cost_threshold = pruned_max_cost;
        } else if (pruned_max_cost > sw->upper_cost_threshold) {
            sw->upper_cost_threshold *= PROTEIN_COST_GROWTH;
        }
    }

    // Cost model fit (pinball NM)
    sw->cm_fitted = 0;
    sw->cm_max_score = max_score;
    sw->cm_upper = sw->upper_cost_threshold;
    int n_valid = 0;
    for (int i = 0; i < sw->succ_n; i++) {
        if (sw->succ_costs[i] > PROTEIN_EPSILON && isfinite(sw->succ_scores[i])) {
            n_valid++;
        }
    }
    if (n_valid >= sw->cm_min_samples) {
        float *x = sw->ext_buf;
        float *y = x + n_valid;
        int j = 0;
        for (int i = 0; i < sw->succ_n; i++) {
            if (sw->succ_costs[i] > PROTEIN_EPSILON && isfinite(sw->succ_scores[i])) {
                x[j] = logf(sw->succ_costs[i]);
                y[j] = sw->succ_scores[i];
                j++;
            }
        }
        float sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < n_valid; i++) {
            sx += x[i];
            sy += y[i];
            sxx += x[i] * x[i];
            sxy += x[i] * y[i];
        }
        float det = n_valid * sxx - sx * sx;
        float a_init = sy / fmaxf(n_valid, 1.0f), b_init = 0.0f;
        if (fabsf(det) >= 1e-10f) {
            b_init = (n_valid * sxy - sx * sy) / det;
            a_init = (sy - b_init * sx) / n_valid;
        }
        float q = sw->early_stop_quantile;
        float nm_x[3] = {a_init, a_init + 0.05f, a_init - 0.05f};
        float nm_y[3] = {b_init, b_init - 0.05f, b_init + 0.05f};
        float nm_v[3];
        for (int i = 0; i < 3; i++) {
            nm_v[i] = pinball(nm_x[i], nm_y[i], x, y, n_valid, q);
        }
        for (int iter = 0; iter < 500; iter++) {
            for (int i = 0; i < 2; i++) {
                for (int jj = i + 1; jj < 3; jj++) {
                    if (nm_v[jj] < nm_v[i]) {
                        float t;
                        t = nm_x[i]; nm_x[i] = nm_x[jj]; nm_x[jj] = t;
                        t = nm_y[i]; nm_y[i] = nm_y[jj]; nm_y[jj] = t;
                        t = nm_v[i]; nm_v[i] = nm_v[jj]; nm_v[jj] = t;
                    }
                }
            }
            if (fabsf(nm_v[2] - nm_v[0]) < 1e-7f) {
                break;
            }
            float cx = (nm_x[0] + nm_x[1]) * 0.5f;
            float cy = (nm_y[0] + nm_y[1]) * 0.5f;
            float rx = cx + (cx - nm_x[2]), ry = cy + (cy - nm_y[2]);
            float rv = pinball(rx, ry, x, y, n_valid, q);
            if (rv < nm_v[1] && rv >= nm_v[0]) {
                nm_x[2] = rx;
                nm_y[2] = ry;
                nm_v[2] = rv;
                continue;
            }
            if (rv < nm_v[0]) {
                float ex = cx + 2.0f * (rx - cx);
                float ey = cy + 2.0f * (ry - cy);
                float ev = pinball(ex, ey, x, y, n_valid, q);
                if (ev >= rv) {
                    ex = rx;
                    ey = ry;
                    ev = rv;
                }
                nm_x[2] = ex;
                nm_y[2] = ey;
                nm_v[2] = ev;
                continue;
            }
            float ccx = (rv < nm_v[2])
                ? cx + 0.5f * (rx - cx) : cx + 0.5f * (nm_x[2] - cx);
            float ccy = (rv < nm_v[2])
                ? cy + 0.5f * (ry - cy) : cy + 0.5f * (nm_y[2] - cy);
            float ccv = pinball(ccx, ccy, x, y, n_valid, q);
            if (ccv < nm_v[2]) {
                nm_x[2] = ccx;
                nm_y[2] = ccy;
                nm_v[2] = ccv;
                continue;
            }
            for (int i = 1; i < 3; i++) {
                nm_x[i] = nm_x[0] + 0.5f * (nm_x[i] - nm_x[0]);
                nm_y[i] = nm_y[0] + 0.5f * (nm_y[i] - nm_y[0]);
                nm_v[i] = pinball(nm_x[i], nm_y[i], x, y, n_valid, q);
            }
        }
        sw->cm_A = nm_x[0];
        sw->cm_B = nm_y[0];
        sw->cm_fitted = 1;
    }

    int *use_pareto = sw->prune_pareto ? sw->pruned_buf : sw->pareto_buf;
    int n_use = sw->prune_pareto ? n_pruned : n_pareto;

    // Search centers
    int n_centers = 0;
    for (int i = 0; i < n_use; i++) {
        memcpy(&sw->centers_buf[n_centers++ * dim],
            &sw->succ_params[use_pareto[i] * dim], dim * sizeof(float));
    }
    float divisors[] = {2.0f, 3.0f};
    int n_shift = (cost_dim >= 0) ? 2 : 1;
    for (int r = 0; r < n_shift; r++) {
        for (int i = 0; i < sw->n_top; i++) {
            const float *src = &sw->succ_params[sw->top_idx[i] * dim];
            float *dst = &sw->centers_buf[n_centers * dim];
            memcpy(dst, src, dim * sizeof(float));
            if (cost_dim >= 0) {
                float orig = src[cost_dim];
                dst[cost_dim] = orig - (orig + 1.0f) / divisors[r];
            }
            n_centers++;
        }
    }

    // Target cost
    float target_cost = 0.0f;
    if (!is_fixed_cost) {
        if (sw->pool_remaining <= 0) {
            for (int i = PROTEIN_NUM_COST_RATIOS - 1; i > 0; i--) {
                int j = (int)(sw->rng() % (i + 1));
                float tmp = sw->ratio_pool[i];
                sw->ratio_pool[i] = sw->ratio_pool[j];
                sw->ratio_pool[j] = tmp;
            }
            sw->pool_remaining = PROTEIN_NUM_COST_RATIOS;
        }
        float ratio = sw->ratio_pool[--sw->pool_remaining];
        ratio = fmaxf(0.0f, fminf(1.0f, ratio + sample_normal(sw)));
        target_cost = (1.0f + sw->expansion_rate) * ratio;
    }

    // Sample candidates
    int n_total = n_centers * sw->suggestions_per_pareto;
    if (n_total > sw->acq_cap) {
        n_total = sw->acq_cap;
    }
    for(int i=0;i<n_total;i++) {
        int center=(int)(sw->rng()%n_centers);
        for(int d=0;d<dim;d++) {
            float v=sw->centers_buf[center*dim+d]+sw->space->spaces[d].scale*sw->global_search_scale*(2*uniform(sw)-1);
            sw->h_cands[i*dim+d]=fmaxf(-1,fminf(1,v));
        }
        if(cost_dim>=0 && is_fixed_cost) sw->h_cands[i*dim+cost_dim]=fixed_cost_norm;
    }
    int n_cands = filter_near_duplicates(
        sw, sw->h_cands, n_total, dim, PROTEIN_EPSILON, sw->kept_buf);
    if (n_cands < n_total) {
        for (int i = 0; i < n_cands; i++) {
            memmove(&sw->h_cands[i * dim], &sw->h_cands[sw->kept_buf[i] * dim],
                dim * sizeof(float));
        }
    }

    if (n_cands == 0) {
        sobol_fallback(sw, out, is_fixed_cost, fixed_cost_norm);
        info.is_random = 1;
        return info;
    }

    // Success classifier
    sw->clf_fitted = 0;
    if (sw->use_success_prob && sw->succ_n > 9 && sw->fail_n > 9) {
        int n_s = sw->succ_n, n_f = sw->fail_n, n_clf = n_s + n_f;
        float *X = sw->ext_buf;
        int *yy = sw->kept_buf;
        memcpy(X, sw->succ_params, n_s * dim * sizeof(float));
        memcpy(X + n_s * dim, sw->fail_params, n_f * dim * sizeof(float));
        for (int i = 0; i < n_s; i++) {
            yy[i] = 1;
        }
        for (int i = 0; i < n_f; i++) {
            yy[n_s + i] = 0;
        }
        int n_pos = 0;
        for (int i = 0; i < n_clf; i++) {
            n_pos += yy[i];
        }
        int n_neg = n_clf - n_pos;
        if (n_pos > 0 && n_neg > 0) {
            float w_pos = n_clf / (2.0f * n_pos);
            float w_neg = n_clf / (2.0f * n_neg);
            memset(sw->clf_w, 0, dim * sizeof(float));
            sw->clf_bias = 0.0f;
            vector<float> grad(dim);
            for (int iter = 0; iter < PROTEIN_CLF_ITERS; iter++) {
                float lr = 0.1f / (1.0f + 0.01f * iter);
                for (int d = 0; d < dim; d++) {
                    grad[d] = 0.0f;
                }
                float grad_b = 0.0f;
                for (int i = 0; i < n_clf; i++) {
                    float z = sw->clf_bias;
                    for (int d = 0; d < dim; d++) {
                        z += X[i * dim + d] * sw->clf_w[d];
                    }
                    float sig = 1.0f / (1.0f + expf(-z));
                    float err = (yy[i] ? w_pos : w_neg) * (sig - yy[i]) / n_clf;
                    for (int d = 0; d < dim; d++) {
                        grad[d] += err * X[i * dim + d];
                    }
                    grad_b += err;
                }
                for (int d = 0; d < dim; d++) {
                    sw->clf_w[d] -= lr * (grad[d] + sw->clf_w[d]);
                }
                sw->clf_bias -= lr * grad_b;
            }
            sw->clf_fitted = 1;
        }
    }

    // GP predict + score/argmax
    for(int i=0;i<n_cands;i++) {
        sw->h_pred[i]=gp_predict(sw->gp_score,&sw->h_cands[i*dim]);
        sw->h_pred[n_cands+i]=gp_predict(sw->gp_cost,&sw->h_cands[i*dim]);
    }

    int best = 0;
    float best_s = -FLT_MAX;
    int opt_dir = sw->space->optimize_direction;
    for (int i = 0; i < n_cands; i++) {
        float s = opt_dir * sw->h_pred[i];
        if (!is_fixed_cost) {
            float cn = sw->h_pred[n_cands + i];
            float c = expf(cn * (log_c_max - log_c_min) + log_c_min);
            s *= (c < sw->max_suggestion_cost ? 1.0f : 0.0f)
                * (1.0f - fabsf(target_cost - cn));
        }
        if (sw->clf_fitted) {
            float z = sw->clf_bias;
            for (int d = 0; d < dim; d++) {
                z += sw->h_cands[i * dim + d] * sw->clf_w[d];
            }
            s *= 1.0f / (1.0f + expf(-z));
        }
        if (s > best_s) {
            best_s = s;
            best = i;
        }
    }
    memcpy(out, &sw->h_cands[best * dim], dim * sizeof(float));
    float y_norm = sw->h_pred[best], c_norm = sw->h_pred[n_cands + best];

    info.score_loss = score_loss;
    info.cost_loss = cost_loss;
    info.predicted_score = y_norm * (max_score - min_score) + min_score;
    info.predicted_cost = expf(c_norm * (log_c_max - log_c_min) + log_c_min);
    info.rating = best_s;
    info.n_pareto = n_use;
    info.n_gp_obs = n_gp;
    info.n_candidates = n_cands;
    return info;
}

extern "C" void protein_sweep_observe(ProteinSweep *s,const float *x,float score,float cost,int failure) { sweep_observe(static_cast<SweepState*>(s),x,score,cost,failure); }
extern "C" int protein_sweep_should_stop(const ProteinSweep *s,float score,float cost) { return sweep_should_stop(static_cast<const SweepState*>(s),score,cost); }
extern "C" ProteinSweepInfo protein_sweep_suggest(ProteinSweep *s,float *out,float fixed) { return sweep_suggest(static_cast<SweepState*>(s),out,fixed); }
