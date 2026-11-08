#ifndef PUFFERLIB_METAL_SHADER_SRC_H
#define PUFFERLIB_METAL_SHADER_SRC_H

static const char *get_all_metal_shader_source() {
  return R"METAL(
#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_atomic>
using namespace metal;

inline float sigmoid_f(float x) {
    float z = exp(-abs(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

inline float sigmoid_backward_f(float x, float grad_output) {
    float sig = sigmoid_f(x);
    return grad_output * sig * (1.0f - sig);
}

inline float fast_tanh_f(float x) {
    float v1 = clamp(x, -9.0f, 9.0f);
    float v2 = v1 * v1;
    // Horner polynomial (matches PyTorch implementation)
    float p = v2 * (-2.76076847742355e-16f) + 2.00018790482477e-13f;
    p = v2 * p + (-8.60467152213735e-11f);
    p = v2 * p + 5.12229709037114e-08f;
    p = v2 * p + 1.48572235717979e-05f;
    p = v2 * p + 6.37261928875436e-04f;
    p = v2 * p + 4.89352455891786e-03f;
    p = v1 * p;
    float q = v2 * 1.19825839466702e-06f + 1.18534705686654e-04f;
    q = v2 * q + 2.26843463243900e-03f;
    q = v2 * q + 4.89352518554385e-03f;
    return p / q;
}

inline float fast_sigmoid_f(float x) {
    float y = fast_tanh_f(x * 0.5f);
    return clamp((y + 1.0f) * 0.5f, 0.0f, 1.0f);
}

inline float tilde_relu_fwd(float x) {
    return x >= 0.0f ? x + 0.5f : fast_sigmoid_f(x);
}

inline float tilde_relu_bwd(float x, float grad) {
    if (x >= 0.0f) return grad;
    float sig = fast_sigmoid_f(x);
    return grad * sig * (1.0f - sig);
}

inline float lerp_f(float a, float b, float w) {
    float diff = b - a;
    return abs(w) < 0.5f ? a + w * diff : b - diff * (1.0f - w);
}

// MSL has no built-in log1p. Goldberg's trick: compensates for rounding in 1+x.
inline float log1p_f(float x) {
    float u = 1.0f + x;
    return (u == 1.0f) ? x : log(u) * x / (u - 1.0f);
}

constant float SOFTPLUS_BETA = 1.0f;
constant float SOFTPLUS_THRESHOLD = 20.0f;

inline float softplus_fwd(float x) {
    float xs = x * SOFTPLUS_BETA;
    return xs > SOFTPLUS_THRESHOLD ? x : log1p_f(exp(xs)) / SOFTPLUS_BETA;
}

inline float softplus_bwd(float grad_output, float x) {
    float beta_x = SOFTPLUS_BETA * x;
    if (beta_x > SOFTPLUS_THRESHOLD) return grad_output;
    float exp_beta_x = exp(beta_x);
    return grad_output * (exp_beta_x / (1.0f + exp_beta_x));
}

inline void log_coeffs_and_values_fwd(float gate, float hidden,
                                       thread float& log_coeff, thread float& log_value) {
    float abs_gate = abs(gate);
    float sp_neg = log1p_f(exp(-abs_gate));
    float softplus_gate, softplus_neg_gate;
    if (gate >= 0.0f) {
        softplus_gate = gate + sp_neg;
        softplus_neg_gate = sp_neg;
    } else {
        softplus_gate = sp_neg;
        softplus_neg_gate = -gate + sp_neg;
    }
    log_coeff = -softplus_gate;
    float log_z = -softplus_neg_gate;
    float log_tilde_h = hidden >= 0.0f ? log(hidden + 0.5f) : -softplus_fwd(-hidden);
    log_value = log_z + log_tilde_h;
}

inline void log_coeffs_and_values_bwd(float grad_lc, float grad_lv,
                                       float gate, float hidden,
                                       thread float& grad_gate, thread float& grad_hidden) {
    float sig_gate = sigmoid_f(gate);
    grad_gate = -grad_lc * sig_gate + grad_lv * (1.0f - sig_gate);
    if (hidden >= 0.0f) {
        grad_hidden = grad_lv / (hidden + 0.5f);
    } else {
        grad_hidden = grad_lv * sigmoid_f(-hidden);
    }
}

inline float relu_f(float x) { return max(0.0f, x); }
inline float relu_backward_f(float x, float grad_output) { return (x > 0.0f) ? grad_output : 0.0f; }

// Must match the host MetalRngState layout in metal_kernels.mm.
struct MetalRngState {
    ulong seed;
    uint x, y, z, w, lane, has_normal;
    float normal;
    uint pad;
};

inline uint4 philox4x32_round(uint4 ctr, uint2 key) {
    constexpr uint PHILOX_M0 = 0xD2511F53u;
    constexpr uint PHILOX_M1 = 0xCD9E8D57u;
    uint hi0 = mulhi(PHILOX_M0, ctr.x);
    uint lo0 = PHILOX_M0 * ctr.x;
    uint hi1 = mulhi(PHILOX_M1, ctr.z);
    uint lo1 = PHILOX_M1 * ctr.z;
    return uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

inline uint4 philox4x32_10(uint4 counter, uint2 key) {
    constexpr uint PHILOX_W0 = 0x9E3779B9u;
    constexpr uint PHILOX_W1 = 0xBB67AE85u;
    for (int i = 0; i < 10; i++) {
        counter = philox4x32_round(counter, key);
        key.x += PHILOX_W0;
        key.y += PHILOX_W1;
    }
    return counter;
}

inline float philox_uniform_value(uint val) {
    return float(val) * 0x1p-32f + 0x1p-33f;
}

inline float philox_uniform(thread uint& state_idx, uint4 rng_out) {
    uint val;
    switch (state_idx & 3) {
        case 0: val = rng_out.x; break;
        case 1: val = rng_out.y; break;
        case 2: val = rng_out.z; break;
        default: val = rng_out.w; break;
    }
    state_idx++;
    // Match cuRAND's uint32-to-float mapping, including its positive minimum.
    return philox_uniform_value(val);
}

// Advance the same 128-bit Philox counter and four-value lane sequence as
// cuRAND's per-thread Philox state. The normal spare survives dispatches.
inline uint philox_next(thread MetalRngState& state, thread uint4& output) {
    uint value = output[state.lane++];
    if (state.lane == 4) {
        state.lane = 0;
        if (++state.x == 0 && ++state.y == 0 && ++state.z == 0) ++state.w;
        output = philox4x32_10(uint4(state.x, state.y, state.z, state.w),
                              uint2(uint(state.seed), uint(state.seed >> 32)));
    }
    return value;
}

inline float philox_normal(thread MetalRngState& state, thread uint4& output) {
    if (state.has_normal) {
        state.has_normal = 0;
        return state.normal;
    }
    uint x = philox_next(state, output);
    uint y = philox_next(state, output);
    float u = float(x) * 0x1p-32f + 0x1p-33f;
    constexpr float angle_scale = 1.4629180792671596e-9f;
    float angle = float(y) * angle_scale + 0.5f * angle_scale;
    float radius = sqrt(-2.0f * log(u));
    float cosine;
    float sine = sincos(angle, cosine);
    state.normal = radius * cosine;
    state.has_normal = 1;
    return radius * sine;
}

// mingru_gate_inference: fused chunk + tilde_relu + lerp + highway output gate
// combined is (B, 3*H) = [hidden, gate, proj], state is (B, H)
// x_in is (B, H) = input before projection
// out = sigmoid(proj) * mingru_out + (1 - sigmoid(proj)) * x_in
// next_state = mingru_out
struct MingruGateParams {
    int H;
    int B;
};

kernel void mingru_gate_inference(
    device float* out               [[buffer(0)]],
    device float* next_state        [[buffer(1)]],
    const device float* combined    [[buffer(2)]],
    const device float* state_in    [[buffer(3)]],
    const device float* x_in        [[buffer(4)]],
    constant MingruGateParams& p    [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    int N = p.B * p.H;
    if ((int)idx >= N) return;

    int b = (int)idx / p.H;
    int h = (int)idx % p.H;
    int base = b * 3 * p.H;

    float hidden = combined[base + h];
    float gate = combined[base + p.H + h];
    float proj = combined[base + 2 * p.H + h];
    float state = state_in[idx];
    float x = x_in[idx];

    float gate_sig = sigmoid_f(gate);
    float hidden_tilde = hidden >= 0.0f ? hidden + 0.5f : sigmoid_f(hidden);
    float mingru_out = lerp_f(state, hidden_tilde, gate_sig);
    float proj_sig = sigmoid_f(proj);

    next_state[idx] = mingru_out;
    out[idx] = proj_sig * mingru_out + (1.0f - proj_sig) * x;
}

constant int CHECKPOINT_INTERVAL = 4;

struct ScanParams {
    int T_seq;
    int H;
    int B;
};

kernel void mingru_scan_forward_checkpointed(
    device float* out               [[buffer(0)]],
    device float* next_state        [[buffer(1)]],
    device float* a_star_buf        [[buffer(2)]],
    device float* s_buf             [[buffer(3)]],
    device float* log_values_buf    [[buffer(4)]],
    const device float* combined    [[buffer(5)]],
    const device float* state       [[buffer(6)]],
    const device float* input       [[buffer(7)]],
    constant ScanParams& p          [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.B * p.H) return;

    int b = (int)idx / p.H;
    int h = (int)idx % p.H;
    int bH = b * p.H;
    int H3 = 3 * p.H;
    int bHT = bH * p.T_seq;
    int out_base = bHT + h;
    int cbase = 3 * bHT;

    float a_star = 0.0f;
    float log_value = 0.0f;
    // fast:: matches CUDA __expf/__logf (kernels.cu:270-305)
    float s = fast::log(state[bH + h]);
    log_value = s;

    int T_out = p.T_seq + 1;
    int buf_base = b * T_out * p.H + h;
    int buf_curr = buf_base;
    a_star_buf[buf_curr] = a_star;
    s_buf[buf_curr] = s;
    log_values_buf[buf_curr] = log_value;

    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 1; t < p.T_seq + 1; t++) {
        float hidden_val = combined[cbase + h + t_offset];
        float gate_val = combined[cbase + p.H + h + t_offset];
        float proj_val = combined[cbase + 2 * p.H + h + t_offset];
        float x_val = input[out_base + (t - 1) * p.H];

        float log_coeff_val;
        log_coeffs_and_values_fwd(gate_val, hidden_val, log_coeff_val, log_value);

        a_star += log_coeff_val;

        float z = log_value - a_star;
        float max_val = fmax(s, z);
        s = max_val + log1p_f(fast::exp(-abs(s - z)));

        float scan_result = fast::exp(a_star + s);
        float proj_sigmoid = sigmoid_f(proj_val);

        out[out_curr] = proj_sigmoid * scan_result + (1.0f - proj_sigmoid) * x_val;

        buf_curr += p.H;
        out_curr += p.H;
        t_offset += H3;

        if (t % CHECKPOINT_INTERVAL == 0) {
            a_star_buf[buf_curr] = a_star;
            s_buf[buf_curr] = s;
            log_values_buf[buf_curr] = log_value;
        }
    }

    // Floor at 1e-30 to prevent log(0)=-inf on the next forward pass.
    // exp(a_star+s) underflows to exactly 0.0f in fp32 when a_star+s < -87.3.
    // A zero state causes log(0)=-inf → permanent -inf propagation through
    // all subsequent scan steps. 1e-30 is well above fp32 denormal range
    // and below any meaningful state value.
    next_state[bH + h] = max(fast::exp(a_star + s), 1e-30f);
}

kernel void mingru_scan_backward_checkpointed(
    device float* grad_combined          [[buffer(0)]],
    device float* grad_state             [[buffer(1)]],
    device float* grad_input             [[buffer(2)]],
    const device float* grad_out         [[buffer(3)]],
    const device float* grad_next_state  [[buffer(4)]],
    const device float* combined         [[buffer(5)]],
    const device float* state            [[buffer(6)]],
    const device float* input            [[buffer(7)]],
    const device float* a_star_buf       [[buffer(8)]],
    const device float* s_buf            [[buffer(9)]],
    const device float* log_values_buf   [[buffer(10)]],
    constant ScanParams& p               [[buffer(11)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.B * p.H) return;

    int b = (int)idx / p.H;
    int h = (int)idx % p.H;
    int bHT = b * p.H * p.T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * p.H;
    int state_idx = b * p.H + h;
    int out_base = bHT + h;

    int T_out = p.T_seq + 1;
    int buf_base = b * T_out * p.H + h;

    float acc = 0.0f;
    float s_val_next = 0.0f;
    float carry_grad_a = 0.0f;

    for (int chunk_end = p.T_seq; chunk_end > 0; chunk_end -= CHECKPOINT_INTERVAL) {
        int chunk_start = (chunk_end > CHECKPOINT_INTERVAL) ? (chunk_end - CHECKPOINT_INTERVAL) : 0;
        int chunk_len = chunk_end - chunk_start;

        // Chunk storage in thread-local arrays
        float chunk_a_star[CHECKPOINT_INTERVAL];
        float chunk_s[CHECKPOINT_INTERVAL];
        float chunk_log_values[CHECKPOINT_INTERVAL];
        float chunk_hidden[CHECKPOINT_INTERVAL];
        float chunk_gate[CHECKPOINT_INTERVAL];

        // Load checkpoint
        int ckpt_buf_idx = buf_base + chunk_start * p.H;
        float recomp_a_star = a_star_buf[ckpt_buf_idx];
        float recomp_s = s_buf[ckpt_buf_idx];
        float recomp_log_value = log_values_buf[ckpt_buf_idx];

        // Forward recompute within chunk
        for (int i = 0; i < chunk_len; i++) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;
            float hv = combined[cbase + h + t_offset];
            float gv = combined[cbase + p.H + h + t_offset];

            float lc;
            log_coeffs_and_values_fwd(gv, hv, lc, recomp_log_value);
            recomp_a_star += lc;

            float z = recomp_log_value - recomp_a_star;
            float mv = fmax(recomp_s, z);
            recomp_s = mv + log1p_f(fast::exp(-abs(recomp_s - z)));

            chunk_a_star[i] = recomp_a_star;
            chunk_s[i] = recomp_s;
            chunk_log_values[i] = recomp_log_value;
            chunk_hidden[i] = hv;
            chunk_gate[i] = gv;
        }

        // Backward through chunk
        for (int i = chunk_len - 1; i >= 0; i--) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;

            float a_star_t = chunk_a_star[i];
            float s_t = chunk_s[i];
            float log_value_t = chunk_log_values[i];
            float hidden_val = chunk_hidden[i];
            float gate_val = chunk_gate[i];
            float proj_val = combined[cbase + 2 * p.H + h + t_offset];
            int input_idx = out_base + (t - 1) * p.H;
            float x_val = input[input_idx];

            float scan_result = fast::exp(a_star_t + s_t);
            float z = log_value_t - a_star_t;

            float grad_out_val = grad_out[input_idx];
            float grad_scan_from_next = (t == p.T_seq) ? grad_next_state[state_idx] : 0.0f;

            float proj_sigmoid = sigmoid_f(proj_val);
            float grad_scan_result = grad_scan_from_next + grad_out_val * proj_sigmoid;
            float grad_proj = grad_out_val * (scan_result - x_val) * proj_sigmoid * (1.0f - proj_sigmoid);
            grad_input[input_idx] = grad_out_val * (1.0f - proj_sigmoid);

            float grad_log_h = grad_scan_result * scan_result;
            float grad_s = grad_log_h;

            if (t == p.T_seq) {
                acc = grad_s;
            } else {
                acc = grad_s + acc * fast::exp(s_t - s_val_next);
            }
            float grad_z = acc * fast::exp(z - s_t);
            s_val_next = s_t;

            float grad_a = grad_log_h + carry_grad_a - grad_z;
            carry_grad_a = grad_a;

            float grad_g, grad_h;
            log_coeffs_and_values_bwd(grad_a, grad_z, gate_val, hidden_val, grad_g, grad_h);

            grad_combined[cbase + h + t_offset] = grad_h;
            grad_combined[cbase + p.H + h + t_offset] = grad_g;
            grad_combined[cbase + 2 * p.H + h + t_offset] = grad_proj;
        }
    }

    // Gradient for initial state (t=0)
    int ckpt_0_idx = buf_base;
    float a_star_0 = a_star_buf[ckpt_0_idx];
    float s_0 = s_buf[ckpt_0_idx];
    float log_value_0 = log_values_buf[ckpt_0_idx];

    acc = acc * fast::exp(s_0 - s_val_next);
    float grad_z_0 = acc * fast::exp((log_value_0 - a_star_0) - s_0);

    grad_state[state_idx] = (state[state_idx] > 0.0f) ? (grad_z_0 / state[state_idx]) : 0.0f;
}

struct SampleParams {
    // Random state is held per agent in buffer(10).
    int num_atns;
    int num_atns_total;  // sum of act_sizes
    int B;
    int logits_stride;
    int logstd_stride;
    int value_stride;
    int is_continuous;  // 1 for continuous, 0 for discrete
    int mask_stride;    // stride between rows in mask buffer (may differ from num_atns_total)
    int head_stride;    // zero means all heads are consumed
    float verb_epsilon;
};

inline float masked_logit(float l, float m) {
    if (m == 0.0f) l = -1.0e4f;
    return l;
}

kernel void sample_logits_kernel(
    device float* actions               [[buffer(0)]],
    device float* logprobs              [[buffer(1)]],
    device float* value_out             [[buffer(2)]],
    const device float* logits          [[buffer(3)]],
    const device float* logstd          [[buffer(4)]],
    const device float* value           [[buffer(5)]],
    const device int* act_sizes         [[buffer(6)]],
    constant SampleParams& sp           [[buffer(7)]],
    const device float* action_mask     [[buffer(8)]],
    const device char* head_map         [[buffer(9)]],
    device MetalRngState* rng_states    [[buffer(10)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= sp.B) return;

    MetalRngState rng = rng_states[idx];
    uint4 rng_out = philox4x32_10(uint4(rng.x, rng.y, rng.z, rng.w),
                                  uint2(uint(rng.seed), uint(rng.seed >> 32)));

    int logits_base = (int)idx * sp.logits_stride;
    float total_log_prob = 0.0f;

    if (sp.is_continuous) {
        constexpr float LOG_2PI = 1.8378770664093453f;
        int logstd_base = (int)idx * sp.logstd_stride;

        for (int h = 0; h < sp.num_atns; h++) {
            float raw_mean = logits[logits_base + h];
            float raw_log_std = logstd[logstd_base + h];
            float mean = isnan(raw_mean) ? 0.0f : clamp(raw_mean, -1.0e6f, 1.0e6f);
            float log_std = isnan(raw_log_std) ? 0.0f : clamp(raw_log_std, -20.0f, 2.0f);
            float std = exp(log_std);

            float noise = philox_normal(rng, rng_out);
            float sampled = mean + std * noise;
            float action = isnan(sampled) ? 0.0f : clamp(sampled, -1.0e6f, 1.0e6f);

            float normalized = (action - mean) / std;
            float log_prob = -0.5f * normalized * normalized - 0.5f * LOG_2PI - log_std;

            actions[(int)idx * sp.num_atns + h] = action;
            total_log_prob += log_prob;
        }
    } else {
        // Discrete action sampling (multinomial)
        // CUDA joint-ratio: accumulate scalar total_log_prob across heads
        int logits_offset = 0;

        for (int h = 0; h < sp.num_atns; h++) {
            int A = act_sizes[h];

            // Mask base index for this env (mask_stride allows non-contiguous layout)
            int mask_base = (int)idx * sp.mask_stride;

            // Max + logsumexp (with mask)
            float max_val = -INFINITY;
            for (int a = 0; a < A; a++) {
                float l = masked_logit(logits[logits_base + logits_offset + a],
                                       action_mask[mask_base + logits_offset + a]);
                max_val = fmax(max_val, l);
            }
            float sum_exp = 0.0f;
            for (int a = 0; a < A; a++) {
                float l = masked_logit(logits[logits_base + logits_offset + a],
                                       action_mask[mask_base + logits_offset + a]);
                sum_exp += exp(l - max_val);
            }
            float logsumexp_val = max_val + log(sum_exp);
            float eps = h == 0 ? sp.verb_epsilon : 0.0f;
            int legal_count = 0;
            if (eps > 0.0f) {
                for (int a = 0; a < A; ++a)
                    legal_count += action_mask[mask_base + logits_offset + a] != 0.0f;
                if (legal_count == 0) legal_count = A;
            }

            // Random sample
            float rand_val = philox_uniform_value(philox_next(rng, rng_out));

            // Inverse CDF sampling (with mask)
            float cumsum = 0.0f;
            int sampled_action = A - 1;
            for (int a = 0; a < A; a++) {
                float l = masked_logit(logits[logits_base + logits_offset + a],
                                       action_mask[mask_base + logits_offset + a]);
                float prob = exp(l - logsumexp_val);
                cumsum += eps > 0.0f
                    ? (1.0f - eps) * prob + eps *
                        (action_mask[mask_base + logits_offset + a] != 0.0f ? 1.0f : 0.0f) / float(legal_count)
                    : prob;
                if (rand_val < cumsum) {
                    sampled_action = a;
                    break;
                }
            }
            if (sampled_action == A - 1) {
                for (int a = A - 1; a >= 0; --a) {
                    if (action_mask[mask_base + logits_offset + a] != 0.0f) {
                        sampled_action = a;
                        break;
                    }
                }
            }

            float log_prob = masked_logit(
                logits[logits_base + logits_offset + sampled_action],
                action_mask[mask_base + logits_offset + sampled_action]) - logsumexp_val;

            actions[(int)idx * sp.num_atns + h] = float(sampled_action);
            int verb = int(actions[(int)idx * sp.num_atns]);
            if (h == 0 || sp.head_stride == 0 || head_map[verb * sp.head_stride + h]) {
                total_log_prob += eps > 0.0f
                    ? log((1.0f - eps) * exp(log_prob) + eps / float(legal_count))
                    : log_prob;
            }

            logits_offset += A;
        }
    }
    // Scalar joint logprob (matches CUDA kernels.cu:995)
    logprobs[(int)idx] = total_log_prob;
    value_out[idx] = value[(int)idx * sp.value_stride];
    rng_states[idx] = rng;
}

//
// Used when CPU inference produces actions but logprobs need GPU-precision
// exp/log to match PPO training kernels. Without this, the importance ratio
// exp(new_logp - old_logp) sees a systematic IEEE vs fast::exp bias that
// causes NaN with overlap (stale weights amplify the mismatch).

struct RecomputeLogprobsParams {
    int B;
    int num_atns;
    int logits_stride;   // fused_cols per row
    int mask_stride;     // 0 = broadcast (all-ones)
};

kernel void recompute_logprobs_kernel(
    device float* logprobs              [[buffer(0)]],
    const device float* logits          [[buffer(1)]],
    const device float* actions_f32     [[buffer(2)]],
    const device int* act_sizes         [[buffer(3)]],
    const device float* action_mask     [[buffer(4)]],
    constant RecomputeLogprobsParams& rp [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= rp.B) return;

    int logits_base = (int)idx * rp.logits_stride;
    int mask_base = (rp.mask_stride == 0) ? 0 : (int)idx * rp.mask_stride;

    float total_log_prob = 0.0f;
    int logits_offset = 0;

    for (int h = 0; h < rp.num_atns; h++) {
        int A = act_sizes[h];
        int act = int(actions_f32[(int)idx * rp.num_atns + h]);

        // Max + logsumexp (with mask)
        float max_val = -INFINITY;
        bool has_valid_action = false;
        for (int a = 0; a < A; a++) {
            has_valid_action = has_valid_action || action_mask[mask_base + logits_offset + a] >= 0.5f;
            max_val = fmax(max_val, masked_logit(
                logits[logits_base + logits_offset + a],
                action_mask[mask_base + logits_offset + a]));
        }
        if (!has_valid_action) {
            total_log_prob = NAN;
            logits_offset += A;
            continue;
        }
        float sum_exp = 0.0f;
        for (int a = 0; a < A; a++) {
            sum_exp += exp(masked_logit(
                logits[logits_base + logits_offset + a],
                action_mask[mask_base + logits_offset + a]) - max_val);
        }
        float logsumexp_val = max_val + log(sum_exp);

        float head_lp = masked_logit(
            logits[logits_base + logits_offset + act],
            action_mask[mask_base + logits_offset + act]) - logsumexp_val;
        total_log_prob += head_lp;

        logits_offset += A;
    }
    // Scalar joint logprob (matches CUDA kernels.cu:995)
    logprobs[(int)idx] = total_log_prob;
}

constant int PPO_THREADS = 256;
constant int LOSS_PG = 0;
constant int LOSS_VF = 1;
constant int LOSS_ENT = 2;
constant int LOSS_TOTAL = 3;
constant int LOSS_OLD_APPROX_KL = 4;
constant int LOSS_APPROX_KL = 5;
constant int LOSS_CLIPFRAC = 6;
constant int LOSS_N = 7;
constant int MAX_ATN_HEADS = 16;

// Float atomic add via CAS loop (MSL has no native atomic<float>)
inline void atomic_add_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        float desired = current + val;
        uint desired_bits = as_type<uint>(desired);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, desired_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

inline void ppo_discrete_head(
    const device float* logits,
    int logits_base, int logits_stride_a, int logits_offset,
    int A, int act,
    const device float* mask, int mask_offset,
    thread float& out_logsumexp, thread float& out_entropy, thread float& out_logp
) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    float act_logit = 0.0f;
    bool has_valid_action = false;

    for (int a = 0; a < A; a++) {
        float m = mask[mask_offset + a];
        float l = masked_logit(logits[logits_base + (logits_offset + a) * logits_stride_a], m);
        if (a == act) act_logit = l;
        if (m < 0.5f) continue;
        has_valid_action = true;
        if (l > max_logit) {
            sum *= exp(max_logit - l);
            max_logit = l;
        }
        sum += exp(l - max_logit);
    }
    if (!has_valid_action || !isfinite(max_logit) || !isfinite(sum) || sum <= 0.0f) {
        out_logsumexp = NAN;
        out_entropy = NAN;
        out_logp = NAN;
        return;
    }
    float lse = max_logit + log(sum);

    float ent = 0.0f;
    for (int a = 0; a < A; a++) {
        if (mask[mask_offset + a] < 0.5f) continue;
        float l = masked_logit(logits[logits_base + (logits_offset + a) * logits_stride_a],
                               mask[mask_offset + a]);
        float logp = l - lse;
        float p = exp(clamp(logp, -80.0f, 80.0f));
        ent -= p * logp;
    }

    out_logsumexp = lse;
    out_entropy = ent;
    out_logp = act_logit - lse;
}

// PPO helper: compute log_prob and entropy for a single continuous head
inline void ppo_continuous_head(
    float mean, float log_std, float action,
    thread float& out_logp, thread float& out_entropy
) {
    constexpr float HALF_LOG_2PI = 0.9189385332046727f;
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    float std = exp(log_std);
    float normalized = (action - mean) / std;
    out_logp = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
    out_entropy = HALF_1_PLUS_LOG_2PI + log_std;
}

struct PPOFusedParams {
    int num_atns;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    int T_seq;
    int A_total;
    int N;
    int logits_stride_n;
    int logits_stride_t;
    int logits_stride_a;
    int values_stride_n;
    int values_stride_t;
    int is_continuous;
    int num_atns_total;  // sum of act_sizes, for mask buffer indexing
    int mask_stride;     // stride in floats between consecutive mask rows in obs
};

// Fused PPO forward + backward: computes loss partials AND gradients in one pass.
// Per-block partial sums written to ppo_partials, reduced by ppo_loss_reduce_kernel.
kernel void ppo_loss_fwd_bwd_kernel(
    device float* ppo_partials              [[buffer(0)]],
    device float* grad_logits               [[buffer(1)]],
    device float* grad_logstd               [[buffer(2)]],
    device float* grad_values_pred          [[buffer(3)]],
    const device float* logits              [[buffer(4)]],
    const device float* logstd              [[buffer(5)]],
    const device float* values_pred         [[buffer(6)]],
    const device float* actions             [[buffer(7)]],
    const device float* old_logprobs        [[buffer(8)]],
    const device float* advantages          [[buffer(9)]],
    const device float* prio                [[buffer(10)]],
    const device float* values              [[buffer(11)]],
    const device float* returns_buf         [[buffer(12)]],
    const device float* adv_mean            [[buffer(13)]],
    const device float* adv_var             [[buffer(14)]],
    const device int* act_sizes             [[buffer(15)]],
    constant PPOFusedParams& pp             [[buffer(16)]],
    const device float* action_mask         [[buffer(17)]],
    device float* out_ratio                 [[buffer(18)]],
    device float* out_newvalue              [[buffer(19)]],
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint block_id [[threadgroup_position_in_grid]]
) {
    int total_elements = pp.N * pp.T_seq;
    float inv_NT = 1.0f / float(total_elements);

    threadgroup float block_losses[7][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) {
        block_losses[c][tid] = 0.0f;
    }

    // MSL has no goto — use if/else instead of the CUDA goto pattern
    if ((int)idx < total_elements) {
        int n = (int)idx / pp.T_seq;
        int t = (int)idx % pp.T_seq;
        int nt = n * pp.T_seq + t;

        int logits_base = n * pp.logits_stride_n + t * pp.logits_stride_t;
        int values_idx = n * pp.values_stride_n + t * pp.values_stride_t;
        int grad_logits_base = nt * pp.A_total;

        // Shared computation (used by both forward and backward)
        float adv = advantages[nt];
        float w = prio[n];
        float val = values[nt];
        float ret = returns_buf[nt];
        float val_pred = values_pred[values_idx];
        out_newvalue[nt] = val_pred;

        float adv_std = sqrt(adv_var[0]);
        float adv_normalized = (adv - adv_mean[0]) / (adv_std + 1e-8f);

        // grad_loss is always 1.0
        float dL = inv_NT;
        float d_pg_loss = dL;
        float d_entropy_term = dL * (-pp.ent_coef);

        // Value loss (forward) + value gradient (backward)
        float v_error = val_pred - val;
        float v_clipped = val + clamp(v_error, -pp.vf_clip_coef, pp.vf_clip_coef);
        float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
        float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
        float v_loss = 0.5f * fmax(v_loss_unclipped, v_loss_clipped);

        float d_val_pred = 0.0f;
        if (v_loss_clipped > v_loss_unclipped) {
            if (v_error >= -pp.vf_clip_coef && v_error <= pp.vf_clip_coef) {
                d_val_pred = v_clipped - ret;
            }
        } else {
            d_val_pred = val_pred - ret;
        }
        grad_values_pred[nt] = dL * w * pp.vf_coef * d_val_pred;

        // Policy loss + gradients. Both branches produce pg_loss, total_entropy,
        // logratio, ratio — the block-loss accumulation is shared after the if/else.
        float pg_loss, total_entropy, logratio, ratio;

        if (pp.is_continuous) {
            // Continuous: old_logprobs is scalar (matches CUDA)
            float old_logp = old_logprobs[nt];
            float total_log_prob = 0.0f;
            total_entropy = 0.0f;
            for (int h = 0; h < pp.num_atns; h++) {
                float mean = logits[logits_base + h * pp.logits_stride_a];
                float log_std = logstd[h];
                float action = actions[nt * pp.num_atns + h];
                float lp, ent;
                ppo_continuous_head(mean, log_std, action, lp, ent);
                total_log_prob += lp;
                total_entropy += ent;
            }

            logratio = total_log_prob - old_logp;
            ratio = exp(logratio);
            out_ratio[nt] = ratio;
            float ratio_clipped = clamp(ratio, 1.0f - pp.clip_coef, 1.0f + pp.clip_coef);
            float wa = -w * adv_normalized;
            pg_loss = fmax(wa * ratio, wa * ratio_clipped);

            // Backward: policy gradient
            float d_ratio = wa * d_pg_loss;
            if (wa * ratio_clipped > wa * ratio) {
                if (ratio <= (1.0f - pp.clip_coef) || ratio >= (1.0f + pp.clip_coef))
                    d_ratio = 0.0f;
            }
            float d_new_logp = d_ratio * ratio;

            for (int h = 0; h < pp.num_atns; h++) {
                float mean = logits[logits_base + h * pp.logits_stride_a];
                float log_std = logstd[h];
                float std = exp(log_std);
                float var = std * std;
                float action = actions[nt * pp.num_atns + h];
                float diff = action - mean;

                grad_logits[grad_logits_base + h] = d_new_logp * diff / var;
                grad_logstd[grad_logits_base + h] = d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
            }
        } else {
            // Discrete joint-ratio clipping (matches CUDA kernels.cu:738-807).
            // Sum per-head logprobs into scalar, compute single joint ratio, clip once.
            float head_logsumexp[MAX_ATN_HEADS];
            float head_entropy[MAX_ATN_HEADS];
            int head_act[MAX_ATN_HEADS];
            int mask_base = (int)idx * pp.mask_stride;

            int logits_offset = 0;
            float total_log_prob = 0.0f;
            total_entropy = 0.0f;

            for (int h = 0; h < pp.num_atns; h++) {
                int A = act_sizes[h];
                int act = int(actions[nt * pp.num_atns + h]);
                head_act[h] = act;
                float lse, ent, lp;
                ppo_discrete_head(logits, logits_base, pp.logits_stride_a, logits_offset,
                    A, act, action_mask, mask_base + logits_offset, lse, ent, lp);
                head_logsumexp[h] = lse;
                head_entropy[h] = ent;
                total_log_prob += lp;
                total_entropy += ent;
                logits_offset += A;
            }

            float old_logp = old_logprobs[nt];
            logratio = total_log_prob - old_logp;
            ratio = exp(logratio);
            out_ratio[nt] = ratio;
            float ratio_clipped = clamp(ratio, 1.0f - pp.clip_coef, 1.0f + pp.clip_coef);
            float wa = -w * adv_normalized;
            pg_loss = fmax(wa * ratio, wa * ratio_clipped);

            // Backward: single joint d_new_logp shared across all heads
            float d_ratio = wa * d_pg_loss;
            if (wa * ratio_clipped > wa * ratio) {
                if (ratio <= (1.0f - pp.clip_coef) || ratio >= (1.0f + pp.clip_coef))
                    d_ratio = 0.0f;
            }
            float d_new_logp = d_ratio * ratio;

            // Gradient pass over logits (reuses head_logsumexp, head_entropy)
            logits_offset = 0;
            for (int h = 0; h < pp.num_atns; h++) {
                int A = act_sizes[h];
                int act = head_act[h];
                float lse = head_logsumexp[h];
                float ent = head_entropy[h];

                for (int a = 0; a < A; a++) {
                    float raw_l = logits[logits_base + (logits_offset + a) * pp.logits_stride_a];
                    float m = action_mask[mask_base + logits_offset + a];
                    if (m < 0.5f) {
                        grad_logits[grad_logits_base + logits_offset + a] = 0.0f;
                        continue;
                    }
                    float l = masked_logit(raw_l, m);
                    float logp = l - lse;
                    float p = exp(logp);
                    float d_logit = (a == act) ? d_new_logp : 0.0f;
                    d_logit -= p * d_new_logp;
                    d_logit += d_entropy_term * p * (-ent - logp);
                    grad_logits[grad_logits_base + logits_offset + a] = d_logit;
                }
                logits_offset += A;
            }
        }

        // Shared loss accumulation (both branches produce pg_loss, total_entropy, logratio, ratio)
        block_losses[LOSS_PG][tid] = pg_loss * inv_NT;
        block_losses[LOSS_VF][tid] = v_loss * inv_NT;
        block_losses[LOSS_ENT][tid] = total_entropy * inv_NT;
        block_losses[LOSS_TOTAL][tid] = (pg_loss + pp.vf_coef * v_loss - pp.ent_coef * total_entropy) * inv_NT;
        block_losses[LOSS_OLD_APPROX_KL][tid] = (-logratio) * inv_NT;
        block_losses[LOSS_APPROX_KL][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
        block_losses[LOSS_CLIPFRAC][tid] = (abs(ratio - 1.0f) > pp.clip_coef ? 1.0f : 0.0f) * inv_NT;
    } // end if (idx < total_elements)

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Block reduction (tree reduction)
    for (int stride = PPO_THREADS / 2; stride > 0; stride >>= 1) {
        if ((int)tid < stride) {
            for (int c = 0; c < LOSS_N; c++) {
                block_losses[c][tid] += block_losses[c][tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        int base_out = (int)block_id * (LOSS_N + 1);
        ppo_partials[base_out] = block_losses[LOSS_TOTAL][0];
        for (int c = 0; c < LOSS_N; c++) {
            ppo_partials[base_out + 1 + c] = block_losses[c][0];
        }
    }
}

// Deterministic reduction of per-block PPO loss partials + count increment
struct PPOReduceParams {
    int num_blocks;
};

kernel void ppo_loss_reduce_kernel(
    device float* loss                      [[buffer(0)]],
    device float* losses_acc                [[buffer(1)]],
    const device float* partials            [[buffer(2)]],
    constant PPOReduceParams& pp            [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if ((int)tid > LOSS_N) return;

    float sum = 0.0f;
    for (int b = 0; b < pp.num_blocks; b++) {
        sum += partials[b * (LOSS_N + 1) + (int)tid];
    }

    if (tid == 0) {
        *loss += sum;
    } else {
        losses_acc[(int)tid - 1] += sum;
    }

    // Fold add_scalar: increment epoch count
    if (tid == 0) {
        losses_acc[LOSS_N] += 1.0f;
    }
}

struct AdvantageParams {
    float gamma;
    float lambda;
    float rho_clip;
    float c_clip;
    int num_steps;
    int horizon;
};

kernel void puff_advantage_kernel(
    const device float* values      [[buffer(0)]],
    const device float* rewards     [[buffer(1)]],
    const device float* dones       [[buffer(2)]],
    const device float* importance  [[buffer(3)]],
    device float* advantages        [[buffer(4)]],
    constant AdvantageParams& p     [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if ((int)row >= p.num_steps) return;

    int offset = (int)row * p.horizon;
    const device float* v = values + offset;
    const device float* r = rewards + offset;
    const device float* d = dones + offset;
    const device float* imp = importance + offset;
    device float* adv = advantages + offset;

    float lastpufferlam = 0.0f;
    adv[p.horizon - 1] = 0.0f;
    for (int t = p.horizon - 2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0f - d[t_next];
        float rho_t = min(imp[t], p.rho_clip);
        float c_t = min(imp[t], p.c_clip);
        float delta = rho_t * (r[t_next] + p.gamma * v[t_next] * nextnonterminal - v[t]);
        lastpufferlam = delta + p.gamma * p.lambda * c_t * lastpufferlam * nextnonterminal;
        adv[t] = lastpufferlam;
    }
}

struct PrioParams {
    float prio_alpha;
    int stride;
};

kernel void prio_adv_reduction_kernel(
    const device float* advantages  [[buffer(0)]],
    device float* prio_weights      [[buffer(1)]],
    constant PrioParams& pp         [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tx [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    int offset = (int)row * pp.stride;

    float local_sum = 0.0f;
    for (int t = (int)tx; t < pp.stride; t += 32) {
        local_sum += abs(advantages[offset + t]);
    }

    // Simdgroup reduction
    local_sum = simd_sum(local_sum);

    if (simd_lane == 0 && tx < 32) {
        // epsilon floor prevents pow(0, alpha>0) = 0 which would permanently
        // exclude zero-advantage segments from sampling in sparse-reward envs
        float pw = pow(local_sum + 1e-6f, pp.prio_alpha);
        if (isnan(pw) || isinf(pw)) pw = 1e-6f;
        prio_weights[row] = pw;
    }
}

struct PrioNormParams {
    int length;
};

kernel void prio_normalize_kernel(
    device float* prio_weights          [[buffer(0)]],
    constant PrioNormParams& pp         [[buffer(1)]],
    uint tx [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr float eps = 1e-6f;
    constexpr int NUM_WARPS = 8;  // 256 / 32

    threadgroup float shmem[NUM_WARPS];
    threadgroup float block_sum;

    float local_sum = 0.0f;
    for (int t = (int)tx; t < pp.length; t += 256) {
        local_sum += prio_weights[t];
    }

    // Simdgroup reduction
    local_sum = simd_sum(local_sum);

    if (simd_lane == 0) shmem[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float val = (simd_lane < NUM_WARPS) ? shmem[simd_lane] : 0.0f;
        val = simd_sum(val);
        // add eps * length so numerator and denominator are consistent:
        // sum((w_i + eps)) = sum(w_i) + eps*length
        if (tx == 0) block_sum = val + eps * float(pp.length);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = (int)tx; t < pp.length; t += 256) {
        prio_weights[t] = (prio_weights[t] + eps) / block_sum;
    }
}

struct PrioImpParams {
    int total_agents;
    float anneal_beta;
    int minibatch_segments;
};

struct PrioSampleParams {
    uint64_t seed;
    uint base_offset;
    int total_segments;
    int minibatch_segments;
};

kernel void prio_sample_kernel(
    device int64_t* indices            [[buffer(0)]],
    const device float* prio_probs     [[buffer(1)]],
    constant PrioSampleParams& pp      [[buffer(2)]],
    uint tx [[thread_position_in_grid]]
) {
    if ((int)tx >= pp.minibatch_segments) return;

    uint4 counter = uint4(pp.base_offset + tx, 0u, 0u, 0u);
    uint2 key = uint2((uint)(pp.seed & 0xFFFFFFFF), (uint)(pp.seed >> 32));
    uint4 rng_out = philox4x32_10(counter, key);
    uint rng_idx = 0;
    float u = philox_uniform(rng_idx, rng_out);

    float cumsum = 0.0f;
    int sampled = pp.total_segments - 1;
    for (int i = 0; i < pp.total_segments; i++) {
        cumsum += prio_probs[i];
        if (u <= cumsum) {
            sampled = i;
            break;
        }
    }
    indices[tx] = sampled;
}

kernel void prio_imp_weights_kernel(
    const device int64_t* indices       [[buffer(0)]],
    const device float* prio_probs      [[buffer(1)]],
    device float* mb_prio               [[buffer(2)]],
    constant PrioImpParams& pp          [[buffer(3)]],
    uint tx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Each thread covers a strided subset so the reduction includes the
    // entire minibatch even when it has more than 256 segments.
    float w = 0.0f;
    if ((int)tx < pp.minibatch_segments) {
        float value = prio_probs[indices[tx]] * float(pp.total_agents);
        w = pow(value, -pp.anneal_beta);
    }

    constexpr int NUM_WARPS = 8;
    threadgroup float shmem[NUM_WARPS];
    float local_max = w;
    for (int i = int(tx) + 256; i < pp.minibatch_segments; i += 256) {
        float value = prio_probs[indices[i]] * float(pp.total_agents);
        local_max = max(local_max, pow(value, -pp.anneal_beta));
    }
    local_max = simd_max(local_max);
    if (simd_lane == 0) shmem[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float val = (simd_lane < NUM_WARPS) ? shmem[simd_lane] : 0.0f;
        val = simd_max(val);
        if (simd_lane == 0) shmem[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w_max = max(shmem[0], 1e-8f);

    if ((int)tx < pp.minibatch_segments) {
        mb_prio[tx] = w / w_max;
    }
    for (int i = int(tx) + 256; i < pp.minibatch_segments; i += 256) {
        float value = prio_probs[indices[i]] * float(pp.total_agents);
        mb_prio[i] = pow(value, -pp.anneal_beta) / w_max;
    }
}

struct FillParams {
    float val;
    int n;
};

kernel void fill_f32(
    device float* dst           [[buffer(0)]],
    constant FillParams& p      [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] = p.val;
}

kernel void copy_f32(
    device float* dst           [[buffer(0)]],
    device const float* src     [[buffer(1)]],
    constant int& n             [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < n) dst[idx] = src[idx];
}

kernel void fill_f16_zero(
    device ushort* dst [[buffer(0)]], constant int& n [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx < uint(n)) dst[idx] = 0;
}

kernel void copy_f16(
    device ushort* dst [[buffer(0)]], const device ushort* src [[buffer(1)]],
    constant int& n [[buffer(2)]], uint idx [[thread_position_in_grid]]) {
    if (idx < uint(n)) dst[idx] = src[idx];
}

struct ClampParams {
    float lo;
    float hi;
    int n;
};

kernel void clamp_f32(
    device float* dst           [[buffer(0)]],
    constant ClampParams& p     [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] = clamp(dst[idx], p.lo, p.hi);
}

struct ScaleParams {
    float alpha;
    int n;
};

kernel void scale_f32(
    device float* dst           [[buffer(0)]],
    constant ScaleParams& p     [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] *= p.alpha;
}

struct AxpyParams {
    float alpha;
    int n;
};

kernel void axpy_f32(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant AxpyParams& p          [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] += p.alpha * src[idx];
}

struct AxpbyParams {
    float alpha;
    float beta;
    int n;
};

kernel void axpby_f32(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant AxpbyParams& p         [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] = p.alpha * src[idx] + p.beta * dst[idx];
}

struct AddParams {
    int n;
};

kernel void add_f32(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant AddParams& p           [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] += src[idx];
}

kernel void add_f16(
    device half* dst                [[buffer(0)]],
    const device half* src          [[buffer(1)]],
    constant AddParams& p           [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] = half(float(dst[idx]) + float(src[idx]));
}

struct NesterovParams {
    float mu;
    int n;
};

// Fused Nesterov momentum: mb = mu*mb + gc; gc = gc + mu*mb (note: gc uses updated mb)
kernel void nesterov_f32(
    device float* mb                [[buffer(0)]],
    device float* gc                [[buffer(1)]],
    constant NesterovParams& p      [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) {
        float m = p.mu * mb[idx] + gc[idx];
        mb[idx] = m;
        gc[idx] += p.mu * m;
    }
}

struct ScaleDevParams {
    int n;
};

// Scale by device-side scalar: dst[i] *= *alpha_ptr
kernel void scale_f32_dev(
    device float* dst                   [[buffer(0)]],
    const device float* alpha_ptr       [[buffer(1)]],
    constant ScaleDevParams& p          [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    float alpha = *alpha_ptr;
    if ((int)idx < p.n) dst[idx] *= alpha;
}

struct AxpyDevParams {
    int n;
};

// dst += (*alpha) * src
kernel void axpy_f32_dev(
    device float* dst                   [[buffer(0)]],
    const device float* src             [[buffer(1)]],
    const device float* alpha_ptr       [[buffer(2)]],
    constant AxpyDevParams& p           [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    float alpha = *alpha_ptr;
    if ((int)idx < p.n) dst[idx] += alpha * src[idx];
}

struct AddScalarParams {
    float val;
};

// *ptr += val (single element)
kernel void add_scalar(
    device float* ptr               [[buffer(0)]],
    constant AddScalarParams& p     [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx == 0) *ptr += p.val;
}

// Reads LR from device, computes neg_lr = -lr
kernel void compute_lr_scalars_kernel(
    const device float* lr          [[buffer(0)]],
    device float* neg_lr            [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx == 0) {
        *neg_lr = -(*lr);
    }
}

struct MuonParams {
    int n;
    float scale;
};

// Weight update: wb = wb * (1 - lr * wd) - lr * scale * up  (matches CUDA muon.cu)
kernel void muon_weight_update_kernel(
    device float* wb                    [[buffer(0)]],
    const device float* up              [[buffer(1)]],
    const device float* lr_ptr          [[buffer(2)]],
    constant MuonParams& p              [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.n) return;
    float lr = *lr_ptr;
    wb[idx] = wb[idx] - lr * p.scale * up[idx];
}

struct TransposeParams {
    int R;
    int C;
};

// Transpose R x C matrix: dst[c*R + r] = src[r*C + c]
kernel void transpose_f32(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant TransposeParams& p     [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.R * p.C) return;
    dst[(idx % p.C) * p.R + idx / p.C] = src[idx];
}

struct Transpose01Params {
    int A;
    int B;
    int C;
};

// Transpose dims 0 and 1 of (A, B, C) tensor: dst[b*A*C + a*C + c] = src[a*B*C + b*C + c]
kernel void transpose_01(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant Transpose01Params& p   [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    int total = p.A * p.B * p.C;
    if ((int)idx >= total) return;
    int a = (int)idx / (p.B * p.C);
    int rem = (int)idx % (p.B * p.C);
    int b = rem / p.C;
    int c = rem % p.C;
    dst[b * p.A * p.C + a * p.C + c] = src[idx];
}

kernel void transpose_01_u16(
    device ushort* dst [[buffer(0)]], const device ushort* src [[buffer(1)]],
    constant Transpose01Params& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]]) {
    int total = p.A * p.B * p.C;
    if (idx >= uint(total)) return;
    int a = int(idx) / (p.B * p.C), rem = int(idx) % (p.B * p.C);
    int b = rem / p.C, c = rem % p.C;
    dst[b * p.A * p.C + a * p.C + c] = src[idx];
}

kernel void transpose_01_u8(
    device uchar* dst [[buffer(0)]], const device uchar* src [[buffer(1)]],
    constant Transpose01Params& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]]) {
    int total = p.A * p.B * p.C;
    if (idx >= uint(total)) return;
    int a = int(idx) / (p.B * p.C), rem = int(idx) % (p.B * p.C);
    int b = rem / p.C, c = rem % p.C;
    dst[b * p.A * p.C + a * p.C + c] = src[idx];
}

// Transpose dims 0 and 1 for 8-byte elements (f64/u64), using uint2 pairs.
// Metal has no native double support, so we treat each 8-byte element as uint2.
// Same index math as transpose_01 — just operates on uint2 instead of float.
kernel void transpose_01_u64(
    device uint2* dst                   [[buffer(0)]],
    const device uint2* src             [[buffer(1)]],
    constant Transpose01Params& p       [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    int total = p.A * p.B * p.C;
    if ((int)idx >= total) return;
    int a = (int)idx / (p.B * p.C);
    int rem = (int)idx % (p.B * p.C);
    int b = rem / p.C;
    int c = rem % p.C;
    dst[b * p.A * p.C + a * p.C + c] = src[idx];
}

struct NormParams {
    int n;
};

// Per-block sum of squares (partial reduction)
kernel void norm_f32_kernel(
    device float* partials              [[buffer(0)]],
    const device float* src             [[buffer(1)]],
    constant NormParams& p              [[buffer(2)]],
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint block_id [[threadgroup_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NUM_WARPS = 8;  // 256 / 32
    threadgroup float sdata[NUM_WARPS];
    float sum = 0.0f;
    for (int i = (int)idx; i < p.n; i += (int)grid_size) {
        sum += src[i] * src[i];
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) sdata[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        sum = (simd_lane < NUM_WARPS) ? sdata[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) partials[block_id] = sum;
    }
}

struct NormReduceParams {
    int num_blocks;
};

// Reduce per-block partials to a single sum-of-squares value
kernel void norm_reduce_kernel(
    device float* out                   [[buffer(0)]],
    const device float* partials        [[buffer(1)]],
    constant NormReduceParams& p        [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NUM_WARPS = 8;
    threadgroup float sdata[NUM_WARPS];
    float val = ((int)tid < p.num_blocks) ? partials[tid] : 0.0f;
    val = simd_sum(val);
    if (simd_lane == 0) sdata[simd_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        val = (simd_lane < NUM_WARPS) ? sdata[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) *out = val;
    }
}

struct ClipByNormParams {
    float max_norm;
    float eps;
    int n;
};

// Clip gradient by global norm, with the same arithmetic as CUDA Muon.
kernel void clip_by_norm_f32(
    device float* dst                       [[buffer(0)]],
    const device float* sum_sq_ptr          [[buffer(1)]],
    constant ClipByNormParams& p            [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    float clip_coef = min(p.max_norm / (sqrt(*sum_sq_ptr) + p.eps), 1.0f);
    if ((int)idx < p.n) {
        dst[idx] *= clip_coef;
    }
}

struct NormalizeParams {
    float eps;
    int n;
};

// dst[i] /= max(sqrt(*norm), eps) — matches CUDA (no cap)
kernel void normalize_f32(
    device float* dst                       [[buffer(0)]],
    const device float* norm_ptr            [[buffer(1)]],
    constant NormalizeParams& p             [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    float inv_norm = 1.0f / max(sqrt(*norm_ptr), p.eps);
    if ((int)idx < p.n) dst[idx] = dst[idx] * inv_norm;
}

struct VarMeanParams {
    int n;
};

// Compute variance and mean of a float array (single threadgroup)
kernel void var_mean_kernel(
    const device float* src         [[buffer(0)]],
    device float* var_out           [[buffer(1)]],
    device float* mean_out          [[buffer(2)]],
    constant VarMeanParams& p       [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NUM_WARPS = 8;
    threadgroup float sdata[NUM_WARPS];

    // Pass 1: compute mean
    float sum = 0.0f;
    for (int i = (int)tid; i < p.n; i += 256) sum += src[i];
    sum = simd_sum(sum);
    if (simd_lane == 0) sdata[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        sum = (simd_lane < NUM_WARPS) ? sdata[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) sdata[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = sdata[0] / float(p.n);
    if (tid == 0) *mean_out = mean;

    // Pass 2: compute variance
    float ss = 0.0f;
    for (int i = (int)tid; i < p.n; i += 256) {
        float d = src[i] - mean;
        ss += d * d;
    }
    ss = simd_sum(ss);
    if (simd_lane == 0) sdata[simd_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        ss = (simd_lane < NUM_WARPS) ? sdata[simd_lane] : 0.0f;
        ss = simd_sum(ss);
        if (simd_lane == 0) sdata[0] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) *var_out = sdata[0] / float(p.n - 1);
}

struct SumRowsParams {
    int R;
    int C;
};

// dst[c] = sum over rows of src[:, c]
kernel void sum_rows_to_f32_kernel(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant SumRowsParams& p       [[buffer(2)]],
    uint col [[thread_position_in_grid]]
) {
    if ((int)col >= p.C) return;
    float sum = 0.0f;
    for (int r = 0; r < p.R; r++) sum += src[r * p.C + (int)col];
    dst[col] = sum;
}

// --- Sum rows fp16 (for bias/LN param grads) ---

kernel void sum_rows_f16_kernel(
    device half* dst                [[buffer(0)]],
    const device half* src          [[buffer(1)]],
    constant SumRowsParams& p       [[buffer(2)]],
    uint col [[thread_position_in_grid]]
) {
    if ((int)col >= p.C) return;
    float sum = 0.0f;
    for (int r = 0; r < p.R; r++) sum += float(src[r * p.C + (int)col]);
    dst[col] = half(sum);
}

struct AssembleDecoderGradParams {
    int B_TT;
    int od;
    int od_plus_1;
};

// Assemble gradient: dst = [grad_logits | grad_value] in fused layout
kernel void assemble_decoder_grad_f32(
    device float* dst                       [[buffer(0)]],
    const device float* grad_logits         [[buffer(1)]],
    const device float* grad_value          [[buffer(2)]],
    constant AssembleDecoderGradParams& p   [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.B_TT * p.od_plus_1) return;
    int row = (int)idx / p.od_plus_1;
    int col = (int)idx % p.od_plus_1;
    dst[idx] = (col < p.od) ? grad_logits[row * p.od + col] : grad_value[row];
}

struct SelectCopyParams {
    int obs_row_bytes;
    int act_row_bytes;
    int lp_row_bytes;
    int horizon;
    int train_fp16;  // 1: encoder reads fp16_obs_out (skip mb_obs f32 write); 0: encoder reads mb_obs (skip f16 write)
};

// Minibatch assembly: copy observations, actions, logprobs, values+advantages+returns, prio
// Channel 0 fuses obs gather + f32→f16 cast. Only the variant the encoder will read gets written.
// Dispatched as (minibatch_size, 5) threadgroups, each handles one channel for one row.
kernel void select_copy_kernel(
    device char* mb_obs                 [[buffer(0)]],
    device char* mb_actions             [[buffer(1)]],
    device char* mb_logprobs            [[buffer(2)]],
    device float* mb_values             [[buffer(3)]],
    device float* mb_advantages         [[buffer(4)]],
    device float* mb_returns            [[buffer(5)]],
    device float* mb_prio_out           [[buffer(6)]],
    const device char* src_obs          [[buffer(7)]],
    const device char* src_actions      [[buffer(8)]],
    const device char* src_logprobs     [[buffer(9)]],
    const device float* src_values      [[buffer(10)]],
    const device float* advantages      [[buffer(11)]],
    const device int64_t* idx           [[buffer(12)]],
    const device float* mb_prio_in      [[buffer(13)]],
    constant SelectCopyParams& p        [[buffer(14)]],
    device half* fp16_obs_out           [[buffer(15)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    int mb = (int)group_id.x;
    int ch = (int)group_id.y;
    int src_row = (int)idx[mb];

    if (ch == 0) {
        // Fused obs gather + (optional) f32→f16 cast. Encoder reads exactly one variant
        // depending on train_fp16; the other is dead-on-arrival, so skip writing it.
        const device float* sptr = (const device float*)(src_obs + (int64_t)src_row * p.obs_row_bytes);
        int count = p.obs_row_bytes / 4;
        if (p.train_fp16) {
            device half* f16ptr = fp16_obs_out + (int64_t)mb * count;
            for (int i = (int)tid; i < count; i += 256) {
                f16ptr[i] = half(sptr[i]);
            }
        } else {
            device float* f32ptr = (device float*)(mb_obs + (int64_t)mb * p.obs_row_bytes);
            for (int i = (int)tid; i < count; i += 256) {
                f32ptr[i] = sptr[i];
            }
        }
    } else if (ch == 1) {
        // Copy actions
        const device int* sptr = (const device int*)(src_actions + (int64_t)src_row * p.act_row_bytes);
        device int* dptr = (device int*)(mb_actions + (int64_t)mb * p.act_row_bytes);
        for (int i = (int)tid; i < p.act_row_bytes / 4; i += 256) dptr[i] = sptr[i];
    } else if (ch == 2) {
        // Copy logprobs
        const device int* sptr = (const device int*)(src_logprobs + (int64_t)src_row * p.lp_row_bytes);
        device int* dptr = (device int*)(mb_logprobs + (int64_t)mb * p.lp_row_bytes);
        for (int i = (int)tid; i < p.lp_row_bytes / 4; i += 256) dptr[i] = sptr[i];
    } else if (ch == 3) {
        // Copy values + advantages, compute returns = values + advantages
        int srh = src_row * p.horizon;
        int drh = mb * p.horizon;
        for (int i = (int)tid; i < p.horizon; i += 256) {
            float val = src_values[srh + i];
            float adv = advantages[srh + i];
            mb_values[drh + i] = val;
            mb_advantages[drh + i] = adv;
            mb_returns[drh + i] = val + adv;
        }
    } else if (ch == 4) {
        // Copy prio weight
        if (tid == 0) {
            mb_prio_out[mb] = mb_prio_in[mb];
        }
    }
}

struct IndexCopyParams {
    int num_idx;
    int row_bytes;
};

// Indexed copy: for each i, copy src row i to dst row idx[i]
kernel void index_copy_kernel(
    device char* dst                    [[buffer(0)]],
    const device int64_t* idx           [[buffer(1)]],
    const device char* src              [[buffer(2)]],
    constant IndexCopyParams& p         [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if ((int)i >= p.num_idx) return;
    int64_t dst_row = idx[i];
    const device char* s = src + (int64_t)i * p.row_bytes;
    device char* d = dst + dst_row * p.row_bytes;
    // Copy as 4-byte words, then handle remainder bytes
    int words = p.row_bytes / 4;
    const device uint* s4 = (const device uint*)s;
    device uint* d4 = (device uint*)d;
    for (int b = 0; b < words; b++) d4[b] = s4[b];
    for (int b = words * 4; b < p.row_bytes; b++) d[b] = s[b];
}

// index_gather_kernel: dst[i] = src[idx[i]] (gather, inverse of index_copy)
kernel void index_gather_kernel(
    device char* dst                    [[buffer(0)]],
    const device int64_t* idx           [[buffer(1)]],
    const device char* src              [[buffer(2)]],
    constant IndexCopyParams& p         [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if ((int)i >= p.num_idx) return;
    int64_t src_row = idx[i];
    const device char* s = src + src_row * p.row_bytes;
    device char* d = dst + (int64_t)i * p.row_bytes;
    int words = p.row_bytes / 4;
    const device uint* s4 = (const device uint*)s;
    device uint* d4 = (device uint*)d;
    for (int b = 0; b < words; b++) d4[b] = s4[b];
    for (int b = words * 4; b < p.row_bytes; b++) d[b] = s[b];
}

struct CastU8Params {
    int n;
};

kernel void cast_u8_to_f32(
    device float* dst                       [[buffer(0)]],
    const device uchar* src                 [[buffer(1)]],
    constant CastU8Params& p                [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] = float(src[idx]);
}

// IEEE 754 f64→f32 bit cast. Metal has no native double, so we read each
// 8-byte double as uint2 and extract
// sign, exponent, mantissa via bit manipulation. Subnormals flush to zero.
kernel void cast_f64_to_f32(
    device const uint2* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant int& count     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= count) return;
    uint2 bits = src[gid];
    uint hi = bits.y, lo = bits.x;
    uint sign = hi >> 31;
    int biased_exp = int((hi >> 20) & 0x7FFu);
    int exp_f32 = biased_exp - 1023 + 127;
    // Top 23 bits of the 52-bit mantissa: 20 from hi + 3 from lo
    uint mantissa = ((hi & 0xFFFFFu) << 3) | (lo >> 29);
    uint result;
    if (biased_exp == 0)          result = sign << 31;          // zero / subnormal → ±0
    else if (biased_exp == 0x7FF) result = (sign << 31) | 0x7F800000u; // inf/nan → ±inf
    else if (exp_f32 <= 0)        result = sign << 31;          // underflow → ±0
    else if (exp_f32 >= 255)      result = (sign << 31) | 0x7F800000u; // overflow → ±inf
    else                          result = (sign << 31) | (uint(exp_f32) << 23) | mantissa;
    dst[gid] = as_type<float>(result);
}

// ============================================================================
// Section 20: Tiled GEMM — C = alpha * op(A) @ op(B) + beta * C
//
// 64x64 tiled simdgroup_matrix GEMM for f32. Supports NT, NN, TN layouts
// via trans_a/trans_b parameters. Runs on the compute encoder.
// ============================================================================

struct GemmParams {
    int M;       // result rows
    int N;       // result columns
    int K;       // inner dimension
    int lda;     // leading dimension of A (physical columns)
    int ldb;     // leading dimension of B (physical columns)
    int ldc;     // leading dimension of C (= N)
    float alpha;
    float beta;
    int trans_a; // 0 = no transpose, 1 = transpose
    int trans_b;
};

constant int BM = 32;   // tile rows (shared by ksplit and fp16 register GEMMs)
constant int BN = 32;   // tile cols
constant int BK = 16;   // tile K dimension
constant int TM = 4;    // per-thread tile rows
constant int TN = 4;    // per-thread tile cols

// ============================================================================
// Section 22: K-split GEMM — for tall-K backward weight-gradient GEMMs
//
// When M and N are small but K is large (e.g. M=128, N=128, K=32768),
// regular sgemm_reg has too few threadgroups (M/32 * N/32 = 16). K-split
// partitions K across multiple TGs in the Z axis, writing partial sums to
// a scratch buffer. A second kernel reduces the partials into C.
// ============================================================================

kernel void sgemm_ksplit(
    device const float* A          [[buffer(0)]],
    device const float* B          [[buffer(1)]],
    device float* partials         [[buffer(2)]],
    constant GemmParams& p         [[buffer(3)]],
    constant int& k_per_split      [[buffer(4)]],
    uint3 group_id    [[threadgroup_position_in_grid]],
    uint3 local_id    [[thread_position_in_threadgroup]]
) {
    threadgroup float sA[BM][BK];
    threadgroup float sB[BK][BN];

    int trow = (int)local_id.y;
    int tcol = (int)local_id.x;
    int tid = trow * (BN / TN) + tcol;

    float acc[TM][TN];
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            acc[m][n] = 0.0f;

    int k_start = (int)group_id.z * k_per_split;
    int k_end = min(k_start + k_per_split, p.K);
    int num_k_tiles = (k_end - k_start + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_base = k_start + kt * BK;

        for (int i = 0; i < (BM * BK) / 64; i++) {
            int idx = tid + i * 64;
            int r = idx / BK;
            int c = idx % BK;
            int gr = (int)group_id.y * BM + r;
            int gc = k_base + c;
            sA[r][c] = (gr < p.M && gc < k_end)
                ? (p.trans_a ? A[gc * p.lda + gr] : A[gr * p.lda + gc])
                : 0.0f;
        }

        for (int i = 0; i < (BK * BN) / 64; i++) {
            int idx = tid + i * 64;
            int r = idx / BN;
            int c = idx % BN;
            int gr = k_base + r;
            int gc = (int)group_id.x * BN + c;
            sB[r][c] = (gr < k_end && gc < p.N)
                ? (p.trans_b ? B[gc * p.ldb + gr] : B[gr * p.ldb + gc])
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < BK; k++) {
            float a_reg[TM];
            float b_reg[TN];
            for (int m = 0; m < TM; m++) a_reg[m] = sA[trow * TM + m][k];
            for (int n = 0; n < TN; n++) b_reg[n] = sB[k][tcol * TN + n];
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    acc[m][n] += a_reg[m] * b_reg[n];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial results: partials[split_idx * M * N + row * N + col]
    int split_offset = (int)group_id.z * p.M * p.N;
    int row_base = (int)group_id.y * BM + trow * TM;
    int col_base = (int)group_id.x * BN + tcol * TN;
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int r = row_base + m;
            int c = col_base + n;
            if (r < p.M && c < p.N) {
                partials[split_offset + r * p.N + c] = acc[m][n];
            }
        }
    }
}

struct ReduceKsplitParams {
    int MN;
    int num_splits;
    float alpha;
    float beta;
};

kernel void reduce_ksplit(
    device const float* partials   [[buffer(0)]],
    device float* C                [[buffer(1)]],
    constant ReduceKsplitParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.MN) return;
    float sum = 0.0f;
    for (int s = 0; s < p.num_splits; s++) {
        sum += partials[s * p.MN + (int)gid];
    }
    C[gid] = p.alpha * sum + p.beta * C[gid];
}

// Reads fp32 grad_logits and grad_value (from PPO kernel), writes fp16 grad_out.
// grad_out[row * od1 + col] = (col < od) ? grad_logits[row * od + col] : grad_value[row]
kernel void assemble_decoder_grad_f32_to_f16(
    device half* grad_out                   [[buffer(0)]],
    const device float* grad_logits         [[buffer(1)]],
    const device float* grad_value          [[buffer(2)]],
    constant AssembleDecoderGradParams& p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.B_TT * p.od_plus_1) return;
    int row = (int)gid / p.od_plus_1;
    int col = (int)gid % p.od_plus_1;
    float val = (col < p.od) ? grad_logits[row * p.od + col] : grad_value[row];
    // Clamp to fp16 range to prevent inf (Metal fp16 max ~65504, unlike CUDA bf16)
    grad_out[gid] = half(clamp(val, -65000.0f, 65000.0f));
}

kernel void cast_f32_to_f16(
    device half* dst            [[buffer(0)]],
    const device float* src     [[buffer(1)]],
    constant int& count         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= count) return;
    dst[gid] = half(src[gid]);
}

kernel void cast_f16_to_f32(
    device float* dst           [[buffer(0)]],
    const device half* src      [[buffer(1)]],
    constant int& count         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= count) return;
    dst[gid] = float(src[gid]);
}

kernel void mingru_scan_forward_checkpointed_fp16(
    device half* out                [[buffer(0)]],
    device half* next_state         [[buffer(1)]],
    device float* a_star_buf        [[buffer(2)]],
    device float* s_buf             [[buffer(3)]],
    device float* log_values_buf    [[buffer(4)]],
    const device half* combined     [[buffer(5)]],
    const device half* state        [[buffer(6)]],
    const device half* input        [[buffer(7)]],
    constant ScanParams& p          [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.B * p.H) return;

    int b = (int)idx / p.H;
    int h = (int)idx % p.H;
    int bH = b * p.H;
    int H3 = 3 * p.H;
    int bHT = bH * p.T_seq;
    int out_base = bHT + h;
    int cbase = 3 * bHT;

    float a_star = 0.0f;
    float log_value = 0.0f;
    float s = log(float(state[bH + h]));
    log_value = s;

    int T_out = p.T_seq + 1;
    int buf_base = b * T_out * p.H + h;
    int buf_curr = buf_base;
    a_star_buf[buf_curr] = a_star;
    s_buf[buf_curr] = s;
    log_values_buf[buf_curr] = log_value;

    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 1; t < p.T_seq + 1; t++) {
        float hidden_val = float(combined[cbase + h + t_offset]);
        float gate_val = float(combined[cbase + p.H + h + t_offset]);
        float proj_val = float(combined[cbase + 2 * p.H + h + t_offset]);
        float x_val = float(input[out_base + (t - 1) * p.H]);

        float log_coeff_val;
        log_coeffs_and_values_fwd(gate_val, hidden_val, log_coeff_val, log_value);

        a_star += log_coeff_val;

        float z = log_value - a_star;
        float max_val = fmax(s, z);
        s = max_val + log1p_f(exp(-abs(s - z)));

        float scan_result = exp(a_star + s);
        float proj_sigmoid = sigmoid_f(proj_val);
        float out_val = proj_sigmoid * scan_result + (1.0f - proj_sigmoid) * x_val;

        out[out_curr] = half(clamp(out_val, -65000.0f, 65000.0f));

        buf_curr += p.H;
        out_curr += p.H;
        t_offset += H3;

        if (t % CHECKPOINT_INTERVAL == 0) {
            a_star_buf[buf_curr] = a_star;
            s_buf[buf_curr] = s;
            log_values_buf[buf_curr] = log_value;
        }
    }

    float next_state_val = max(exp(a_star + s), 1e-30f);
    next_state[bH + h] = half(min(next_state_val, 65000.0f));
}

kernel void mingru_scan_backward_checkpointed_fp16(
    device half* grad_combined            [[buffer(0)]],
    device half* grad_state               [[buffer(1)]],
    device half* grad_input               [[buffer(2)]],
    const device half* grad_out           [[buffer(3)]],
    const device half* grad_next_state    [[buffer(4)]],
    const device half* combined           [[buffer(5)]],
    const device half* state              [[buffer(6)]],
    const device half* input              [[buffer(7)]],
    const device float* a_star_buf        [[buffer(8)]],
    const device float* s_buf             [[buffer(9)]],
    const device float* log_values_buf    [[buffer(10)]],
    constant ScanParams& p                [[buffer(11)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.B * p.H) return;

    int b = (int)idx / p.H;
    int h = (int)idx % p.H;
    int bHT = b * p.H * p.T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * p.H;
    int state_idx = b * p.H + h;
    int out_base = bHT + h;

    int T_out = p.T_seq + 1;
    int buf_base = b * T_out * p.H + h;

    float acc = 0.0f;
    float s_val_next = 0.0f;
    float carry_grad_a = 0.0f;

    for (int chunk_end = p.T_seq; chunk_end > 0; chunk_end -= CHECKPOINT_INTERVAL) {
        int chunk_start = (chunk_end > CHECKPOINT_INTERVAL) ? (chunk_end - CHECKPOINT_INTERVAL) : 0;
        int chunk_len = chunk_end - chunk_start;

        float chunk_a_star[CHECKPOINT_INTERVAL];
        float chunk_s[CHECKPOINT_INTERVAL];
        float chunk_log_values[CHECKPOINT_INTERVAL];
        float chunk_hidden[CHECKPOINT_INTERVAL];
        float chunk_gate[CHECKPOINT_INTERVAL];

        int ckpt_buf_idx = buf_base + chunk_start * p.H;
        float recomp_a_star = a_star_buf[ckpt_buf_idx];
        float recomp_s = s_buf[ckpt_buf_idx];
        float recomp_log_value = log_values_buf[ckpt_buf_idx];

        for (int i = 0; i < chunk_len; i++) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;
            float hv = float(combined[cbase + h + t_offset]);
            float gv = float(combined[cbase + p.H + h + t_offset]);

            float lc;
            log_coeffs_and_values_fwd(gv, hv, lc, recomp_log_value);
            recomp_a_star += lc;

            float z = recomp_log_value - recomp_a_star;
            float mv = fmax(recomp_s, z);
            recomp_s = mv + log1p_f(exp(-abs(recomp_s - z)));

            chunk_a_star[i] = recomp_a_star;
            chunk_s[i] = recomp_s;
            chunk_log_values[i] = recomp_log_value;
            chunk_hidden[i] = hv;
            chunk_gate[i] = gv;
        }

        for (int i = chunk_len - 1; i >= 0; i--) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;

            float a_star_t = chunk_a_star[i];
            float s_t = chunk_s[i];
            float log_value_t = chunk_log_values[i];
            float hidden_val = chunk_hidden[i];
            float gate_val = chunk_gate[i];
            float proj_val = float(combined[cbase + 2 * p.H + h + t_offset]);
            int input_idx = out_base + (t - 1) * p.H;
            float x_val = float(input[input_idx]);

            float scan_result = exp(a_star_t + s_t);
            float z = log_value_t - a_star_t;

            float grad_out_val = float(grad_out[input_idx]);
            float grad_scan_from_next = (t == p.T_seq) ? float(grad_next_state[state_idx]) : 0.0f;

            float proj_sigmoid = sigmoid_f(proj_val);
            float grad_scan_result = grad_scan_from_next + grad_out_val * proj_sigmoid;
            float grad_proj = grad_out_val * (scan_result - x_val) * proj_sigmoid * (1.0f - proj_sigmoid);
            float grad_input_val = grad_out_val * (1.0f - proj_sigmoid);
            grad_input[input_idx] = half(clamp(grad_input_val, -65000.0f, 65000.0f));

            float grad_log_h = grad_scan_result * scan_result;
            float grad_s = grad_log_h;

            if (t == p.T_seq) {
                acc = grad_s;
            } else {
                acc = grad_s + acc * exp(s_t - s_val_next);
            }
            float grad_z = acc * exp(z - s_t);
            s_val_next = s_t;

            float grad_a = grad_log_h + carry_grad_a - grad_z;
            carry_grad_a = grad_a;

            float grad_g, grad_h;
            log_coeffs_and_values_bwd(grad_a, grad_z, gate_val, hidden_val, grad_g, grad_h);

            // Clamp to fp16 range to prevent inf (Metal fp16 max ~65504)
            grad_combined[cbase + h + t_offset] = half(clamp(grad_h, -65000.0f, 65000.0f));
            grad_combined[cbase + p.H + h + t_offset] = half(clamp(grad_g, -65000.0f, 65000.0f));
            grad_combined[cbase + 2 * p.H + h + t_offset] = half(clamp(grad_proj, -65000.0f, 65000.0f));
        }
    }

    int ckpt_0_idx = buf_base;
    float a_star_0 = a_star_buf[ckpt_0_idx];
    float s_0 = s_buf[ckpt_0_idx];
    float log_value_0 = log_values_buf[ckpt_0_idx];

    acc = acc * exp(s_0 - s_val_next);
    float grad_z_0 = acc * exp((log_value_0 - a_star_0) - s_0);

    float state_val = float(state[state_idx]);
    float grad_state_val = (state_val > 0.0f) ? (grad_z_0 / state_val) : 0.0f;
    grad_state[state_idx] = half(clamp(grad_state_val, -65000.0f, 65000.0f));
}

// ============================================================================
// Section 26: Steel GEMM — C = alpha * op(A) @ op(B) + beta * C
//
// MLX-inspired 64x64 tiled GEMM using simdgroup_matrix (Apple Silicon M3+).
// 4 simdgroups in 2x2 layout (128 threads), each computing 32x32 output
// via a 4x4 grid of 8x8 simdgroup_matrix multiply-accumulate operations.
//
// Hot loop uses direct device memory loads (ThunderMittens-validated:
// Apple Silicon's L2 cache provides effective data reuse without explicit
// threadgroup staging, eliminating barrier overhead). K-remainder and edge
// tile stores use threadgroup fallback.
//
// ============================================================================

kernel void steel_gemm(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant GemmParams& p     [[buffer(3)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint sgid                  [[simdgroup_index_in_threadgroup]],
    uint lane                  [[thread_index_in_simdgroup]]
) {
    // 64x64 output tile, 4 simdgroups (2x2), each 32x32 = 4x4 grid of 8x8
    constexpr int BM = 64, BN = 64;
    constexpr int WM = 2, WN = 2;
    constexpr int TM = BM / (8 * WM); // 4
    constexpr int TN = BN / (8 * WN); // 4

    const int bm = (int)tgid.y * BM;
    const int bn = (int)tgid.x * BN;
    const int wm = (int)(sgid / WN);
    const int wn = (int)(sgid % WN);
    const int tid = (int)(sgid * 32 + lane);

    // Direct matrix loads below are only valid for complete output tiles.
    // Bound both operands explicitly for edge tiles.
    if (bm + BM > p.M || bn + BN > p.N) {
        for (int i = tid; i < BM * BN; i += 128) {
            int row = bm + i / BN, col = bn + i % BN;
            if (row < p.M && col < p.N) {
                float sum = 0.0f;
                for (int k = 0; k < p.K; ++k) {
                    float av = p.trans_a ? A[(long)k * p.lda + row]
                                         : A[(long)row * p.lda + k];
                    float bv = p.trans_b ? B[(long)col * p.ldb + k]
                                         : B[(long)k * p.ldb + col];
                    sum += av * bv;
                }
                long dst = (long)row * p.ldc + col;
                C[dst] = p.beta == 0.0f ? p.alpha * sum
                                          : p.alpha * sum + p.beta * C[dst];
            }
        }
        return;
    }

    // Per-simdgroup 32x32 output starts at (sm, sn)
    const int sm = bm + wm * 32;
    const int sn = bn + wn * 32;

    // 4x4 accumulator grid of 8x8 simdgroup matrices (16 per simd group)
    simdgroup_float8x8 acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            acc[i][j] = simdgroup_float8x8(0);

    const int K_aligned = (p.K / 8) * 8;

    // ---- Main loop: direct device loads, no threadgroup, no barriers ----
    // Each simd group loads its own A and B fragments independently.
    // L2 cache provides cross-simdgroup data reuse (validated by
    // ThunderMittens: 9% faster than MLX's threadgroup approach on M2 Pro).

    if (!p.trans_a && p.trans_b) {
        // NT: C = A(M,K) @ B(N,K)^T — forward pass, Muon
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_float8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)(sm + i*8) * p.lda + k, p.lda);
            for (int j = 0; j < TN; j++) {
                simdgroup_float8x8 b_frag;
                simdgroup_load(b_frag, B + (long)(sn + j*8) * p.ldb + k, p.ldb,
                               ulong2(0,0), true);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    } else if (!p.trans_a && !p.trans_b) {
        // NN: C = A(M,K) @ B(K,N) — backward input grad, Muon addmm
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_float8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)(sm + i*8) * p.lda + k, p.lda);
            for (int j = 0; j < TN; j++) {
                simdgroup_float8x8 b_frag;
                simdgroup_load(b_frag, B + (long)k * p.ldb + sn + j*8, p.ldb);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    } else if (p.trans_a && !p.trans_b) {
        // TN: C = A(K,M)^T @ B(K,N) — backward weight grad
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_float8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)k * p.lda + sm + i*8, p.lda,
                               ulong2(0,0), true);
            for (int j = 0; j < TN; j++) {
                simdgroup_float8x8 b_frag;
                simdgroup_load(b_frag, B + (long)k * p.ldb + sn + j*8, p.ldb);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    }

    // Single threadgroup allocation shared between K-remainder and store phases.
    // These phases are sequential (barriers between), so memory is safely reused.
    // The compiler doubles explicit threadgroup memory (simdgroup register spill),
    // so separate arrays for sA+sB+sC exceed 32KB. Aliasing them into one buffer
    // keeps total at BM*BN*4*2 = 32768 = limit.
    constexpr int SMEM_STRIDE = BN;
    threadgroup float _smem[BM * SMEM_STRIDE];

    // ---- K-remainder: threadgroup fallback for last partial chunk ----
    // Only triggered when K % 8 != 0 (e.g., K=373). Loads remaining
    // elements into zero-padded 8-wide threadgroup tiles.
    // Reinterprets _smem as sA (BM×9 float) and sB (8×(BN+1) float).
    if (K_aligned < p.K) {
        threadgroup float* sA = _smem;                // BM*9 = 576 floats
        threadgroup float* sB = _smem + BM * 9;       // 8*(BN+1) = 520 floats
        constexpr int sB_stride = BN + 1;

        int k = K_aligned;
        int rem = p.K - k;

        // Cooperative load A: BM × 8 (zero-padded beyond rem)
        int total_a = BM * 8;
        for (int idx = tid; idx < total_a; idx += 128) {
            int r = idx / 8;
            int c = idx % 8;
            int gr = bm + r;
            int gc = k + c;
            sA[r * 9 + c] = (gr < p.M && c < rem)
                ? (p.trans_a ? A[(long)gc * p.lda + gr] : A[(long)gr * p.lda + gc])
                : 0.0f;
        }

        // Cooperative load B: 8 × BN (zero-padded beyond rem)
        int total_b = 8 * BN;
        for (int idx = tid; idx < total_b; idx += 128) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = k + r;
            int gc = bn + c;
            sB[r * sB_stride + c] = (r < rem && gc < p.N)
                ? (p.trans_b ? B[(long)gc * p.ldb + gr] : B[(long)gr * p.ldb + gc])
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a_frag[TM];
        for (int i = 0; i < TM; i++)
            simdgroup_load(a_frag[i], sA + (wm * 32 + i * 8) * 9, 9);
        for (int j = 0; j < TN; j++) {
            simdgroup_float8x8 b_frag;
            simdgroup_load(b_frag, sB + (wn * 32 + j * 8), sB_stride);
            for (int i = 0; i < TM; i++)
                simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
        }
    }

    // ---- Store results ----
    // Fast path: direct simdgroup_store for interior tiles with alpha=1, beta=0.
    // Slow path: threadgroup staging via _smem for edge tiles and non-trivial alpha/beta.
    // CRITICAL: condition must be uniform across the threadgroup — all simdgroups
    // must take the same branch, otherwise the threadgroup_barrier in the slow
    // path causes undefined behavior (some threads skip it).
    bool fast_store = (p.alpha == 1.0f && p.beta == 0.0f
                       && bm + BM <= p.M && bn + BN <= p.N);
    if (fast_store) {
        for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
                simdgroup_store(acc[i][j],
                    C + (long)(sm + i*8) * p.ldc + sn + j*8, p.ldc);
    } else {
        // Other simdgroups may still be reading the partial-K tile in _smem.
        if (K_aligned < p.K)
            threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
                simdgroup_store(acc[i][j],
                    _smem + (wm*32 + i*8) * SMEM_STRIDE + (wn*32 + j*8),
                    SMEM_STRIDE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative store to device: each thread handles BM*BN/128 = 32 elements
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = bm + r;
            int gc = bn + c;
            if (gr < p.M && gc < p.N) {
                long out_idx = (long)gr * p.ldc + gc;
                float val = _smem[r * SMEM_STRIDE + c];
                if (p.beta == 0.0f)
                    C[out_idx] = p.alpha * val;
                else
                    C[out_idx] = p.alpha * val + p.beta * C[out_idx];
            }
        }
    }
}

// Used when N doesn't meet tensor_ops alignment (N%32!=0).
// One threadgroup per output row, threads partition N columns.
// C(M,N) = A(M,K) @ B(N,K)^T, all row-major.

struct SmallGemmParams {
    uint M;
    uint N;
    uint K;
    uint rows;
};

kernel void small_gemm_nt_f32(
    const device float* A          [[buffer(0)]],
    const device float* B          [[buffer(1)]],
    device float* C                [[buffer(2)]],
    constant SmallGemmParams& p    [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    // tgid indexes a block of p.rows output rows; threads partition (row, column).
    uint row_in_tile = tid / p.N;
    uint col = tid % p.N;
    if (row_in_tile >= p.rows) return;
    uint m = tgid * p.rows + row_in_tile;
    if (m >= p.M || col >= p.N) return;

    const device float* a_row = A + m * p.K;
    const device float* b_row = B + col * p.K;

    float sum = 0.0f;
    // float4 vectorized accumulation (K must be multiple of 4)
    uint K4 = p.K & ~3u;
    for (uint k = 0; k < K4; k += 4) {
        float4 a4 = float4(*reinterpret_cast<const device packed_float4*>(a_row + k));
        float4 b4 = float4(*reinterpret_cast<const device packed_float4*>(b_row + k));
        sum += dot(a4, b4);
    }
    // handle remainder (K not multiple of 4)
    for (uint k = K4; k < p.K; k++)
        sum += a_row[k] * b_row[k];

    C[m * p.N + col] = sum;
}

// ============================================================================
// Section 27: Steel GEMM fp16 — half I/O, float accumulation
//
// Same 64x64 tiled GEMM as steel_gemm but with half-precision inputs/outputs.
// Uses simdgroup_half8x8 for loads, simdgroup_float8x8 for accumulation
// (mixed-precision multiply_accumulate). Stores back as half.
// ============================================================================

kernel void steel_gemm_f16(
    device const half* A       [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device half* C             [[buffer(2)]],
    constant GemmParams& p     [[buffer(3)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint sgid                  [[simdgroup_index_in_threadgroup]],
    uint lane                  [[thread_index_in_simdgroup]]
) {
    constexpr int BM = 64, BN = 64;
    constexpr int WM = 2, WN = 2;
    constexpr int TM = BM / (8 * WM); // 4
    constexpr int TN = BN / (8 * WN); // 4

    const int bm = (int)tgid.y * BM;
    const int bn = (int)tgid.x * BN;
    const int wm = (int)(sgid / WN);
    const int wn = (int)(sgid % WN);
    const int tid = (int)(sgid * 32 + lane);

    // As in steel_gemm, direct matrix loads require a complete output tile.
    if (bm + BM > p.M || bn + BN > p.N) {
        for (int i = tid; i < BM * BN; i += 128) {
            int row = bm + i / BN, col = bn + i % BN;
            if (row < p.M && col < p.N) {
                float sum = 0.0f;
                for (int k = 0; k < p.K; ++k) {
                    float av = float(p.trans_a ? A[(long)k * p.lda + row]
                                               : A[(long)row * p.lda + k]);
                    float bv = float(p.trans_b ? B[(long)col * p.ldb + k]
                                               : B[(long)k * p.ldb + col]);
                    sum += av * bv;
                }
                long dst = (long)row * p.ldc + col;
                C[dst] = half(p.beta == 0.0f ? p.alpha * sum
                    : p.alpha * sum + p.beta * float(C[dst]));
            }
        }
        return;
    }

    const int sm = bm + wm * 32;
    const int sn = bn + wn * 32;

    // f32 accumulators for numerical stability
    simdgroup_float8x8 acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            acc[i][j] = simdgroup_float8x8(0);

    const int K_aligned = (p.K / 8) * 8;

    if (!p.trans_a && p.trans_b) {
        // NT: C = A(M,K) @ B(N,K)^T
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_half8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)(sm + i*8) * p.lda + k, p.lda);
            for (int j = 0; j < TN; j++) {
                simdgroup_half8x8 b_frag;
                simdgroup_load(b_frag, B + (long)(sn + j*8) * p.ldb + k, p.ldb,
                               ulong2(0,0), true);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    } else if (!p.trans_a && !p.trans_b) {
        // NN: C = A(M,K) @ B(K,N)
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_half8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)(sm + i*8) * p.lda + k, p.lda);
            for (int j = 0; j < TN; j++) {
                simdgroup_half8x8 b_frag;
                simdgroup_load(b_frag, B + (long)k * p.ldb + sn + j*8, p.ldb);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    } else if (p.trans_a && !p.trans_b) {
        // TN: C = A(K,M)^T @ B(K,N)
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_half8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)k * p.lda + sm + i*8, p.lda,
                               ulong2(0,0), true);
            for (int j = 0; j < TN; j++) {
                simdgroup_half8x8 b_frag;
                simdgroup_load(b_frag, B + (long)k * p.ldb + sn + j*8, p.ldb);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    }

    // Single threadgroup allocation shared between K-remainder and store phases.
    // These phases are sequential (barriers between), so memory is safely reused.
    // Store needs BM*BN floats = 16384 bytes (largest consumer).
    // K-remainder needs BM*9 + 8*(BN+1) halves = 2192 bytes (fits inside).
    // Without aliasing, the compiler allocates all three arrays statically and
    // exceeds the 32KB threadgroup memory limit. The compiler doubles explicit
    // threadgroup memory (simdgroup register spill), so BM*BN*4*2 = 32768 = limit.
    // No +1 stride padding: minor bank conflict on simdgroup_store vs. correctness.
    constexpr int SMEM_STRIDE = BN;
    threadgroup float _smem[BM * SMEM_STRIDE];

    // K-remainder: reinterpret _smem as half arrays for partial-K accumulation
    if (K_aligned < p.K) {
        threadgroup half* sA = (threadgroup half*)_smem;
        threadgroup half* sB = sA + BM * 9;
        constexpr int sB_stride = BN + 1;

        int k = K_aligned;
        int rem = p.K - k;

        int total_a = BM * 8;
        for (int idx = tid; idx < total_a; idx += 128) {
            int r = idx / 8;
            int c = idx % 8;
            int gr = bm + r;
            int gc = k + c;
            sA[r * 9 + c] = (gr < p.M && c < rem)
                ? (p.trans_a ? A[(long)gc * p.lda + gr] : A[(long)gr * p.lda + gc])
                : half(0);
        }

        int total_b = 8 * BN;
        for (int idx = tid; idx < total_b; idx += 128) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = k + r;
            int gc = bn + c;
            sB[r * sB_stride + c] = (r < rem && gc < p.N)
                ? (p.trans_b ? B[(long)gc * p.ldb + gr] : B[(long)gr * p.ldb + gc])
                : half(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 a_frag[TM];
        for (int i = 0; i < TM; i++)
            simdgroup_load(a_frag[i], sA + (wm * 32 + i * 8) * 9, 9);
        for (int j = 0; j < TN; j++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, sB + (wn * 32 + j * 8), sB_stride);
            for (int i = 0; i < TM; i++)
                simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
        }
    }

    // Barrier: K-remainder reads _smem as half, store writes it as float.
    // Without this, a fast simdgroup's float writes corrupt a slow one's half reads.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store: f32 acc → half output via _smem staging (reuses same memory)
    // simdgroup_store of float8x8 requires float* destination, so we stage
    // through _smem then convert element-by-element to half for output.
    {
        for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
                simdgroup_store(acc[i][j],
                    _smem + (wm*32 + i*8) * SMEM_STRIDE + (wn*32 + j*8),
                    SMEM_STRIDE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int idx = tid; idx < BM * BN; idx += 128) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = bm + r;
            int gc = bn + c;
            if (gr < p.M && gc < p.N) {
                long out_idx = (long)gr * p.ldc + gc;
                float val = _smem[r * SMEM_STRIDE + c];
                if (p.beta == 0.0f)
                    C[out_idx] = half(p.alpha * val);
                else
                    C[out_idx] = half(p.alpha * val + p.beta * float(C[out_idx]));
            }
        }
    }
}

// Linear recurrent scan matching the 5.0 training recurrence. The saved
// state is the post-reset input to each step, not the output of the step.
struct LinearScanParams { int T, H, B, has_terminals; };

template<typename T>
void linear_scan_fwd(device T* out, device T* next, device float* saved,
    const device T* combined, const device T* state, const device T* input,
    const device float* terminals, constant LinearScanParams& p, uint idx) {
    if (idx >= uint(p.B * p.H)) return;
    int b = idx / p.H, h = idx % p.H;
    float ht = float(state[idx]);
    for (int t = 0; t < p.T; ++t) {
        int x = (b * p.T + t) * p.H + h;
        int c = (b * p.T + t) * 3 * p.H + h;
        if (p.has_terminals && terminals[b * p.T + t] != 0.0f) ht = 0.0f;
        saved[x] = ht;
        float hidden = float(combined[c]);
        float z = sigmoid_f(float(combined[c + p.H]));
        float candidate = hidden >= 0.0f ? hidden + 0.5f : sigmoid_f(hidden);
        float diff = candidate - ht;
        ht = abs(z) < 0.5f ? ht + z * diff : candidate - diff * (1.0f - z);
        float s = sigmoid_f(float(combined[c + 2 * p.H]));
        out[x] = T(s * ht + (1.0f - s) * float(input[x]));
        ht = float(T(ht));
    }
    next[idx] = T(ht);
}

template<typename T>
void linear_scan_bwd(device T* grad_combined, device T* grad_state,
    device T* grad_input, const device T* grad_out, const device T* grad_next,
    const device T* combined, const device T* input, const device float* saved,
    const device float* terminals, constant LinearScanParams& p, uint idx) {
    if (idx >= uint(p.B * p.H)) return;
    int b = idx / p.H, h = idx % p.H;
    float dh = float(grad_next[idx]);
    for (int t = p.T - 1; t >= 0; --t) {
        int x = (b * p.T + t) * p.H + h;
        int c = (b * p.T + t) * 3 * p.H + h;
        float prev = saved[x], hidden = float(combined[c]);
        float z = sigmoid_f(float(combined[c + p.H]));
        float candidate = hidden >= 0.0f ? hidden + 0.5f : sigmoid_f(hidden);
        float diff = candidate - prev;
        float ht = abs(z) < 0.5f ? prev + z * diff : candidate - diff * (1.0f - z);
        float s = sigmoid_f(float(combined[c + 2 * p.H]));
        float go = float(grad_out[x]);
        grad_input[x] = T(go * (1.0f - s));
        grad_combined[c + 2 * p.H] = T(go * (ht - float(input[x])) * s * (1.0f - s));
        float total = dh + go * s;
        float dc = total * z;
        grad_combined[c] = T(hidden >= 0.0f ? dc : dc * candidate * (1.0f - candidate));
        grad_combined[c + p.H] = T(total * (candidate - prev) * z * (1.0f - z));
        dh = total * (1.0f - z);
        if (p.has_terminals && terminals[b * p.T + t] != 0.0f) dh = 0.0f;
    }
    grad_state[idx] = T(dh);
}

#define LINEAR_SCAN_KERNELS(SUFFIX, TYPE) \
kernel void linear_scan_forward_##SUFFIX( \
    device TYPE* out [[buffer(0)]], device TYPE* next [[buffer(1)]], \
    device float* saved [[buffer(2)]], const device TYPE* combined [[buffer(5)]], \
    const device TYPE* state [[buffer(6)]], const device TYPE* input [[buffer(7)]], \
    constant LinearScanParams& p [[buffer(8)]], const device float* terminals [[buffer(9)]], \
    uint idx [[thread_position_in_grid]]) { \
    linear_scan_fwd(out, next, saved, combined, state, input, terminals, p, idx); \
} \
kernel void linear_scan_backward_##SUFFIX( \
    device TYPE* gc [[buffer(0)]], device TYPE* gs [[buffer(1)]], \
    device TYPE* gi [[buffer(2)]], const device TYPE* go [[buffer(3)]], \
    const device TYPE* gn [[buffer(4)]], const device TYPE* combined [[buffer(5)]], \
    const device TYPE* input [[buffer(7)]], const device float* saved [[buffer(8)]], \
    constant LinearScanParams& p [[buffer(11)]], const device float* terminals [[buffer(12)]], \
    uint idx [[thread_position_in_grid]]) { \
    linear_scan_bwd(gc, gs, gi, go, gn, combined, input, saved, terminals, p, idx); \
}
LINEAR_SCAN_KERNELS(f32, float)
LINEAR_SCAN_KERNELS(f16, half)
#undef LINEAR_SCAN_KERNELS

// 5.0 GAE: values are the current minibatch predictions, while rewards and
// terminals are aligned with observations (the transition is at t + 1).
struct Puf5AdvantageParams {
    float gamma, lambda, rho_clip, c_clip;
    int rows, horizon, use_importance;
};
kernel void puf5_advantage(
    const device float* values [[buffer(0)]],
    const device float* rewards [[buffer(1)]],
    const device float* terminals [[buffer(2)]],
    const device float* importance [[buffer(3)]],
    device float* advantages [[buffer(4)]],
    device float* returns [[buffer(5)]],
    constant Puf5AdvantageParams& p [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= uint(p.rows)) return;
    int base = int(row) * p.horizon;
    int last = base + p.horizon - 1;
    float next_v = values[last];
    float next_d = terminals[last];
    float next_r = rewards[last];
    float carry = 0.0f;
    advantages[last] = 0.0f;
    returns[last] = next_v;
    for (int t = p.horizon - 2; t >= 0; --t) {
        int i = base + t;
        float v = values[i];
        float imp = p.use_importance ? importance[i] : 1.0f;
        float nonterminal = 1.0f - next_d;
        float delta = min(imp, p.rho_clip) *
            (next_r + p.gamma * next_v * nonterminal - v);
        carry = delta + p.gamma * p.lambda * min(imp, p.c_clip) *
            carry * nonterminal;
        advantages[i] = carry;
        returns[i] = v + carry;
        next_v = v;
        next_d = terminals[i];
        next_r = rewards[i];
    }
}

struct Puf5PPOParams {
    int count, heads, actions, continuous;
    float clip, vf_clip, vf_coef, ent_coef;
};

struct Puf5NethackParams {
    int head_stride;
    float verb_epsilon;
};

struct Puf5MetricsPartial {
    float policy, value, entropy, total, old_kl, kl, clipfrac, importance;
    uint nonfinite;
};

// One fixed-size threadgroup produces one minibatch's sums. Each lane visits
// rows in increasing strided order, then the tree reduces in a fixed order.
kernel void puf5_reduce_metrics(
    device Puf5MetricsPartial* partial [[buffer(0)]],
    const device float* logits [[buffer(1)]],
    const device float* old_values [[buffer(2)]],
    const device float* advantages [[buffer(3)]],
    const device float* returns [[buffer(4)]],
    const device float* ratios [[buffer(5)]],
    const device float* entropy [[buffer(6)]],
    const device float* logratio [[buffer(7)]],
    constant Puf5PPOParams& p [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[8][256];
    threadgroup uint bad[256];
    float local[8] = {};
    uint nonfinite = 0;
    for (int i = int(tid); i < p.count; i += 256) {
        float ratio = ratios[i], adv = advantages[i];
        float v = logits[i * (p.actions + 1) + p.actions];
        float old_v = old_values[i], target = returns[i];
        float clipped_v = old_v + fmax(-p.vf_clip, fmin(p.vf_clip, v - old_v));
        float error = v - target, clipped_error = clipped_v - target;
        float vf = 0.5f * fmax(error * error, clipped_error * clipped_error);
        float clipped_ratio = fmax(1.0f - p.clip, fmin(1.0f + p.clip, ratio));
        float pg = fmax(-adv * ratio, -adv * clipped_ratio);
        float ent = entropy[i], lr = logratio[i];
        float total = pg + p.vf_coef * vf - p.ent_coef * ent;
        nonfinite += uint(!isfinite(pg) || !isfinite(vf) || !isfinite(ratio) ||
                          !isfinite(ent) || !isfinite(lr) || !isfinite(total));
        local[0] += pg;
        local[1] += vf;
        local[2] += ent;
        local[3] += total;
        local[4] -= lr;
        local[5] += (ratio - 1.0f) - lr;
        local[6] += abs(ratio - 1.0f) > p.clip ? 1.0f : 0.0f;
        local[7] += ratio;
    }
    for (int c = 0; c < 8; ++c) sums[c][tid] = local[c];
    bad[tid] = nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int c = 0; c < 8; ++c) sums[c][tid] += sums[c][tid + stride];
            bad[tid] += bad[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partial->policy = sums[0][0];
        partial->value = sums[1][0];
        partial->entropy = sums[2][0];
        partial->total = sums[3][0];
        partial->old_kl = sums[4][0];
        partial->kl = sums[5][0];
        partial->clipfrac = sums[6][0];
        partial->importance = sums[7][0];
        partial->nonfinite = bad[0];
    }
}

inline float puf5_verb_logprob(float lp, const device float* mask, int size,
                               float eps, thread float& scale) {
    if (eps <= 0.0f) return lp;
    int legal_count = 0;
    for (int j = 0; j < size; ++j) legal_count += mask[j] != 0.0f;
    if (legal_count == 0) legal_count = size;
    float prob = exp(lp), mixed = (1.0f - eps) * prob + eps / float(legal_count);
    scale = (1.0f - eps) * prob / mixed;
    return log(mixed);
}

inline float puf5_finite_clamp(float x, float lo, float hi) {
    return isnan(x) ? 0.0f : clamp(x, lo, hi);
}

inline float puf5_logit(const device float* logits, const device float* mask,
                        int index) {
    return mask[index] == 0.0f ? -1.0e4f : logits[index];
}

// Return the current joint action log probability. The same masked
// distribution is used by both the V-trace cache and PPO gradients.
inline float puf5_logprob(const device float* logits, const device float* logstd,
                          const device float* actions, const device float* mask,
                          const device int* sizes, constant Puf5PPOParams& p,
                          const device char* head_map, constant Puf5NethackParams& nh,
                          int row) {
    int off = row * (p.actions + 1);
    float lp = 0.0f;
    int head_off = 0;
    for (int h = 0; h < p.heads; ++h) {
        if (p.continuous) {
            float mean = puf5_finite_clamp(logits[off + h], -1.0e6f, 1.0e6f);
            float ls = puf5_finite_clamp(logstd[h], -20.0f, 2.0f);
            float z = (actions[row * p.heads + h] - mean) / exp(ls);
            lp += -0.5f * z * z - 0.9189385332046727f - ls;
        } else {
            int size = sizes[h];
            int verb = int(actions[row * p.heads]);
            if (h != 0 && nh.head_stride != 0 && !head_map[verb * nh.head_stride + h]) {
                head_off += size;
                continue;
            }
            float maximum = -INFINITY;
            for (int j = 0; j < size; ++j)
                maximum = max(maximum, puf5_logit(logits + off, mask + row * p.actions, head_off + j));
            float sum = 0.0f;
            for (int j = 0; j < size; ++j)
                sum += exp(puf5_logit(logits + off, mask + row * p.actions, head_off + j) - maximum);
            int action = int(actions[row * p.heads + h]);
            float lse = maximum + log(sum);
            float head_lp = puf5_logit(logits + off, mask + row * p.actions, head_off + action) - lse;
            float scale = 1.0f;
            lp += h == 0 ? puf5_verb_logprob(head_lp, mask + row * p.actions,
                                              size, nh.verb_epsilon, scale) : head_lp;
            head_off += size;
        }
    }
    return lp;
}

kernel void puf5_cache_policy(
    const device float* logits [[buffer(0)]], const device float* logstd [[buffer(1)]],
    const device float* actions [[buffer(2)]], const device float* old_logp [[buffer(3)]],
    const device float* mask [[buffer(4)]], const device int* sizes [[buffer(5)]],
    device float* values [[buffer(6)]], device float* importance [[buffer(7)]],
    constant Puf5PPOParams& p [[buffer(8)]],
    const device char* head_map [[buffer(9)]],
    constant Puf5NethackParams& nh [[buffer(10)]], uint row [[thread_position_in_grid]]) {
    if (row >= uint(p.count)) return;
    values[row] = logits[row * (p.actions + 1) + p.actions];
    importance[row] = exp(puf5_logprob(logits, logstd, actions, mask, sizes, p, head_map, nh, int(row)) - old_logp[row]);
}

kernel void puf5_ppo_grad(
    const device float* logits [[buffer(0)]], const device float* logstd [[buffer(1)]],
    const device float* actions [[buffer(2)]], const device float* old_logp [[buffer(3)]],
    const device float* mask [[buffer(4)]], const device int* sizes [[buffer(5)]],
    const device float* old_values [[buffer(6)]], const device float* advantages [[buffer(7)]],
    const device float* returns [[buffer(8)]], device float* grad_logits [[buffer(9)]],
    device float* grad_logstd [[buffer(10)]], device float* grad_values [[buffer(11)]],
    constant Puf5PPOParams& p [[buffer(12)]],
    const device char* head_map [[buffer(13)]],
    constant Puf5NethackParams& nh [[buffer(14)]],
    device float* out_entropy [[buffer(15)]], device float* out_logratio [[buffer(16)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= uint(p.count)) return;
    int r = int(row), out = r * (p.actions + 1), base = r * p.actions;
    float logratio = puf5_logprob(logits, logstd, actions, mask, sizes, p, head_map, nh, r) - old_logp[r];
    float ratio = exp(logratio);
    out_logratio[r] = logratio;
    float clipped = clamp(ratio, 1.0f - p.clip, 1.0f + p.clip);
    float wa = -advantages[r];
    float dlogp = wa / float(p.count);
    if (wa * clipped > wa * ratio && (ratio <= 1.0f - p.clip || ratio >= 1.0f + p.clip))
        dlogp = 0.0f;
    dlogp *= ratio;
    float v = logits[out + p.actions], old_v = old_values[r], ret = returns[r];
    float vc = old_v + clamp(v - old_v, -p.vf_clip, p.vf_clip);
    float raw_error = v - ret, clipped_error = vc - ret;
    grad_values[r] = p.vf_coef / float(p.count) *
        (clipped_error * clipped_error > raw_error * raw_error ? 0.0f : raw_error);
    float total_entropy = 0.0f;
    int head_off = 0;
    for (int h = 0; h < p.heads; ++h) {
        if (p.continuous) {
            float mean = puf5_finite_clamp(logits[out + h], -1.0e6f, 1.0e6f);
            float ls = puf5_finite_clamp(logstd[h], -20.0f, 2.0f);
            total_entropy += 1.4189385332046727f + ls;
            float action = puf5_finite_clamp(actions[r * p.heads + h], -1.0e6f, 1.0e6f);
            float diff = action - mean, std = exp(ls), variance = std * std;
            grad_logits[base + h] = dlogp * diff / variance;
            grad_logstd[base + h] = dlogp * (diff * diff / variance - 1.0f) - p.ent_coef / float(p.count);
        } else {
            int size = sizes[h], act = int(actions[r * p.heads + h]);
            int verb = int(actions[r * p.heads]);
            if (h != 0 && nh.head_stride != 0 && !head_map[verb * nh.head_stride + h]) {
                for (int j = 0; j < size; ++j) grad_logits[base + head_off + j] = 0.0f;
                head_off += size;
                continue;
            }
            float maximum = -INFINITY;
            for (int j = 0; j < size; ++j)
                maximum = max(maximum, puf5_logit(logits + out, mask + base, head_off + j));
            float sum = 0.0f;
            for (int j = 0; j < size; ++j)
                sum += exp(puf5_logit(logits + out, mask + base, head_off + j) - maximum);
            float lse = maximum + log(sum), entropy = 0.0f;
            for (int j = 0; j < size; ++j) {
                float logp = puf5_logit(logits + out, mask + base, head_off + j) - lse;
                entropy -= exp(logp) * logp;
            }
            total_entropy += entropy;
            float head_dlogp = dlogp;
            if (h == 0 && nh.verb_epsilon > 0.0f) {
                float scale = 1.0f;
                float action_lp = puf5_logit(logits + out, mask + base, act) - lse;
                puf5_verb_logprob(action_lp, mask + base, size, nh.verb_epsilon, scale);
                head_dlogp *= scale;
            }
            for (int j = 0; j < size; ++j) {
                float logp = puf5_logit(logits + out, mask + base, head_off + j) - lse;
                float prob = exp(logp);
                grad_logits[base + head_off + j] = (float(j == act) - prob) * head_dlogp
                    - p.ent_coef / float(p.count) * prob * (-entropy - logp);
            }
            head_off += size;
        }
    }
    out_entropy[r] = total_entropy;
}

)METAL";
}

#endif // PUFFERLIB_METAL_SHADER_SRC_H
