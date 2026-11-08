// Native 5.0 Metal training executable. Compile this translation unit together
// with metal_platform.mm (metal_kernels.mm is included for its policy API).
#include "ini.h"
// Metal headers must parse before environment code: some environments define
// function-like min/max macros that collide with the SDK's declarations.
#include "metal_platform.h"
#include ENV_HEADER
#include "metal_env.h"
// Some environments define function-like min/max macros. They have already been
// expanded inside the environment header; drop them so C++ and SDK headers parse.
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include "metal_kernels.mm"
#include <algorithm>
#include <vector>
#include <cerrno>
#include <ctime>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <exception>
#include <chrono>
#include <type_traits>

static FloatTensor fview(float *p, int a, int b = 0, int c = 0) {
    FloatTensor t = {}; t.data = p; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c; return t;
}
static PrecisionTensor pview(float *p, int a, int b = 0, int c = 0) {
    PrecisionTensor t = {}; t.data = p; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c; t.dtype_size = 4; return t;
}
static void wrap(Allocator &a) { a.create(); if (a.mem) mtl_wrap_allocator(&a); }
#include "metal_custom_encoder.h"
#ifdef PUFFER_NETHACK
#include "metal_nethack_trainer.h"
#endif
static float metal_cosine_annealing(float base, float min_value, long t, long total) {
    double ratio = (double)t / (double)total;
    return min_value + 0.5f * (base - min_value) * (1.0f + (float)std::cos(M_PI * ratio));
}


// 5.0's default BF16 parameter allocator uses 16-byte alignment; its saved
// master weights are float32 with the corresponding eight-float alignment.
static size_t checkpoint_size(Allocator &params) {
    size_t n = 0;
    for (auto &e : params.regs) {
        n = (n + 7) & ~size_t(7);
        n += (size_t)puf_numel(e.shape);
    }
    return n;
}
static void checkpoint_io(Allocator &params, const char *path, bool save) {
    std::vector<float> data(checkpoint_size(params), 0.0f);
    FILE *fp = fopen(path, save ? "wb" : "rb");
    if (!fp) { perror(path); exit(1); }
    if (!save) {
        if (fread(data.data(), sizeof(float), data.size(), fp) != data.size() || fgetc(fp) != EOF || ferror(fp)) {
            fprintf(stderr, "invalid checkpoint size: %s\n", path); exit(1);
        }
    }
    size_t off = 0;
    for (auto &e : params.regs) {
        off = (off + 7) & ~size_t(7);
        size_t n = (size_t)puf_numel(e.shape);
        if (save) memcpy(data.data() + off, *e.data_ptr, n * sizeof(float));
        else memcpy(*e.data_ptr, data.data() + off, n * sizeof(float));
        off += n;
    }
    if (save && fwrite(data.data(), sizeof(float), data.size(), fp) != data.size()) {
        perror(path); exit(1);
    }
    if (fclose(fp)) { perror(path); exit(1); }
}
static void make_dirs(std::string path) {
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            std::string dir = path.substr(0, i);
            if (mkdir(dir.c_str(), 0777) && errno != EEXIST) { perror(dir.c_str()); exit(1); }
        }
    }
}
static void find_latest_impl(const std::string &dir, std::string &best,
        time_t &best_time, long &best_nsec) {
    DIR *dp = opendir(dir.c_str());
    if (!dp) return;
    for (dirent *ent = readdir(dp); ent; ent = readdir(dp)) {
        if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) continue;
        std::string path = dir + "/" + ent->d_name;
        struct stat st;
        if (stat(path.c_str(), &st)) continue;
        if (S_ISDIR(st.st_mode)) find_latest_impl(path, best, best_time, best_nsec);
        else if (S_ISREG(st.st_mode) && path.size() >= 4 &&
                 path.compare(path.size() - 4, 4, ".bin") == 0 &&
                 (st.st_ctime > best_time ||
                  (st.st_ctime == best_time &&
                   (st.st_ctimespec.tv_nsec > best_nsec ||
                    (st.st_ctimespec.tv_nsec == best_nsec && (best.empty() || path > best)))))) {
            best = path; best_time = st.st_ctime; best_nsec = st.st_ctimespec.tv_nsec;
        }
    }
    closedir(dp);
}
static void find_latest(const std::string &dir, std::string &best, time_t &best_time) {
    long best_nsec = -1;
    find_latest_impl(dir, best, best_time, best_nsec);
}
#include "metal_eval.h"
#include "metal_selfplay.h"
struct MetalTrainResult {
    float score = 0, cost = 0, steps = 0;
    int points = 0;
    float scores[64] = {}, costs[64] = {}, step_points[64] = {};
};
struct MetalTrainPoint { double score, cost, steps; };


static void train_metal(Ini *ini) {
#ifdef PUFFER_NETHACK
    const bool nethack = true;
#else
    const bool nethack = false;
#endif
    if (puf_ini_get(ini, "train", "gpus") != 1 ||
            puf_ini_get(ini, "base", "gpu_offset") != 0) {
        fprintf(stderr, "Metal training supports one default GPU (train.gpus=1, base.gpu_offset=0)\n");
        exit(1);
    }
    bool async = puf_ini_get(ini, "base", "async") != 0;
    bool selfplay = puf_ini_get(ini, "selfplay", "enabled") != 0;
    if (!selfplay) {
        puf_ini_put(ini, "vec.num_policies", "1");
        puf_ini_put(ini, "vec.hist_policy_percent", "0");
    }
    int agents = (int)puf_ini_get(ini, "vec", "total_agents");
    int buffers = (int)puf_ini_get(ini, "vec", "num_buffers");
    int hidden = (int)puf_ini_get(ini, "policy", "hidden_size");
    int layers = (int)puf_ini_get(ini, "policy", "num_layers");
    int horizon = (int)puf_ini_get(ini, "train", "horizon");
    int mbsize = (int)puf_ini_get(ini, "train", "minibatch_size");
    long steps = (long)puf_ini_get(ini, "train", "total_timesteps");
    int policies = (int)puf_ini_get(ini, "vec", "num_policies");
    int pool_max = (int)puf_ini_get(ini, "selfplay", "max_size");
    if (selfplay && (policies < 2 || policies > 9 || pool_max <= 0 ||
            puf_ini_get(ini, "vec", "hist_policy_percent") <= 0 ||
            puf_ini_get(ini, "vec", "hist_policy_percent") > 1)) {
        fprintf(stderr, "invalid selfplay policy count, pool size, or historical percentage\n"); exit(1);
    }
    int sizes[] = ACT_SIZES, action_dim = 0, continuous = 1, any_continuous = 0;
    for (int x : sizes) { action_dim += x; continuous &= x == 1; any_continuous |= x == 1; }
    if (any_continuous && !continuous) { fprintf(stderr, "mixed action spaces are unsupported\n"); exit(1); }
    if (agents <= 0 || buffers <= 0 || hidden <= 0 || layers <= 0 || horizon <= 1 ||
        mbsize <= 0 || mbsize % horizon || agents % (mbsize / horizon) || agents % buffers) {
        fprintf(stderr, "unsupported Metal configuration: require positive dimensions, horizon > 1, and integral minibatch segments dividing agents\n"); exit(1);
    }
    int segments = mbsize / horizon, count = agents * horizon;
    long epochs = steps / count;
    if (epochs <= 0) {
        fprintf(stderr, "train.total_timesteps must cover at least one complete rollout (%d steps)\n", count);
        exit(1);
    }
    float replay_ratio = puf_ini_get(ini, "train", "replay_ratio");
    int updates = (int)(replay_ratio * count / mbsize);
    int seed = (int)puf_ini_get(ini, "base", "seed");
    float gamma = puf_ini_get(ini, "train", "gamma"), lambda = puf_ini_get(ini, "train", "gae_lambda");
    float lr = puf_ini_get(ini, "train", "learning_rate");
    float momentum = puf_ini_get(ini, "train", "momentum");
    float max_norm = puf_ini_get(ini, "train", "max_grad_norm");
    bool vtrace = puf_ini_get(ini, "train", "vtrace") != 0;
    float rho = puf_ini_get(ini, "train", "vtrace_rho_clip"), c = puf_ini_get(ini, "train", "vtrace_c_clip");
    float clip = puf_ini_get(ini, "train", "clip_coef"), vfclip = puf_ini_get(ini, "train", "vf_clip_coef");
    float vfcoef = puf_ini_get(ini, "train", "vf_coef"), ent = puf_ini_get(ini, "train", "ent_coef");
    float min_lr_ratio = puf_ini_get(ini, "train", "min_lr_ratio");
    float min_ent_ratio = puf_ini_get(ini, "train", "min_ent_coef_ratio");

    MetalEnvVec env = {};
    metal_env_create(&env, agents, buffers, policies, puf_ini_section(ini, "vec", 0), puf_ini_section(ini, "env", 0));
    metal_env_reset(&env);
    mtl_init(); puf_set_gpu_training(true);
    cudaStream_t actor_stream = mtl_stream();
    cudaStream_t stream = async ? mtl_train_stream() : actor_stream;
    EncoderWeights ew = {}; ew.in_dim = OBS_SIZE; ew.out_dim = hidden;
    DecoderWeights dw = {}; dw.hidden_dim = hidden; dw.output_dim = action_dim; dw.continuous = continuous;
    MinGRUWeights nw = {}; nw.hidden = hidden; nw.num_layers = layers; nw.horizon = horizon; nw.weights.resize(layers);
    Allocator params, acts, grads, optalloc;
    // Mixed precision for the steering GEMMs: half operands, float32 masters.
    bool fp16_mingru = !nethack && puf_ini_get(ini, "base", "mingru_fp16") != 0;
    Allocator mingru_lo;
    MetalEncoderKind encoder_kind = metal_encoder_kind();
    MetalCustomWeights cw, actor_cw;
#ifdef PUFFER_NETHACK
    MetalNethackModel nh_train, nh_actor;
    metal_nethack_model_register(nh_train, mbsize, hidden, params, grads, acts, true);
#else
    if (encoder_kind == METAL_LINEAR) encoder_reg_params(&ew, &params, 4);
    else metal_custom_params(cw, &params, encoder_kind, hidden);
    decoder_reg_params(&dw, &params, 4);
#endif
    mingru_reg_params(&nw, &params, 4);
    if (fp16_mingru) mingru_reg_params_lo(&nw, &mingru_lo);
    EncoderWeights actor_ew = {}; DecoderWeights actor_dw = {}; MinGRUWeights actor_nw = {};
    Allocator actor_params;
    if (async || nethack) {
        actor_ew.in_dim = OBS_SIZE; actor_ew.out_dim = hidden;
        actor_dw.hidden_dim = hidden; actor_dw.output_dim = action_dim; actor_dw.continuous = continuous;
        actor_nw.hidden = hidden; actor_nw.num_layers = layers; actor_nw.horizon = horizon; actor_nw.weights.resize(layers);
#ifdef PUFFER_NETHACK
        metal_nethack_model_register(nh_actor, agents, hidden, actor_params, nh_actor.unused_gradients, acts, false);
#else
        if (encoder_kind == METAL_LINEAR) encoder_reg_params(&actor_ew, &actor_params, 4);
        else metal_custom_params(actor_cw, &actor_params, encoder_kind, hidden);
        decoder_reg_params(&actor_dw, &actor_params, 4);
#endif
        mingru_reg_params(&actor_nw, &actor_params, 4);
        wrap(actor_params);
    }
    wrap(params);
    uint64_t initseed = (uint64_t)seed;
#ifdef PUFFER_NETHACK
    metal_nethack_model_init(nh_train, &initseed, stream);
#else
    if (encoder_kind == METAL_LINEAR) encoder_init_weights(&ew, &initseed, stream);
    else metal_custom_init(cw, &initseed, stream);
    decoder_init_weights(&dw, &initseed, stream);
#endif
    mingru_init_weights(&nw, &initseed, stream);
    const char *load = puf_ini_get_str(ini, "base", "load_model_path");
    if (load && strcmp(load, "None") != 0) {
        std::string latest;
        if (!strcmp(load, "latest")) {
            time_t best_time = 0;
            find_latest(std::string(puf_ini_get_str(ini, "base", "checkpoint_dir")) + "/" + PUFFER_ENV_NAME,
                        latest, best_time);
            if (latest.empty()) { fprintf(stderr, "no checkpoint found for %s\n", PUFFER_ENV_NAME); exit(1); }
            load = latest.c_str();
        }
        mtl_ensure_stream_synced(stream);
        checkpoint_io(params, load, false);
    }
    EncoderActivations re = {}, te = {}; DecoderActivations rd = {}, td = {}; MinGRUActivations rn = {}, tn = {};
    MetalCustomActivations cr, ct;
#ifndef PUFFER_NETHACK
    if (encoder_kind == METAL_LINEAR) encoder_reg_rollout(&ew, &re, &acts, agents);
    else metal_custom_activations(cw, cr, &acts, nullptr, agents);
    decoder_reg_rollout(&dw, &rd, &acts, agents);
    if (encoder_kind == METAL_LINEAR) encoder_reg_train(&ew, &te, &acts, &grads, mbsize, 4);
    else metal_custom_activations(cw, ct, &acts, &grads, mbsize);
    decoder_reg_train(&dw, &td, &acts, &grads, mbsize, 4);
#endif
    mingru_reg_rollout(&nw, &rn, &acts, agents);
    mingru_reg_train(&nw, &tn, &acts, &grads, mbsize, fp16_mingru ? 2 : 4,
                     fp16_mingru ? &mingru_lo : nullptr);
    PrecisionTensor obs = pview(nullptr, agents, OBS_SIZE), state = pview(nullptr, layers, agents, hidden);
    PrecisionTensor trainobs = pview(nullptr, segments, horizon, OBS_SIZE), trainstate = pview(nullptr, layers, segments, hidden);
    PrecisionTensor trainstate_h = pview(nullptr, layers, segments, hidden);
    PrecisionTensor x_h = pview(nullptr, mbsize, hidden);
    PrecisionTensor dh_h = pview(nullptr, segments, horizon, hidden);
    FloatTensor y_f = fview(nullptr, mbsize, hidden), gi_f = fview(nullptr, mbsize, hidden);
    if (fp16_mingru) {
        trainstate_h.dtype_size = 2; x_h.dtype_size = 2; dh_h.dtype_size = 2;
        alloc_register(&mingru_lo, &trainstate_h);
        alloc_register(&mingru_lo, &x_h);
        alloc_register(&mingru_lo, &dh_h);
        alloc_register(&acts, &y_f);
        alloc_register(&acts, &gi_f);
    }
    FloatTensor sampled = fview(nullptr, agents, NUM_ATNS), logp = fview(nullptr, agents), vals = fview(nullptr, agents);
    FloatTensor mask = fview(nullptr, agents, action_dim), mbmask = fview(nullptr, segments, horizon, action_dim);
    FloatTensor mba = fview(nullptr, segments, horizon, NUM_ATNS), mblp = fview(nullptr, segments, horizon);
    FloatTensor mbv = fview(nullptr, segments, horizon), mbr = fview(nullptr, segments, horizon), mbd = fview(nullptr, segments, horizon);
    FloatTensor livev = fview(nullptr, segments, horizon), imp = fview(nullptr, segments, horizon), adv = fview(nullptr, segments, horizon);
    FloatTensor gl = fview(nullptr, segments, horizon, action_dim), gs = fview(nullptr, segments, horizon, action_dim), gv = fview(nullptr, segments, horizon);
    FloatTensor ppo_entropy = fview(nullptr, segments, horizon), ppo_logratio = fview(nullptr, segments, horizon);
    PufTensor metric_partials = {};
    metric_partials.shape[0] = (int64_t)std::max(updates, 1) * sizeof(Puf5MetricsPartial);
    metric_partials.dtype_size = 1;
    PufTensor rng_state = {};
    rng_state.shape[0] = (int64_t)agents * sizeof(MetalRngState);
    rng_state.dtype_size = 1;
    FloatTensor norm = fview(nullptr, 1);
    IntTensor as = {}; as.shape[0] = NUM_ATNS;
    alloc_register(&acts, &obs); alloc_register(&acts, &state); alloc_register(&acts, &trainobs); alloc_register(&acts, &trainstate);
    alloc_register(&acts, &sampled); alloc_register(&acts, &logp); alloc_register(&acts, &vals); alloc_register(&acts, &mask); alloc_register(&acts, &mbmask);
    alloc_register(&acts, &mba); alloc_register(&acts, &mblp); alloc_register(&acts, &mbv); alloc_register(&acts, &mbr); alloc_register(&acts, &mbd);
    alloc_register(&acts, &livev); alloc_register(&acts, &imp); alloc_register(&acts, &adv); alloc_register(&acts, &gl); alloc_register(&acts, &gs); alloc_register(&acts, &gv); alloc_register(&acts, &norm); alloc_register(&acts, &as);
    alloc_register(&acts, &ppo_entropy); alloc_register(&acts, &ppo_logratio);
    alloc_register(&acts, &metric_partials);
    alloc_register(&acts, &rng_state);
#ifdef PUFFER_NETHACK
    int n_verbs = 0, head_stride = 0;
    const signed char *head_host = env_head_consume_map(&n_verbs, &head_stride);
    PufTensor head_map = {};
    if (head_host && n_verbs > 0 && head_stride > 0) {
        head_map.shape[0] = n_verbs * head_stride; head_map.dtype_size = 1;
        alloc_register(&acts, &head_map);
    }
#endif
    wrap(acts); wrap(grads); memcpy(as.data, sizes, sizeof(sizes));
    if (fp16_mingru) wrap(mingru_lo);
    mtl_rng_init((MetalRngState*)rng_state.bytes, agents, (uint64_t)seed, env.agents_per_buffer, 0);
#ifdef PUFFER_NETHACK
    metal_nethack_model_prepare(nh_train, stream);
    metal_nethack_model_prepare(nh_actor, stream);
    if (head_map.bytes) memcpy(head_map.bytes, head_host, (size_t)n_verbs * head_stride);
    float nethack_epsilon = metal_nethack_verb_epsilon(
        puf_ini_get(ini, "train", "verb_eps"), puf_ini_get(ini, "train", "verb_eps_anneal_start"),
        puf_ini_get(ini, "train", "verb_eps_anneal_end"), 0, steps);
#endif
    int64_t bytes = 0; for (auto &e : params.regs) { bytes = (bytes + 15) & ~int64_t(15); bytes += puf_numel(e.shape) * e.elem_size; }
    FloatTensor weights = fview((float*)params.mem, (int)(bytes / 4));
    Muon muon = {}; muon_init(&muon, &params, weights, lr, momentum, optalloc); wrap(optalloc);
    muon.lr_ptr = muon.lr_puf.data; muon.lr_derived_ptr = muon.lr_derived_puf.data; muon.ns.norm_ptr = muon.ns_norm_puf.data; *muon.lr_ptr = lr;
    struct RolloutSlot {
        FloatTensor obs, act, mask, logp, value, reward, terminal, initial;
    };
    RolloutSlot slots[2] = {};
    Allocator rollout_alloc[2];
    for (int i = 0; i < (async ? 2 : 1); ++i) {
        RolloutSlot &s = slots[i];
        s.obs = fview(nullptr, count, OBS_SIZE); s.act = fview(nullptr, count, NUM_ATNS);
        s.mask = fview(nullptr, count, action_dim); s.logp = fview(nullptr, count);
        s.value = fview(nullptr, count); s.reward = fview(nullptr, count);
        s.terminal = fview(nullptr, count); s.initial = fview(nullptr, layers, agents, hidden);
        alloc_register(&rollout_alloc[i], &s.obs); alloc_register(&rollout_alloc[i], &s.act);
        alloc_register(&rollout_alloc[i], &s.mask); alloc_register(&rollout_alloc[i], &s.logp);
        alloc_register(&rollout_alloc[i], &s.value); alloc_register(&rollout_alloc[i], &s.reward);
        alloc_register(&rollout_alloc[i], &s.terminal); alloc_register(&rollout_alloc[i], &s.initial);
        wrap(rollout_alloc[i]);
    }
    int result_fd = (int)puf_ini_get(ini, "base", "result_fd");
    MetalTrainResult result;
    std::vector<MetalTrainPoint> history;
    bool have_log_snapshot = false;
    auto started = std::chrono::steady_clock::now();
    bool profile = puf_ini_get(ini, "base", "profile") != 0;
    if (profile) mtl_enable_gpu_timing(true);
    const char *metric_name = puf_ini_get_str(ini, "sweep", "metric");
    std::string metric_key = metric_name && strchr(metric_name, '/') ? metric_name : std::string("env/") + (metric_name ? metric_name : "score");
    const char *run_id = puf_ini_get_str(ini, "base", "run_id");
    std::string run = run_id && strcmp(run_id, "None") ? run_id :
        std::to_string((long long)time(nullptr)) + "-" + std::to_string(getpid());
    std::string checkpoint_dir = std::string(puf_ini_get_str(ini, "base", "checkpoint_dir")) +
        "/" + PUFFER_ENV_NAME + "/" + run;
    make_dirs(checkpoint_dir);
    long checkpoint_interval = (long)puf_ini_get(ini, "base", "checkpoint_interval");
    MetalEvalPolicy frozen[8];
    std::vector<std::string> pool;
    std::string final_checkpoint;
    unsigned int pool_rng = (unsigned int)puf_ini_get(ini, "selfplay", "seed");
    long opponent_started[8] = {};
    long timeout = (long)puf_ini_get(ini, "selfplay", "opp_timeout_steps");
    if (selfplay) {
        mtl_ensure_stream_synced(stream);
        std::string initial_checkpoint = checkpoint_dir + "/0000000000000000.bin";
        checkpoint_io(params, initial_checkpoint.c_str(), true);
        pool.push_back(initial_checkpoint);
        for (int p = 1; p < policies; ++p) {
            int n = buffers * (env.policy_layout[p + 1] - env.policy_layout[p]);
            const std::string &opponent = pool[rand_r(&pool_rng) % pool.size()];
            metal_frozen_create(frozen[p - 1], encoder_kind, n, hidden, layers, horizon, action_dim, continuous, sizes, opponent,
                (uint64_t)seed, env.policy_layout[p + 1] - env.policy_layout[p], env.policy_layout[p]);
        }
    }
    auto snapshot_actor = [&]() {
        mtl_ensure_stream_synced(stream);
        for (size_t i = 0; i < params.regs.size(); ++i) {
            auto &src = params.regs[i]; auto &dst = actor_params.regs[i];
            memcpy(*dst.data_ptr, *src.data_ptr, (size_t)puf_numel(src.shape) * src.elem_size);
        }
    };
    auto rollout = [&](RolloutSlot &slot) {
        float *robs = slot.obs.data, *ract = slot.act.data, *rmask = slot.mask.data;
        float *rlp = slot.logp.data, *rv = slot.value.data, *rr = slot.reward.data;
        float *rt = slot.terminal.data, *initial = slot.initial.data;
        mtl_ensure_stream_synced(actor_stream);
        if (puf_ini_get(ini, "base", "reset_every_horizon")) memset(state.data, 0, (size_t)layers * agents * hidden * 4);
        if (selfplay && puf_ini_get(ini, "base", "reset_every_horizon")) {
            for (int p = 1; p < policies; ++p) if (frozen[p - 1].count)
                memset(frozen[p - 1].state.data, 0, (size_t)layers * frozen[p - 1].count * hidden * sizeof(float));
        }
        for (int t = 0; t < horizon; ++t) {
            if constexpr (std::is_same<obs_t, float>::value)
                memcpy(obs.data, env.observations, (size_t)agents * OBS_SIZE * sizeof(float));
            for (int a = 0; a < agents; ++a) {
                int i = a * horizon + t;
                if constexpr (std::is_same<obs_t, float>::value)
                    memcpy(robs + (size_t)i * OBS_SIZE, obs.data + (size_t)a * OBS_SIZE, OBS_SIZE * sizeof(float));
                else
                    for (int j = 0; j < OBS_SIZE; ++j) obs.data[a * OBS_SIZE + j] = robs[(size_t)i * OBS_SIZE + j] = (float)env.observations[a * OBS_SIZE + j];
                for (int j = 0; j < action_dim; ++j) mask.data[a * action_dim + j] = rmask[(size_t)i * action_dim + j] = env.action_mask[a * action_dim + j];
                rr[i] = std::max(-1.0f, std::min(1.0f, env.rewards[a])); rt[i] = env.terminals[a];
                if (env.terminals[a]) for (int l = 0; l < layers; ++l) memset(state.data + (l * agents + a) * hidden, 0, hidden * sizeof(float));
            }
            if (t == 0) {
                memcpy(initial, state.data, (size_t)layers * agents * hidden * sizeof(float));
                if (selfplay) for (int b = 0; b < buffers; ++b)
                    for (int a = env.policy_layout[1]; a < env.agents_per_buffer; ++a)
                        for (int l = 0; l < layers; ++l)
                            memset(initial + (l * agents + b * env.agents_per_buffer + a) * hidden, 0, hidden * sizeof(float));
            }
            MinGRUWeights *roll_nw = (async || nethack) ? &actor_nw : &nw;
#ifdef PUFFER_NETHACK
            PrecisionTensor x = metal_nethack_model_encode(nh_actor, obs, actor_stream);
            x = mingru_forward(roll_nw, x, state, &rn, actor_stream);
            x = metal_nethack_model_decode(nh_actor, x, actor_stream);
            mtl_nethack_sample_logits_dispatch_to(x, as, sampled.data, logp.data, vals.data,
                mask.data, action_dim, (MetalRngState*)rng_state.bytes, (const signed char*)head_map.bytes,
                head_stride, nethack_epsilon, actor_stream);
#else
            EncoderWeights *roll_ew = async ? &actor_ew : &ew;
            DecoderWeights *roll_dw = async ? &actor_dw : &dw;
            PrecisionTensor x = encoder_kind == METAL_LINEAR ? encoder_forward(roll_ew, &re, obs, actor_stream)
                : metal_custom_forward(async ? actor_cw : cw, cr, obs, actor_stream);
            x = mingru_forward(roll_nw, x, state, &rn, actor_stream);
            x = decoder_forward(roll_dw, &rd, x, actor_stream);
            if (continuous) puf5_sample_continuous(x, roll_dw->logstd, as, sampled.data, logp.data, vals.data, mask.data, (MetalRngState*)rng_state.bytes, actor_stream);
            else mtl_sample_logits_dispatch_to(x, as, sampled.data, logp.data, vals.data, mask.data, action_dim, (MetalRngState*)rng_state.bytes, actor_stream);
#endif
            if (selfplay) for (int p = 1; p < policies; ++p)
#ifdef PUFFER_NETHACK
                metal_frozen_step(frozen[p - 1], encoder_kind, env, p, layers, hidden, action_dim, continuous, actor_stream, nethack_epsilon);
#else
                metal_frozen_step(frozen[p - 1], encoder_kind, env, p, layers, hidden, action_dim, continuous, actor_stream);
#endif
            mtl_ensure_stream_synced(actor_stream);
            for (int a = 0; a < agents; ++a) {
                int i = a * horizon + t; rlp[i] = logp.data[a]; rv[i] = vals.data[a];
                for (int j = 0; j < NUM_ATNS; ++j) env.actions[a * NUM_ATNS + j] = ract[(size_t)i * NUM_ATNS + j] = sampled.data[a * NUM_ATNS + j];
            }
            if (selfplay) for (int p = 1; p < policies; ++p) {
                MetalEvalPolicy &pol = frozen[p - 1];
                int per = env.policy_layout[p + 1] - env.policy_layout[p];
                for (int b = 0; b < buffers; ++b) for (int a = 0; a < per; ++a) {
                    int dst = b * env.agents_per_buffer + env.policy_layout[p] + a, src = b * per + a;
                    int i = dst * horizon + t;
                    rlp[i] = pol.logp.data[src]; rv[i] = pol.values.data[src];
                    for (int j = 0; j < NUM_ATNS; ++j)
                        env.actions[dst * NUM_ATNS + j] = ract[(size_t)i * NUM_ATNS + j] = pol.actions.data[src * NUM_ATNS + j];
                }
            }
            for (int b = 0; b < buffers; ++b) metal_env_step(&env, b);
        }
    };
    if (async && epochs > 0) { snapshot_actor(); rollout(slots[0]); }
    // A completed update with no internal stream synchronization establishes
    // the fixed constant-ring footprint for this model and minibatch shape.
    NSUInteger update_ring_bytes = 0;
    for (long epoch = 0; epoch < epochs; ++epoch) {
        int ready = async ? (int)(epoch % 2) : 0;
        std::exception_ptr actor_error;
        struct ActorThread {
            std::thread thread;
            ~ActorThread() { if (thread.joinable()) thread.join(); }
        } actor;
        if (async) {
            if (epoch + 1 < epochs) {
#ifdef PUFFER_NETHACK
                nethack_epsilon = metal_nethack_verb_epsilon(
                    puf_ini_get(ini, "train", "verb_eps"), puf_ini_get(ini, "train", "verb_eps_anneal_start"),
                    puf_ini_get(ini, "train", "verb_eps_anneal_end"), epoch * (long)count, steps);
#endif
                snapshot_actor();
                actor.thread = std::thread([&, ready] {
                    @autoreleasepool {
                        try { rollout(slots[1 - ready]); }
                        catch (...) { actor_error = std::current_exception(); }
                    }
                });
            }
        } else {
#ifdef PUFFER_NETHACK
            nethack_epsilon = metal_nethack_verb_epsilon(
                puf_ini_get(ini, "train", "verb_eps"), puf_ini_get(ini, "train", "verb_eps_anneal_start"),
                puf_ini_get(ini, "train", "verb_eps_anneal_end"), epoch * (long)count, steps);
            snapshot_actor();
#endif
            rollout(slots[ready]);
        }
        RolloutSlot &slot = slots[ready];
        float *robs = slot.obs.data, *ract = slot.act.data, *rmask = slot.mask.data;
        float *rlp = slot.logp.data, *rv = slot.value.data, *rr = slot.reward.data;
        float *rt = slot.terminal.data, *initial = slot.initial.data;
        mtl_ensure_stream_synced(stream);
        if (puf_ini_get(ini, "train", "anneal_lr"))
            *muon.lr_ptr = metal_cosine_annealing(lr, lr * min_lr_ratio, epoch, epochs);
        float current_ent = ent;
        if (puf_ini_get(ini, "train", "anneal_ent_coef"))
            current_ent = metal_cosine_annealing(ent, ent * min_ent_ratio, epoch, epochs);
        double pg_sum = 0.0, vf_sum = 0.0, entropy_sum = 0.0, total_sum = 0.0;
        double old_kl_sum = 0.0, kl_sum = 0.0, clipfrac_sum = 0.0, ratio_sum = 0.0;
        long metric_count = 0, nonfinite = 0;
        MetalStream *train_ms = mtl_resolve_stream(stream);
        Puf5MetricsPartial *partials = (Puf5MetricsPartial*)metric_partials.bytes;
        for (int mb = 0; mb < updates; ++mb) {
            if (!update_ring_bytes ||
                    train_ms->const_ring_offset > MTL_CONST_RING_SIZE - update_ring_bytes)
                mtl_ensure_stream_synced(stream);
            uint64_t sync_before = train_ms->sync_event_value;
            NSUInteger ring_before = train_ms->const_ring_offset;
            int start = (mb * segments) % agents, base = start * horizon;
            mtl_barrier(train_ms);
            mtl_copy_f32(trainobs.data, robs + (size_t)base * OBS_SIZE, mbsize * OBS_SIZE, stream);
            mtl_copy_f32(mba.data, ract + (size_t)base * NUM_ATNS, mbsize * NUM_ATNS, stream);
            mtl_copy_f32(mbmask.data, rmask + (size_t)base * action_dim, mbsize * action_dim, stream);
            mtl_copy_f32(mblp.data, rlp + base, mbsize, stream);
            mtl_copy_f32(mbv.data, rv + base, mbsize, stream);
            mtl_copy_f32(mbr.data, rr + base, mbsize, stream);
            mtl_copy_f32(mbd.data, rt + base, mbsize, stream);
            if (fp16_mingru) {
                for (int l = 0; l < layers; ++l)
                    mtl_cast_f32_to_f16(trainstate_h.data + l * segments * hidden,
                        initial + (l * agents + start) * hidden, segments * hidden, stream);
                mingru_sync_weights_lo(&nw, stream);
            } else {
                for (int l = 0; l < layers; ++l)
                    mtl_copy_f32(trainstate.data + l * segments * hidden,
                        initial + (l * agents + start) * hidden, segments * hidden, stream);
            }
            mtl_barrier(train_ms);
            PrecisionTensor flatobs = pview(trainobs.data, mbsize, OBS_SIZE);
#ifdef PUFFER_NETHACK
            PrecisionTensor x = metal_nethack_model_encode(nh_train, flatobs, stream);
#else
            PrecisionTensor x = encoder_kind == METAL_LINEAR ? encoder_forward(&ew, &te, flatobs, stream)
                : metal_custom_forward(cw, ct, flatobs, stream);
#endif
            x = pview(x.data, segments, horizon, hidden);
            if (fp16_mingru) {
                mtl_cast_f32_to_f16(x_h.data, x.data, mbsize * hidden, stream);
                x = pview(x_h.data, segments, horizon, hidden);
                x.dtype_size = 2;
            }
            x = mingru_forward_train(&nw, x, fp16_mingru ? trainstate_h : trainstate, mbd, &tn, stream);
            x = pview(x.data, mbsize, hidden);
            if (fp16_mingru) {
                mtl_barrier(mtl_resolve_stream(stream));
                mtl_cast_f16_to_f32(y_f.data, x.data, mbsize * hidden, stream);
                x = pview(y_f.data, mbsize, hidden);
            }
#ifdef PUFFER_NETHACK
            x = metal_nethack_model_decode(nh_train, x, stream);
#else
            x = decoder_forward(&dw, &td, x, stream);
#endif
            Puf5PPOParams pp = {mbsize, NUM_ATNS, action_dim, continuous, clip, vfclip, vfcoef, current_ent};
#ifdef PUFFER_NETHACK
            puf5_nethack_cache_policy(x.data, nullptr, mba.data, mblp.data, mbmask.data, as.data,
                livev.data, imp.data, pp, (const signed char*)head_map.bytes, head_stride, nethack_epsilon, stream);
#else
            puf5_cache_policy(x.data, continuous ? dw.logstd.data : nullptr, mba.data, mblp.data, mbmask.data, as.data, livev.data, imp.data, pp, stream);
#endif
            mtl_barrier(mtl_resolve_stream(stream));
            puf5_advantage(livev, mbr, mbd, vtrace ? &imp : nullptr, adv, livev, gamma, lambda, rho, c, stream);
            mtl_barrier(mtl_resolve_stream(stream));
#ifdef PUFFER_NETHACK
            puf5_nethack_ppo_grad(x.data, nullptr, mba.data, mblp.data, mbmask.data, as.data,
                mbv.data, adv.data, livev.data, gl.data, nullptr, gv.data, ppo_entropy.data, ppo_logratio.data, pp,
                (const signed char*)head_map.bytes, head_stride, nethack_epsilon, stream);
#else
            puf5_ppo_grad(x.data, continuous ? dw.logstd.data : nullptr, mba.data, mblp.data, mbmask.data, as.data, mbv.data, adv.data, livev.data, gl.data, continuous ? gs.data : nullptr, gv.data, ppo_entropy.data, ppo_logratio.data, pp, stream);
#endif
            mtl_barrier(mtl_resolve_stream(stream));
            puf5_reduce_metrics(partials + mb, x.data, mbv.data, adv.data, livev.data,
                imp.data, ppo_entropy.data, ppo_logratio.data, pp, stream);
            mtl_barrier(train_ms);
#ifdef PUFFER_NETHACK
            PrecisionTensor dh = metal_nethack_model_backward_decoder(nh_train, gl, gv, mbsize, stream);
#else
            PrecisionTensor dh = decoder_backward(&dw, &td, gl, continuous ? gs : FloatTensor{}, gv, stream);
#endif
            dh = pview(dh.data, segments, horizon, hidden);
            if (fp16_mingru) {
                mtl_barrier(mtl_resolve_stream(stream));
                mtl_cast_f32_to_f16(dh_h.data, dh.data, mbsize * hidden, stream);
                dh = pview(dh_h.data, segments, horizon, hidden);
                dh.dtype_size = 2;
            }
            dh = mingru_backward(&nw, dh, &tn, stream);
            if (fp16_mingru) {
                mtl_barrier(mtl_resolve_stream(stream));
                mtl_cast_f16_to_f32(gi_f.data, dh.data, mbsize * hidden, stream);
                dh = pview(gi_f.data, segments, horizon, hidden);
            }
#ifdef PUFFER_NETHACK
            metal_nethack_model_backward_encoder(nh_train, pview(dh.data, mbsize, hidden), stream);
#else
            if (encoder_kind == METAL_LINEAR) encoder_backward(&ew, &te, dh, stream);
            else metal_custom_backward(cw, ct, pview(dh.data, mbsize, hidden), stream);
#endif
            mtl_barrier(mtl_resolve_stream(stream));
            mtl_copy_f32(muon.gc_puf.data, (float*)grads.mem, (int)weights.shape[0], stream);
            mtl_barrier(mtl_resolve_stream(stream));
            clip_grad_norm_f32(muon.gc_puf, norm.data, max_norm, 1e-6f, stream);
            mtl_barrier(train_ms); muon_step(&muon, stream);
            if (!update_ring_bytes && train_ms->sync_event_value == sync_before)
                update_ring_bytes = train_ms->const_ring_offset - ring_before;
        }
        mtl_ensure_stream_synced(stream);
        for (int mb = 0; mb < updates; ++mb) {
            const Puf5MetricsPartial &p = partials[mb];
            pg_sum += p.policy; vf_sum += p.value; entropy_sum += p.entropy; total_sum += p.total;
            old_kl_sum += p.old_kl; kl_sum += p.kl; clipfrac_sum += p.clipfrac;
            ratio_sum += p.importance; nonfinite += p.nonfinite; metric_count += mbsize;
        }
        if (actor.thread.joinable()) actor.thread.join();
        if (actor_error) std::rethrow_exception(actor_error);
        Dict logs = {}; metal_env_log(&env, &logs, 1);
        double inv_metric_count = metric_count ? 1.0 / metric_count : 0.0;
        dict_set(&logs, "loss/policy", pg_sum * inv_metric_count);
        dict_set(&logs, "loss/value", vf_sum * inv_metric_count);
        dict_set(&logs, "loss/entropy", entropy_sum * inv_metric_count);
        dict_set(&logs, "loss/total", total_sum * inv_metric_count);
        dict_set(&logs, "loss/old_kl", old_kl_sum * inv_metric_count);
        dict_set(&logs, "loss/kl", kl_sum * inv_metric_count);
        dict_set(&logs, "loss/clipfrac", clipfrac_sum * inv_metric_count);
        dict_set(&logs, "importance", ratio_sum * inv_metric_count);
        printf("epoch %ld/%ld steps %ld", epoch + 1, epochs, (epoch + 1) * (long)count);
        if (metric_count) printf(" loss_pg=%.6g loss_vf=%.6g ratio=%.6g nonfinite=%ld",
            pg_sum / metric_count, vf_sum / metric_count, ratio_sum / metric_count, nonfinite);
        for (int i = 0; i < logs.size; ++i) printf(" %s=%.6g", logs.items[i].key, logs.items[i].value);
        if (profile) {
            int syncs; double sync_ms, gpu_ms, sched_ms;
            mtl_sync_stats(&syncs, &sync_ms);
            mtl_gpu_timing_stats(&gpu_ms, &sched_ms);
            printf(" gpu/syncs=%d gpu/sync_ms=%.3f gpu/exec_ms=%.3f gpu/sched_ms=%.3f",
                syncs, sync_ms, gpu_ms, sched_ms);
        }
        printf("\n");
        result.cost = (float)std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        result.steps = (float)((epoch + 1) * (long)count);
        bool episodes = dict_get(&logs, "env/n") > 0;
        DictItem *metric = dict_find(&logs, metric_key.c_str());
        if (episodes || !have_log_snapshot) {
            result.score = metric ? (float)metric->value : 0.0f;
            have_log_snapshot = true;
        }
        if (episodes && metric)
            history.push_back({metric->value, result.cost, result.steps});
        dict_clear(&logs);
        if (epoch + 1 == epochs || (checkpoint_interval > 0 && (epoch + 1) % checkpoint_interval == 0)) {
            char filename[64]; snprintf(filename, sizeof(filename), "/%016ld.bin", (epoch + 1) * (long)count);
            std::string path = checkpoint_dir + filename;
            std::string tmp = path + ".tmp";
            checkpoint_io(params, tmp.c_str(), true);
            if (rename(tmp.c_str(), path.c_str())) { perror(path.c_str()); exit(1); }
            final_checkpoint = path;
            if (selfplay) {
                if ((int)pool.size() == pool_max) pool.erase(pool.begin());
                pool.push_back(path);
            }
        }
        if (selfplay && timeout > 0) {
            long current = (epoch + 1) * (long)count;
            for (int p = 1; p < policies; ++p) if (current - opponent_started[p - 1] >= timeout) {
                checkpoint_io(frozen[p - 1].params, pool[rand_r(&pool_rng) % pool.size()].c_str(), false);
                opponent_started[p - 1] = current;
            }
        }
    }
    mtl_ensure_stream_synced(stream); metal_env_close(&env); mtl_destroy();
#ifdef PUFFER_NETHACK
    metal_nethack_model_destroy(nh_train); metal_nethack_model_destroy(nh_actor);
#endif
    for (int p = 1; p < policies; ++p) {
#ifdef PUFFER_NETHACK
        metal_nethack_model_destroy(frozen[p - 1].nh);
#endif
        frozen[p - 1].acts.destroy(); frozen[p - 1].params.destroy();
    }
    rollout_alloc[0].destroy(); rollout_alloc[1].destroy();
    optalloc.destroy(); grads.destroy(); acts.destroy(); actor_params.destroy(); params.destroy();
    if (selfplay && !final_checkpoint.empty()) {
        long bot_games = (long)puf_ini_get(ini, "selfplay", "eval_bot_games");
        long pool_games = (long)puf_ini_get(ini, "selfplay", "eval_games");
        int max_opponents = (int)puf_ini_get(ini, "selfplay", "eval_pool_size");
        puf_ini_put(ini, "base.load_model_path", final_checkpoint.c_str());
        if (bot_games > 0) {
            double ladder[16];
            int rungs = puf_ini_get_list(ini, "selfplay", "eval_bots", ladder, 16);
            if (rungs <= 0) { fprintf(stderr, "selfplay.eval_bot_games requires selfplay.eval_bots\n"); exit(1); }
            puf_ini_put(ini, "env.num_agents", "1");
            puf_ini_put(ini, "env.num_bots", "1");
            puf_ini_put(ini, "selfplay.enabled", "0");
            Dict *overrides = puf_ini_section(ini, "bot_eval", 1);
            for (int i = 0; i < overrides->size; ++i) {
                std::string key = std::string("env.") + overrides->items[i].key;
                puf_ini_put(ini, key.c_str(), overrides->items[i].str);
            }
            char value[32];
            snprintf(value, sizeof(value), "%ld", (long)puf_ini_get(ini, "selfplay", "eval_bot_envs"));
            puf_ini_put(ini, "vec.total_agents", value);
            long threads = (long)puf_ini_get(ini, "selfplay", "eval_bot_threads");
            if (threads > 0) { snprintf(value, sizeof(value), "%ld", threads); puf_ini_put(ini, "vec.num_threads", value); }
            snprintf(value, sizeof(value), "%ld", bot_games); puf_ini_put(ini, "base.eval_episodes", value);
            MetalEvalResult r = eval_metal(ini, false, false, (int)ladder[0], false, ladder, rungs);
            printf("bot_eval mean_perf=%.4f\n", r.perf);
            result.score = (float)r.perf;
        } else if (pool_games > 0 && max_opponents > 0) {
            char value[32]; snprintf(value, sizeof(value), "%ld", pool_games);
            puf_ini_put(ini, "base.eval_episodes", value);
            double sum = 0; int evaluated = 0;
            for (const std::string &opponent : pool) {
                if (opponent == final_checkpoint) continue;
                if (evaluated >= max_opponents) break;
                puf_ini_put(ini, "base.load_enemy_model_path", opponent.c_str());
                MetalEvalResult r = eval_metal(ini, true, false, -1, false);
                printf("selfplay_eval vs %s games=%ld score=%.4f draw=%.4f\n", opponent.c_str(), r.games, r.score, r.draw);
                sum += r.score; ++evaluated;
            }
            if (evaluated) {
                result.score = (float)(sum / evaluated);
                printf("selfplay_eval mean_score=%.4f n=%d\n", sum / evaluated, evaluated);
            }
        }
    }
    if (!final_checkpoint.empty() && (!selfplay ||
            ((puf_ini_get(ini, "selfplay", "eval_games") <= 0 ||
              puf_ini_get(ini, "selfplay", "eval_pool_size") <= 0) &&
             puf_ini_get(ini, "selfplay", "eval_bot_games") <= 0)) &&
            puf_ini_get(ini, "base", "eval_episodes") > 0) {
        puf_ini_put(ini, "base.load_model_path", final_checkpoint.c_str());
        MetalEvalResult evaluation = eval_metal(ini, false, false, -1, false);
        result.score = (float)evaluation.score;
    }
    if (result_fd > 0) {
        int points = selfplay ? 1 : (int)puf_ini_get(ini, "sweep", "downsample");
        if (points < 1 || points > 64) { fprintf(stderr, "sweep.downsample must be in [1,64]\n"); exit(1); }
        result.points = points;
        if (points > 1 && !history.empty()) {
            double sums[3] = {}, fallback[3] = {history[0].score, history[0].cost, history[0].steps};
            int out = 0, n = 0;
            double final_steps = history.back().steps, next = final_steps / (points - 1);
            for (const auto &item : history) {
                sums[0] += item.score; sums[1] += item.cost; sums[2] += item.steps; ++n;
                if (item.steps < next || out >= points - 1) continue;
                for (int k = 0; k < 3; ++k) { fallback[k] = sums[k] / n; sums[k] = 0; }
                result.scores[out] = fallback[0]; result.costs[out] = fallback[1]; result.step_points[out++] = fallback[2];
                n = 0; next += final_steps / (points - 1);
            }
            while (out < points - 1) {
                result.scores[out] = fallback[0]; result.costs[out] = fallback[1]; result.step_points[out++] = fallback[2];
            }
        }
        result.scores[points - 1] = result.score;
        result.costs[points - 1] = result.cost;
        result.step_points[points - 1] = result.steps;
        const char *data = (const char*)&result;
        size_t remaining = sizeof(result);
        while (remaining) {
            ssize_t n = write(result_fd, data, remaining);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { perror("sweep result"); exit(1); }
            data += n; remaining -= (size_t)n;
        }
        close(result_fd);
    }
}
#include "metal_sweep.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s train|eval|match|sweep [MODEL.bin] [--headless] [--section.key=value ...]\n", argv[0]);
        return 1;
    }
    const char *mode = argv[1], *model = nullptr;
    int argi = 2;
    if (!strcmp(mode, "eval") && argi < argc && argv[argi][0] != '-' && !strchr(argv[argi], '='))
        model = argv[argi++];
    bool headless = false;
    std::vector<char*> args;
    for (int i = argi; i < argc; ++i) {
        if (!strcmp(argv[i], "--headless")) headless = true;
        else args.push_back(argv[i]);
    }
    Ini ini = {}; puf_ini_load_env(&ini, PUFFER_ENV_NAME, (int)args.size(), args.data());
    if (model) puf_ini_put(&ini, "base.load_model_path", model);
    if (!strcmp(mode, "train")) train_metal(&ini);
    else if (!strcmp(mode, "sweep")) sweep_metal(&ini, argv[0]);
    else if (!strcmp(mode, "eval") || !strcmp(mode, "match")) {
        eval_metal(&ini, !strcmp(mode, "match"), !headless);
    } else {
        fprintf(stderr, "Metal mode '%s' is not implemented\n", mode);
        puf_ini_free(&ini); return 1;
    }
    puf_ini_free(&ini); return 0;
}
