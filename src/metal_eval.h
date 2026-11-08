// Headless evaluation and two-policy match, using the same environment layout as training.
struct MetalEvalPolicy {
#ifdef PUFFER_NETHACK
    MetalNethackModel nh;
    PufTensor head_map = {};
    int head_stride = 0;
#endif
    MetalCustomWeights custom;
    MetalCustomActivations custom_acts;
    EncoderWeights encoder = {};
    DecoderWeights decoder = {};
    MinGRUWeights network = {};
    EncoderActivations ea = {};
    DecoderActivations da = {};
    MinGRUActivations na = {};
    Allocator params, acts;
    PrecisionTensor obs = {}, state = {}, eval_logits = {};
    FloatTensor actions = {}, logp = {}, values = {}, mask = {};
    PufTensor rng_state = {};
    IntTensor sizes = {};
    int count = 0;
};

static std::string metal_checkpoint_path(Ini *ini, const char *key) {
    const char *path = puf_ini_get_str(ini, "base", key);
    if (!path || !strcmp(path, "None")) return {};
    if (strcmp(path, "latest")) return path;
    std::string best;
    time_t when = 0;
    find_latest(std::string(puf_ini_get_str(ini, "base", "checkpoint_dir")) + "/" + PUFFER_ENV_NAME, best, when);
    if (best.empty()) { fprintf(stderr, "no checkpoint found for %s\n", PUFFER_ENV_NAME); exit(1); }
    return best;
}

struct MetalEvalResult { double score = 0, perf = 0, draw = 0; long games = 0; };
static MetalEvalResult eval_metal(Ini *ini, bool match, bool render, int bot_policy = -1,
        bool verbose = true, const double *ladder = nullptr, int rungs = 0) {
#if defined(METAL_GPU_BREAKOUT) || defined(METAL_GPU_ADMIRAL) || defined(METAL_GPU_BROTATO)
    if (render || match || bot_policy >= 0) {
        fprintf(stderr, "Metal GPU environments do not support rendering, matches, or bot evaluation\n"); exit(1);
    }
#endif
    long episodes = (long)puf_ini_get(ini, "base", "eval_episodes");
    int agents = (int)puf_ini_get(ini, "base", "eval_agents");
    if (agents == -1) agents = (int)puf_ini_get(ini, "vec", "total_agents");
    int buffers = (int)puf_ini_get(ini, "vec", "num_buffers");
    int hidden = (int)puf_ini_get(ini, "policy", "hidden_size");
    int layers = (int)puf_ini_get(ini, "policy", "num_layers");
    int horizon = (int)puf_ini_get(ini, "train", "horizon");
    int seed = (int)puf_ini_get(ini, "base", "seed");
    if (render) {
        Env probe = {};
        puf_init(&probe, puf_ini_section(ini, "env", 0));
        agents = probe.num_agents;
        puf_close(&probe);
        buffers = 1; horizon = 1;
        char value[32]; snprintf(value, sizeof(value), "%d", agents);
        puf_ini_put(ini, "vec.total_agents", value);
        puf_ini_put(ini, "vec.num_buffers", "1");
        puf_ini_put(ini, "vec.num_threads", "1");
    }
    if (!render && puf_ini_get(ini, "base", "eval_agents") != -1) {
        char value[32]; snprintf(value, sizeof(value), "%d", agents);
        puf_ini_put(ini, "vec.total_agents", value);
    }
    if ((!render && episodes <= 0) || agents <= 0 || buffers <= 0 || agents % buffers || hidden <= 0 || layers <= 0 || horizon <= 0) {
        fprintf(stderr, "invalid Metal evaluation dimensions or episode count\n"); exit(1);
    }
    puf_ini_put(ini, "vec.num_policies", match ? "2" : "1");
    puf_ini_put(ini, "vec.hist_policy_percent", match ? "1" : "0");
    std::string paths[2] = {metal_checkpoint_path(ini, "load_model_path"), match ? metal_checkpoint_path(ini, "load_enemy_model_path") : std::string()};
    if (match && (paths[0].empty() || paths[1].empty())) {
        fprintf(stderr, "match requires base.load_model_path and base.load_enemy_model_path\n"); exit(1);
    }
    int sizes[] = ACT_SIZES, action_dim = 0, continuous = 1, any_continuous = 0;
    for (int x : sizes) { action_dim += x; continuous &= x == 1; any_continuous |= x == 1; }
    if (any_continuous && !continuous) { fprintf(stderr, "mixed action spaces are unsupported\n"); exit(1); }
    bool greedy = !continuous && puf_ini_get(ini, "base", "eval_deterministic") != 0;
#ifdef PUFFER_NETHACK
    if (greedy) {
        fprintf(stderr, "eval_deterministic does not support conditional NetHack heads\n");
        exit(1);
    }
#endif
    MetalEnvVec env = {};
    int np = match ? 2 : 1;
    metal_env_create(&env, agents, buffers, np, puf_ini_section(ini, "vec", 0), puf_ini_section(ini, "env", 0));
    if (bot_policy >= 0)
        for (int i = 0; i < env.size; ++i) puf_set_bot_policy(&env.envs[i], bot_policy);
    MetalEncoderKind encoder_kind = metal_encoder_kind();
    metal_env_reset(&env);
    mtl_init(); puf_set_gpu_training(true);
    cudaStream_t stream = mtl_stream();
    MetalEvalPolicy policies[2];
    long nparams = 0;
    for (int p = 0; p < np; ++p) {
        MetalEvalPolicy &pol = policies[p];
        pol.count = (env.policy_layout[p + 1] - env.policy_layout[p]) * buffers;
        pol.encoder.in_dim = OBS_SIZE; pol.encoder.out_dim = hidden;
        pol.decoder.hidden_dim = hidden; pol.decoder.output_dim = action_dim; pol.decoder.continuous = continuous;
        pol.network.hidden = hidden; pol.network.num_layers = layers; pol.network.horizon = horizon; pol.network.weights.resize(layers);
#ifdef PUFFER_NETHACK
        metal_nethack_model_register(pol.nh, std::max(1, pol.count), hidden,
            pol.params, pol.nh.unused_gradients, pol.acts, false);
#else
        if (encoder_kind == METAL_LINEAR) encoder_reg_params(&pol.encoder, &pol.params, 4);
        else metal_custom_params(pol.custom, &pol.params, encoder_kind, hidden);
        decoder_reg_params(&pol.decoder, &pol.params, 4);
#endif
        mingru_reg_params(&pol.network, &pol.params, 4);
        size_t raw = 0; for (auto &e : pol.params.regs) raw += puf_numel(e.shape);
        if (!p) nparams = (long)raw;
        wrap(pol.params);
        uint64_t initseed = (uint64_t)seed;
#ifdef PUFFER_NETHACK
        metal_nethack_model_init(pol.nh, &initseed, stream);
#else
        if (encoder_kind == METAL_LINEAR) encoder_init_weights(&pol.encoder, &initseed, stream);
        else metal_custom_init(pol.custom, &initseed, stream);
        decoder_init_weights(&pol.decoder, &initseed, stream);
#endif
        mingru_init_weights(&pol.network, &initseed, stream);
        if (!paths[p].empty()) { mtl_ensure_stream_synced(stream); checkpoint_io(pol.params, paths[p].c_str(), false); }
        if (!pol.count) continue;
#ifndef PUFFER_NETHACK
        if (encoder_kind == METAL_LINEAR) encoder_reg_rollout(&pol.encoder, &pol.ea, &pol.acts, pol.count);
        else metal_custom_activations(pol.custom, pol.custom_acts, &pol.acts, nullptr, pol.count);
        decoder_reg_rollout(&pol.decoder, &pol.da, &pol.acts, pol.count);
#endif
        mingru_reg_rollout(&pol.network, &pol.na, &pol.acts, pol.count);
        pol.obs = pview(nullptr, pol.count, OBS_SIZE); pol.state = pview(nullptr, layers, pol.count, hidden);
        pol.actions = fview(nullptr, pol.count, NUM_ATNS); pol.logp = fview(nullptr, pol.count);
        pol.values = fview(nullptr, pol.count); pol.mask = fview(nullptr, pol.count, action_dim); pol.sizes.shape[0] = NUM_ATNS;
        alloc_register(&pol.acts, &pol.obs); alloc_register(&pol.acts, &pol.state); alloc_register(&pol.acts, &pol.actions);
        alloc_register(&pol.acts, &pol.logp); alloc_register(&pol.acts, &pol.values); alloc_register(&pol.acts, &pol.mask); alloc_register(&pol.acts, &pol.sizes);
        pol.rng_state.shape[0] = (int64_t)pol.count * sizeof(MetalRngState);
        pol.rng_state.dtype_size = 1;
        alloc_register(&pol.acts, &pol.rng_state);
#ifdef PUFFER_NETHACK
        int verbs = 0; const signed char *map = env_head_consume_map(&verbs, &pol.head_stride);
        if (map && verbs > 0 && pol.head_stride > 0) {
            pol.head_map.shape[0] = verbs * pol.head_stride; pol.head_map.dtype_size = 1;
            alloc_register(&pol.acts, &pol.head_map);
        }
#endif
        wrap(pol.acts); memcpy(pol.sizes.data, sizes, sizeof(sizes));
        mtl_rng_init((MetalRngState*)pol.rng_state.bytes, pol.count, (uint64_t)seed,
            env.policy_layout[p + 1] - env.policy_layout[p], env.policy_layout[p]);
#ifdef PUFFER_NETHACK
        metal_nethack_model_prepare(pol.nh, stream);
        if (pol.head_map.bytes) memcpy(pol.head_map.bytes, map, (size_t)verbs * pol.head_stride);
#endif
    }
    MetalEvalResult result;
    int rung = 0; double rung_sum = 0;
    Dict wipe = {}; metal_env_log(&env, &wipe, 1); dict_clear(&wipe);
    while (true) {
        if (render) {
            puf_render(env.envs);
            if (WindowShouldClose()) break;
        }
        for (int t = 0; t < horizon; ++t) {
            mtl_ensure_stream_synced(stream);
            for (int p = 0; p < np; ++p) {
                MetalEvalPolicy &pol = policies[p]; if (!pol.count) continue;
                int per = env.policy_layout[p + 1] - env.policy_layout[p];
                for (int b = 0; b < buffers; ++b) for (int a = 0; a < per; ++a) {
                    int src = b * env.agents_per_buffer + env.policy_layout[p] + a, dst = b * per + a;
                    for (int j = 0; j < OBS_SIZE; ++j) pol.obs.data[dst * OBS_SIZE + j] = (float)env.observations[src * OBS_SIZE + j];
                    for (int j = 0; j < action_dim; ++j) pol.mask.data[dst * action_dim + j] = env.action_mask[src * action_dim + j];
                    if (env.terminals[src]) for (int l = 0; l < layers; ++l) memset(pol.state.data + (l * pol.count + dst) * hidden, 0, hidden * sizeof(float));
                }
#ifdef PUFFER_NETHACK
                PrecisionTensor x = metal_nethack_model_encode(pol.nh, pol.obs, stream);
                x = mingru_forward(&pol.network, x, pol.state, &pol.na, stream);
                x = metal_nethack_model_decode(pol.nh, x, stream);
                mtl_nethack_sample_logits_dispatch_to(x, pol.sizes, pol.actions.data, pol.logp.data,
                    pol.values.data, pol.mask.data, action_dim, (MetalRngState*)pol.rng_state.bytes,
                    (const signed char*)pol.head_map.bytes, pol.head_stride, render ? 1.0f : 0.0f, stream);
#else
                PrecisionTensor x = encoder_kind == METAL_LINEAR ? encoder_forward(&pol.encoder, &pol.ea, pol.obs, stream)
                    : metal_custom_forward(pol.custom, pol.custom_acts, pol.obs, stream);
                x = mingru_forward(&pol.network, x, pol.state, &pol.na, stream);
                x = decoder_forward(&pol.decoder, &pol.da, x, stream);
                if (continuous) puf5_sample_continuous(x, pol.decoder.logstd, pol.sizes, pol.actions.data, pol.logp.data, pol.values.data, pol.mask.data, (MetalRngState*)pol.rng_state.bytes, stream);
                else if (greedy) pol.eval_logits = x;
                else mtl_sample_logits_dispatch_to(x, pol.sizes, pol.actions.data, pol.logp.data, pol.values.data, pol.mask.data, action_dim, (MetalRngState*)pol.rng_state.bytes, stream);
#endif
            }
            mtl_ensure_stream_synced(stream);
            for (int p = 0; p < np; ++p) {
                MetalEvalPolicy &pol = policies[p]; int per = env.policy_layout[p + 1] - env.policy_layout[p];
                for (int b = 0; b < buffers; ++b) for (int a = 0; a < per; ++a) {
                    int dst = b * env.agents_per_buffer + env.policy_layout[p] + a, src = b * per + a;
                    if (greedy) {
                        const float* logits = pol.eval_logits.data + src * pol.eval_logits.shape[1];
                        const float* legal = pol.mask.data + src * action_dim;
                        int offset = 0;
                        for (int h = 0; h < NUM_ATNS; ++h) {
                            int best = 0;
                            float score = -INFINITY;
                            for (int j = 0; j < sizes[h]; ++j) {
                                if (legal[offset + j] != 0.0f && logits[offset + j] > score) {
                                    score = logits[offset + j];
                                    best = j;
                                }
                            }
                            env.actions[dst * NUM_ATNS + h] = (float)best;
                            offset += sizes[h];
                        }
                    } else {
                        memcpy(env.actions + dst * NUM_ATNS, pol.actions.data + src * NUM_ATNS, NUM_ATNS * sizeof(float));
                    }
                }
            }
            for (int b = 0; b < buffers; ++b) metal_env_step(&env, b);
        }
        if (render) continue;
        Dict logs = {}; metal_env_log(&env, &logs, 0);
        long games = (long)dict_get(&logs, "env/n");
        if (games >= episodes) {
            const char *metric = puf_ini_get_str(ini, "sweep", "metric");
            std::string key = metric && strchr(metric, '/') ? metric : std::string("env/") + (metric ? metric : "score");
            DictItem *item = dict_find(&logs, key.c_str());
            result.score = match ? dict_get(&logs, "env/policy_0_score") : item ? item->value : dict_get(&logs, "env/score");
            result.perf = dict_get(&logs, "env/perf");
            result.draw = match ? dict_get(&logs, "env/draw_rate") : 0;
            result.games = games;
            if (verbose) {
                printf("METAL_EVAL env=%s score=%.6f perf=%.6f games=%ld params=%ld", PUFFER_ENV_NAME, result.score, result.perf, games, nparams);
                if (match) printf(" draw=%.6f", result.draw);
                printf("\n");
            }
            dict_clear(&logs);
            if (ladder && rungs > 0) {
                rung_sum += result.perf;
                printf("bot_eval policy=%d games=%ld perf=%.4f\n", (int)ladder[rung], result.games, result.perf);
                if (++rung < rungs) {
                    for (int i = 0; i < env.size; ++i) puf_set_bot_policy(&env.envs[i], (int)ladder[rung]);
                    Dict cleared = {}; metal_env_log(&env, &cleared, 1); dict_clear(&cleared);
                    metal_env_reset(&env);
                    for (int p = 0; p < np; ++p) if (policies[p].count)
                        memset(policies[p].state.data, 0, (size_t)layers * policies[p].count * hidden * sizeof(float));
                    continue;
                }
                result.perf = rung_sum / rungs;
            }
            break;
        }
        dict_clear(&logs);
    }
    mtl_ensure_stream_synced(stream); metal_env_close(&env); mtl_destroy();
    for (int p = 0; p < np; ++p) {
#ifdef PUFFER_NETHACK
        metal_nethack_model_destroy(policies[p].nh);
#endif
        policies[p].acts.destroy(); policies[p].params.destroy();
    }
    return result;
}
