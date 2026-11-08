#pragma once
// Frozen policies own disjoint parameters, recurrent state, and rollout scratch.
static void metal_frozen_create(MetalEvalPolicy &pol, MetalEncoderKind kind, int count,
        int hidden, int layers, int horizon, int action_dim, int continuous,
        const int *sizes, const std::string &checkpoint,
        uint64_t seed, int per_buffer, int policy_start) {
    pol.count = count;
    pol.encoder.in_dim = OBS_SIZE; pol.encoder.out_dim = hidden;
    pol.decoder.hidden_dim = hidden; pol.decoder.output_dim = action_dim; pol.decoder.continuous = continuous;
    pol.network.hidden = hidden; pol.network.num_layers = layers; pol.network.horizon = horizon; pol.network.weights.resize(layers);
#ifdef PUFFER_NETHACK
    metal_nethack_model_register(pol.nh, std::max(1, count), hidden,
        pol.params, pol.nh.unused_gradients, pol.acts, false);
#else
    if (kind == METAL_LINEAR) encoder_reg_params(&pol.encoder, &pol.params, 4);
    else metal_custom_params(pol.custom, &pol.params, kind, hidden);
    decoder_reg_params(&pol.decoder, &pol.params, 4);
#endif
    mingru_reg_params(&pol.network, &pol.params, 4);
    wrap(pol.params); checkpoint_io(pol.params, checkpoint.c_str(), false);
    if (!count) return;
#ifndef PUFFER_NETHACK
    if (kind == METAL_LINEAR) encoder_reg_rollout(&pol.encoder, &pol.ea, &pol.acts, count);
    else metal_custom_activations(pol.custom, pol.custom_acts, &pol.acts, nullptr, count);
    decoder_reg_rollout(&pol.decoder, &pol.da, &pol.acts, count);
#endif
    mingru_reg_rollout(&pol.network, &pol.na, &pol.acts, count);
    pol.obs = pview(nullptr, count, OBS_SIZE); pol.state = pview(nullptr, layers, count, hidden);
    pol.actions = fview(nullptr, count, NUM_ATNS); pol.logp = fview(nullptr, count);
    pol.values = fview(nullptr, count); pol.mask = fview(nullptr, count, action_dim); pol.sizes.shape[0] = NUM_ATNS;
    alloc_register(&pol.acts, &pol.obs); alloc_register(&pol.acts, &pol.state); alloc_register(&pol.acts, &pol.actions);
    alloc_register(&pol.acts, &pol.logp); alloc_register(&pol.acts, &pol.values); alloc_register(&pol.acts, &pol.mask); alloc_register(&pol.acts, &pol.sizes);
    pol.rng_state.shape[0] = (int64_t)count * sizeof(MetalRngState);
    pol.rng_state.dtype_size = 1;
    alloc_register(&pol.acts, &pol.rng_state);
#ifdef PUFFER_NETHACK
    int verbs = 0; const signed char *map = env_head_consume_map(&verbs, &pol.head_stride);
    if (map && verbs > 0 && pol.head_stride > 0) {
        pol.head_map.shape[0] = verbs * pol.head_stride; pol.head_map.dtype_size = 1;
        alloc_register(&pol.acts, &pol.head_map);
    }
#endif
    wrap(pol.acts); memcpy(pol.sizes.data, sizes, NUM_ATNS * sizeof(int));
    mtl_rng_init((MetalRngState*)pol.rng_state.bytes, count, seed, per_buffer, policy_start);
#ifdef PUFFER_NETHACK
    metal_nethack_model_prepare(pol.nh, mtl_stream());
    if (pol.head_map.bytes) memcpy(pol.head_map.bytes, map, (size_t)verbs * pol.head_stride);
#endif
}
static void metal_frozen_step(MetalEvalPolicy &pol, MetalEncoderKind kind, MetalEnvVec &env,
        int policy, int layers, int hidden, int action_dim, int continuous,
        cudaStream_t stream, float epsilon = 0.0f) {
    if (!pol.count) return;
    int per = env.policy_layout[policy + 1] - env.policy_layout[policy];
    for (int b = 0; b < env.buffers; ++b) for (int a = 0; a < per; ++a) {
        int src = b * env.agents_per_buffer + env.policy_layout[policy] + a, dst = b * per + a;
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
        (const signed char*)pol.head_map.bytes, pol.head_stride, epsilon, stream);
#else
    PrecisionTensor x = kind == METAL_LINEAR ? encoder_forward(&pol.encoder, &pol.ea, pol.obs, stream) : metal_custom_forward(pol.custom, pol.custom_acts, pol.obs, stream);
    x = mingru_forward(&pol.network, x, pol.state, &pol.na, stream);
    x = decoder_forward(&pol.decoder, &pol.da, x, stream);
    if (continuous) puf5_sample_continuous(x, pol.decoder.logstd, pol.sizes, pol.actions.data, pol.logp.data, pol.values.data, pol.mask.data, (MetalRngState*)pol.rng_state.bytes, stream);
    else mtl_sample_logits_dispatch_to(x, pol.sizes, pol.actions.data, pol.logp.data, pol.values.data, pol.mask.data, action_dim, (MetalRngState*)pol.rng_state.bytes, stream);
#endif
}
