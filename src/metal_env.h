#pragma once

// Include after ENV_HEADER, which defines Env, Agent, obs_t and the environment API.
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if defined(METAL_GPU_BREAKOUT) || defined(METAL_GPU_ADMIRAL) || defined(METAL_GPU_BROTATO)
#include "metal_gpu_env.h"
#else

struct MetalEnvVec {
    Env* envs;
    int size, total_agents, buffers, agents_per_buffer, mask_size, num_policies;
    int num_workers;
    int* starts;
    int* counts;
    int* policy_layout;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
};

static inline void metal_env_create(MetalEnvVec* v, int total_agents, int buffers,
        int num_policies, Dict* vec_kwargs, Dict* env_kwargs) {
    assert(PUF_BACKEND == PUF_CPU && "Metal requires a CPU environment");
    assert(total_agents > 0 && buffers > 0 && total_agents % buffers == 0);
    assert(num_policies > 0);
    memset(v, 0, sizeof(*v));
    v->total_agents = total_agents;
    v->buffers = buffers;
    v->agents_per_buffer = total_agents / buffers;
    v->num_policies = num_policies;
    v->num_workers = (int)dict_get(vec_kwargs, "num_threads") / buffers;
    if (v->num_workers < 1) v->num_workers = 1;
    int sizes[] = ACT_SIZES;
    for (int i = 0; i < NUM_ATNS; ++i) v->mask_size += sizes[i];
    assert(v->mask_size > 0);
    v->starts = (int*)calloc(buffers, sizeof(int));
    v->counts = (int*)calloc(buffers, sizeof(int));
    v->policy_layout = (int*)calloc(num_policies + 1, sizeof(int));
    assert(v->starts && v->counts && v->policy_layout);
#ifdef MY_VEC_INIT
    v->envs = my_vec_init(&v->size, v->starts, v->counts, vec_kwargs, env_kwargs);
    assert(v->envs && v->size > 0);
#else
    v->envs = (Env*)calloc(total_agents, sizeof(Env));
    assert(v->envs);
    int created = 0;
    while (created < total_agents) {
        Env* env = &v->envs[v->size];
        env->rng = v->size;
        puf_init(env, env_kwargs);
        assert(env->num_agents > 0 && created + env->num_agents <= total_agents);
        created += env->num_agents;
        ++v->size;
    }
    int buf = 0, agents = 0;
    for (int i = 0; i < v->size; ++i) {
        agents += v->envs[i].num_agents;
        assert(agents <= v->agents_per_buffer && "environment crosses buffer boundary");
        ++v->counts[buf];
        if (agents == v->agents_per_buffer && buf < buffers - 1) {
            v->starts[++buf] = i + 1;
            agents = 0;
        }
    }
#endif
    v->observations = (obs_t*)calloc((size_t)total_agents * OBS_SIZE, sizeof(obs_t));
    v->actions = (float*)calloc((size_t)total_agents * NUM_ATNS, sizeof(float));
    v->rewards = (float*)calloc(total_agents, sizeof(float));
    v->terminals = (float*)calloc(total_agents, sizeof(float));
    v->action_mask = (unsigned char*)malloc((size_t)total_agents * v->mask_size);
    assert(v->observations && v->actions && v->rewards && v->terminals && v->action_mask);
    memset(v->action_mask, 1, (size_t)total_agents * v->mask_size);

    float frozen_pct = dict_get(vec_kwargs, "hist_policy_percent");
    for (int buf = 0; buf < buffers; ++buf) {
        int start = v->starts[buf], count = v->counts[buf];
        assert(start >= 0 && count > 0 && start + count <= v->size);
        int frozen_start = count;
        if (num_policies > 1 && frozen_pct > 0.0f)
            frozen_start -= (int)(frozen_pct * count);
        assert(frozen_start >= 0 && frozen_start <= count);
        int* counts = (int*)calloc(num_policies, sizeof(int));
        int* cursors = (int*)calloc(num_policies, sizeof(int));
        assert(counts && cursors);
        for (int e = 0; e < count; ++e) {
            Env* env = &v->envs[start + e];
            for (int s = 0; s < env->num_agents; ++s) {
                int policy = e < frozen_start ? 0 : env->agents[s].policy;
                assert(policy >= 0 && policy < num_policies);
                ++counts[policy];
            }
        }
        int offset = 0;
        for (int p = 0; p <= num_policies; ++p) {
            if (buf == 0) v->policy_layout[p] = offset;
            else assert(v->policy_layout[p] == offset);
            if (p < num_policies) {
                cursors[p] = buf * v->agents_per_buffer + offset;
                offset += counts[p];
            }
        }
        assert(offset == v->agents_per_buffer);
        for (int e = 0; e < count; ++e) {
            Env* env = &v->envs[start + e];
            int tag = 0;
            for (int s = 0; s < env->num_agents; ++s) {
                Agent* a = &env->agents[s];
                int policy = e < frozen_start ? 0 : a->policy;
                if (policy > tag) tag = policy;
                int n = cursors[policy]++;
                a->observations = v->observations + (size_t)n * OBS_SIZE;
                a->actions = v->actions + (size_t)n * NUM_ATNS;
                a->rewards = v->rewards + n;
                a->terminals = v->terminals + n;
                a->action_mask = v->action_mask + (size_t)n * v->mask_size;
            }
            env->tag = tag;
            env->boundary_reached = 0;
        }
        free(counts);
        free(cursors);
    }
}

static inline void metal_env_reset(MetalEnvVec* v) {
    memset(v->rewards, 0, (size_t)v->total_agents * sizeof(float));
    memset(v->terminals, 0, (size_t)v->total_agents * sizeof(float));
    for (int i = 0; i < v->size; ++i) puf_reset(&v->envs[i]);
}

// Call once per buffer after inference has completed and actions are visible.
static inline void metal_env_step(MetalEnvVec* v, int buffer) {
    assert(buffer >= 0 && buffer < v->buffers);
    int start = buffer * v->agents_per_buffer;
    memset(v->rewards + start, 0, (size_t)v->agents_per_buffer * sizeof(float));
    memset(v->terminals + start, 0, (size_t)v->agents_per_buffer * sizeof(float));
    #pragma omp parallel for schedule(static) num_threads(v->num_workers)
    for (int i = v->starts[buffer]; i < v->starts[buffer] + v->counts[buffer]; ++i)
        puf_step(&v->envs[i]);
}
static inline void metal_env_log(MetalEnvVec* v, Dict* out, int clear) {
    Log aggregate = {};
    float* sum = (float*)&aggregate;
    const int fields = (int)(sizeof(Log) / sizeof(float));
    for (int i = 0; i < v->size; ++i) {
        Log* log = &v->envs[i].log;
        if (log->n != 0.0f) {
            const float* values = (const float*)log;
            for (int j = 0; j < fields; ++j) sum[j] += values[j];
        }
        if (clear) memset(log, 0, sizeof(*log));
    }
    float n = aggregate.n;
    Dict env_out = {};
    if (n > 0.0f) {
        for (int j = 0; j < fields; ++j) sum[j] /= n;
        puf_log(&aggregate, &env_out);
    }
    dict_set(&env_out, "n", n);
    for (int i = 0; i < env_out.size; ++i) {
        char key[256];
        snprintf(key, sizeof(key), "env/%s", env_out.items[i].key);
        dict_set(out, key, env_out.items[i].value);
    }
    dict_clear(&env_out);
}


static inline void metal_env_close(MetalEnvVec* v) {
    for (int i = 0; i < v->size; ++i) puf_close(&v->envs[i]);
#ifdef MY_VEC_CLOSE
    my_vec_close(v->envs);
#endif
    free(v->envs);
    free(v->starts);
    free(v->counts);
    free(v->policy_layout);
    free(v->observations);
    free(v->actions);
    free(v->rewards);
    free(v->terminals);
    free(v->action_mask);
    memset(v, 0, sizeof(*v));
}
#endif
