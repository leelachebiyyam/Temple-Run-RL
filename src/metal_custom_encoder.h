#pragma once
#include "metal_asteroids.h"
#include "metal_minimal.h"
#include "metal_craftax.h"
#include "metal_nmmo3.h"
#include "metal_osrs.h"
#include <random>

// The encoder's registration order is also the native checkpoint and gradient order.
enum MetalEncoderKind { METAL_LINEAR, METAL_MINIMAL, METAL_MINIMAL_ATTN, METAL_ASTEROIDS, METAL_CRAFTAX, METAL_NMMO3, METAL_NMMO3_ATTN, METAL_COLOSSEUM, METAL_INFERNO, METAL_ZULRAH, METAL_PVP };
static MetalEncoderKind metal_encoder_kind() {
    if (!strcmp(PUFFER_ENV_NAME, "osrs_colosseum")) return METAL_COLOSSEUM;
    if (!strcmp(PUFFER_ENV_NAME, "osrs_inferno")) return METAL_INFERNO;
    if (!strcmp(PUFFER_ENV_NAME, "osrs_zulrah")) return METAL_ZULRAH;
    if (!strcmp(PUFFER_ENV_NAME, "osrs_pvp")) return METAL_PVP;
    if (!strcmp(PUFFER_ENV_NAME, "minimal")) {
#ifdef MINIMAL_ATTN
        return METAL_MINIMAL_ATTN;
#else
        return METAL_MINIMAL;
#endif
    }
    if (!strcmp(PUFFER_ENV_NAME, "asteroids")) return METAL_ASTEROIDS;
    if (!strcmp(PUFFER_ENV_NAME, "craftax")) return METAL_CRAFTAX;
    if (!strcmp(PUFFER_ENV_NAME, "nmmo3")) {
#ifdef N3_ATTN
        return METAL_NMMO3_ATTN;
#else
        return METAL_NMMO3;
#endif
    }
    return METAL_LINEAR;
}
static void metal_tensor_register(Allocator *alloc, PufTensor &t, int a, int b = 0, int c = 0) {
    t.shape[0] = a; t.shape[1] = b; t.shape[2] = c; t.dtype_size = 4; alloc_register(alloc, &t);
}
struct MetalCustomWeights {
    MetalEncoderKind kind = METAL_LINEAR;
    int hidden = 0, n = 0;
    PufTensor w[8];
};
struct MetalCustomActivations {
    int batch = 0;
    PufTensor s[20], g[8];
    PrecisionTensor input = {};
};
static void metal_custom_params(MetalCustomWeights &w, Allocator *params, MetalEncoderKind kind, int hidden) {
    w.kind = kind; w.hidden = hidden;
    auto add = [&](int a, int b) { metal_tensor_register(params, w.w[w.n++], a, b); };
    if (kind == METAL_MINIMAL || kind == METAL_MINIMAL_ATTN) {
        add(16, 6);
        if (kind == METAL_MINIMAL_ATTN) { add(16, 16); add(16, 16); add(16, 16); }
        add(hidden, 16);
    } else if (kind == METAL_ASTEROIDS) { add(16, 9); add(hidden, 16); }
    else if (kind == METAL_CRAFTAX) { add(154, 16); add(hidden, 1635); }
    else if (kind == METAL_NMMO3 || kind == METAL_NMMO3_ATTN) {
        add(128, 1475); add(128, 1152);
        if (kind == METAL_NMMO3_ATTN) { add(70, 16); add(16, 16); add(16, 16); add(16, 16); }
        add(128, 32); add(hidden, kind == METAL_NMMO3_ATTN ? 1865 : 1817);
    } else if (kind >= METAL_COLOSSEUM) {
        add(hidden, OBS_SIZE); add(16, 14); add(hidden, 16); add(16, 14); add(hidden, 16);
        if (kind == METAL_COLOSSEUM || kind == METAL_INFERNO) {
            add(16, kind == METAL_COLOSSEUM ? 34 : 26); add(hidden, 16);
        }
    }
}
static void metal_custom_init(MetalCustomWeights &w, uint64_t *seed, cudaStream_t stream) {
    for (int i = 0; i < w.n; ++i) {
        bool normal = (w.kind == METAL_CRAFTAX && i == 0) ||
            (w.kind == METAL_NMMO3 && i == 2) ||
            (w.kind == METAL_NMMO3_ATTN && (i == 2 || i == 6));
        if (normal) {
            mtl_ensure_stream_synced(stream);
            std::mt19937_64 rng((*seed)++);
            std::normal_distribution<float> dist(0.0f, w.kind == METAL_NMMO3_ATTN && i == 2 ? 1.0f / std::sqrt(6.0f) : 1.0f);
            float *p = (float*)w.w[i].bytes;
            for (int64_t j = 0; j < w.w[i].numel(); ++j) p[j] = dist(rng);
        } else {
            float gain = (w.kind >= METAL_COLOSSEUM || ((w.kind == METAL_MINIMAL || w.kind == METAL_MINIMAL_ATTN || w.kind == METAL_ASTEROIDS) && i == 0)) ? std::sqrt(2.0f) : 1.0f;
            puf_kaiming_init(w.w[i], gain, (*seed)++, stream);
        }
    }
}
static void metal_custom_activations(MetalCustomWeights &w, MetalCustomActivations &a, Allocator *acts, Allocator *grads, int batch) {
    a.batch = batch;
    auto reg = [&](int i, int x, int y = 0, int z = 0) { metal_tensor_register(acts, a.s[i], x, y, z); };
    if (w.kind == METAL_MINIMAL || w.kind == METAL_ASTEROIDS) {
        int n = w.kind == METAL_MINIMAL ? 16 : 26, d = w.kind == METAL_MINIMAL ? 6 : 9;
        reg(0, batch, n, d); reg(1, batch, n, 16); reg(2, batch, w.hidden); reg(3, batch, w.hidden);
        if (grads) reg(4, batch, n, 16);
    } else if (w.kind == METAL_MINIMAL_ATTN) {
        reg(0, batch, 16, 6);
        for (int i = 1; i <= 5; ++i) reg(i, batch, 16, 16);
        reg(6, batch, 16); reg(7, batch, w.hidden);
        if (grads) { reg(8, batch, 16); for (int i = 9; i <= 13; ++i) reg(i, batch, 16, 16); }
    } else if (w.kind == METAL_CRAFTAX) {
        reg(0, batch, 1635); reg(1, batch, w.hidden);
    } else if (w.kind == METAL_NMMO3 || w.kind == METAL_NMMO3_ATTN) {
        int concat = w.kind == METAL_NMMO3_ATTN ? 1865 : 1817;
        reg(0, batch, 128, 12); reg(1, batch, 128, 2); reg(2, batch, concat); reg(3, batch, w.hidden);
        if (grads) { reg(4, batch, concat); reg(5, batch, 128, 12); }
        if (w.kind == METAL_NMMO3_ATTN) {
            reg(6, batch, 16, 8); for (int i = 7; i <= 10; ++i) reg(i, batch, 16, 16);
            reg(11, batch, 16); reg(12, batch, 48);
            if (grads) for (int i = 13; i <= 16; ++i) reg(i, batch, 16, 16);
        }
    } else if (w.kind >= METAL_COLOSSEUM) {
        reg(0, batch, w.hidden);
        int branches = w.kind == METAL_COLOSSEUM || w.kind == METAL_INFERNO ? 3 : 2;
        for (int i = 0; i < branches; ++i) {
            int n = i == 0 ? 28 : i == 1 ? 11 : w.kind == METAL_COLOSSEUM ? 24 : 14;
            int f = i < 2 ? 14 : w.kind == METAL_COLOSSEUM ? 34 : 26;
            int k = 1 + 5 * i;
            reg(k, batch, n, f); reg(k + 1, batch, n, 16); reg(k + 2, batch, n, 16);
            if (grads) reg(k + 3, batch, n, 16);
            reg(k + 4, batch, w.hidden);
        }
    }
    if (grads) for (int i = 0; i < w.n; ++i) metal_tensor_register(grads, a.g[i], (int)w.w[i].shape[0], (int)w.w[i].shape[1]);
}
static MetalOsrsEncoder metal_custom_osrs(MetalCustomWeights &w, MetalCustomActivations &a) {
    MetalOsrsEncoder e = {};
    e.encounter = (MetalOsrsEncounter)(w.kind - METAL_COLOSSEUM);
    e.batch = a.batch; e.obs_size = OBS_SIZE; e.hidden = w.hidden;
    e.global_w = w.w[0]; e.global_wgrad = a.g[0]; e.output = a.s[0];
    auto branch = [&](MetalOsrsBranch &b, int i) {
        int k = 1 + 5 * i, p = 1 + 2 * i;
        b.l1_w = w.w[p]; b.l2_w = w.w[p + 1];
        b.flat = a.s[k]; b.z1 = a.s[k + 1]; b.h1 = a.s[k + 2];
        b.grad_z1 = a.s[k + 3]; b.argmax = a.s[k + 4];
        b.l1_wgrad = a.g[p]; b.l2_wgrad = a.g[p + 1];
    };
    branch(e.inventory, 0); branch(e.equipment, 1);
    if (w.kind == METAL_COLOSSEUM || w.kind == METAL_INFERNO) branch(e.npc, 2);
    return e;
}
static PrecisionTensor metal_custom_forward(MetalCustomWeights &w, MetalCustomActivations &a, PrecisionTensor input, cudaStream_t stream) {
    a.input = input;
    PufTensor in = to_puf(input); int b = a.batch, h = w.hidden;
    if (w.kind >= METAL_COLOSSEUM) {
        MetalOsrsEncoder e = metal_custom_osrs(w, a);
        metal_osrs_forward(e, in, stream); return pview((float*)a.s[0].bytes, b, h);
    }
    if (w.kind == METAL_MINIMAL) {
        MetalMinimalEncoder e = {b,h,w.w[0],w.w[1],a.s[0],a.s[1],a.s[2],a.s[3],a.s[4],a.g[0],a.g[1]};
        metal_minimal_forward(e, in, stream); return pview((float*)a.s[2].bytes, b, h);
    }
    if (w.kind == METAL_ASTEROIDS) {
        MetalAsteroidsEncoder e = {b,h,w.w[0],w.w[1],a.s[0],a.s[1],a.s[2],a.s[3],a.s[4],a.g[0],a.g[1]};
        metal_asteroids_forward(e, in, stream); return pview((float*)a.s[2].bytes, b, h);
    }
    if (w.kind == METAL_MINIMAL_ATTN) {
        MetalMinimalAttentionEncoder e = {b,h,w.w[0],w.w[1],w.w[2],w.w[3],w.w[4],a.s[0],a.s[1],a.s[2],a.s[3],a.s[4],a.s[5],a.s[6],a.s[7],a.s[8],a.s[9],a.s[10],a.s[11],a.s[12],a.s[13],a.g[0],a.g[1],a.g[2],a.g[3],a.g[4]};
        metal_minimal_attention_forward(e, in, stream); return pview((float*)a.s[7].bytes, b, h);
    }
    if (w.kind == METAL_CRAFTAX) {
        MetalCraftaxEncoder e = {b,h,w.w[0],w.w[1],a.s[0],a.s[1],a.g[0],a.g[1]};
        metal_craftax_forward(e, in, stream); return pview((float*)a.s[1].bytes, b, h);
    }
    MetalNMMO3Scratch s = {(float*)a.s[0].bytes,(float*)a.s[1].bytes,(float*)a.s[2].bytes,(float*)a.s[3].bytes,(float*)a.s[4].bytes,(float*)a.s[5].bytes};
    if (w.kind == METAL_NMMO3) {
        MetalNMMO3Weights nw = {(float*)w.w[0].bytes,(float*)w.w[1].bytes,(float*)w.w[2].bytes,(float*)w.w[3].bytes};
        metal_nmmo3_forward(nw,s,input.data,b,OBS_SIZE,h,stream);
    } else {
        MetalNMMO3AttentionWeights nw = {(float*)w.w[0].bytes,(float*)w.w[1].bytes,(float*)w.w[2].bytes,(float*)w.w[3].bytes,(float*)w.w[4].bytes,(float*)w.w[5].bytes,(float*)w.w[6].bytes,(float*)w.w[7].bytes};
        MetalNMMO3AttentionScratch ns = {s,(float*)a.s[6].bytes,(float*)a.s[7].bytes,(float*)a.s[8].bytes,(float*)a.s[9].bytes,(float*)a.s[10].bytes,(float*)a.s[11].bytes,(float*)a.s[12].bytes,(float*)a.s[13].bytes,(float*)a.s[14].bytes,(float*)a.s[15].bytes,(float*)a.s[16].bytes};
        metal_nmmo3_attention_forward(nw,ns,input.data,b,OBS_SIZE,h,stream);
    }
    return pview((float*)a.s[3].bytes, b, h);
}
static void metal_custom_backward(MetalCustomWeights &w, MetalCustomActivations &a, PrecisionTensor grad, cudaStream_t stream) {
    int b = a.batch, h = w.hidden; PufTensor g = to_puf(grad), in = to_puf(a.input);
    if (w.kind >= METAL_COLOSSEUM) {
        MetalOsrsEncoder e = metal_custom_osrs(w, a);
        metal_osrs_backward(e, in, g, stream); return;
    }
    if (w.kind == METAL_MINIMAL) {
        MetalMinimalEncoder e = {b,h,w.w[0],w.w[1],a.s[0],a.s[1],a.s[2],a.s[3],a.s[4],a.g[0],a.g[1]}; metal_minimal_backward(e,g,stream); return;
    }
    if (w.kind == METAL_ASTEROIDS) {
        MetalAsteroidsEncoder e = {b,h,w.w[0],w.w[1],a.s[0],a.s[1],a.s[2],a.s[3],a.s[4],a.g[0],a.g[1]}; metal_asteroids_backward(e,g,stream); return;
    }
    if (w.kind == METAL_MINIMAL_ATTN) {
        MetalMinimalAttentionEncoder e = {b,h,w.w[0],w.w[1],w.w[2],w.w[3],w.w[4],a.s[0],a.s[1],a.s[2],a.s[3],a.s[4],a.s[5],a.s[6],a.s[7],a.s[8],a.s[9],a.s[10],a.s[11],a.s[12],a.s[13],a.g[0],a.g[1],a.g[2],a.g[3],a.g[4]}; metal_minimal_attention_backward(e,g,stream); return;
    }
    if (w.kind == METAL_CRAFTAX) {
        MetalCraftaxEncoder e = {b,h,w.w[0],w.w[1],a.s[0],a.s[1],a.g[0],a.g[1]}; metal_craftax_backward(e,in,g,stream); return;
    }
    MetalNMMO3Scratch s = {(float*)a.s[0].bytes,(float*)a.s[1].bytes,(float*)a.s[2].bytes,(float*)a.s[3].bytes,(float*)a.s[4].bytes,(float*)a.s[5].bytes};
    if (w.kind == METAL_NMMO3) {
        MetalNMMO3Weights nw = {(float*)w.w[0].bytes,(float*)w.w[1].bytes,(float*)w.w[2].bytes,(float*)w.w[3].bytes};
        MetalNMMO3Weights ng = {(float*)a.g[0].bytes,(float*)a.g[1].bytes,(float*)a.g[2].bytes,(float*)a.g[3].bytes};
        metal_nmmo3_backward(nw,s,a.input.data,grad.data,ng,b,OBS_SIZE,h,stream);
    } else {
        MetalNMMO3AttentionWeights nw = {(float*)w.w[0].bytes,(float*)w.w[1].bytes,(float*)w.w[2].bytes,(float*)w.w[3].bytes,(float*)w.w[4].bytes,(float*)w.w[5].bytes,(float*)w.w[6].bytes,(float*)w.w[7].bytes};
        MetalNMMO3AttentionWeights ng = {(float*)a.g[0].bytes,(float*)a.g[1].bytes,(float*)a.g[2].bytes,(float*)a.g[3].bytes,(float*)a.g[4].bytes,(float*)a.g[5].bytes,(float*)a.g[6].bytes,(float*)a.g[7].bytes};
        MetalNMMO3AttentionScratch ns = {s,(float*)a.s[6].bytes,(float*)a.s[7].bytes,(float*)a.s[8].bytes,(float*)a.s[9].bytes,(float*)a.s[10].bytes,(float*)a.s[11].bytes,(float*)a.s[12].bytes,(float*)a.s[13].bytes,(float*)a.s[14].bytes,(float*)a.s[15].bytes,(float*)a.s[16].bytes};
        metal_nmmo3_attention_backward(nw,ns,a.input.data,grad.data,ng,b,OBS_SIZE,h,stream);
    }
}
