#ifndef PUFFERLIB_METAL_NMMO3_H
#define PUFFERLIB_METAL_NMMO3_H

#include "metal_platform.h"

// All pointers must refer to Metal-registered float32 storage. Parameter and
// gradient buffers use native CUDA order: [128,1475], [128,1152], [128,32],
// [hidden,1817]. Gradients are overwritten, not accumulated. Observations are
// [batch,obs_size], outputs and their gradients are [batch,hidden]. Player
// categorical values must be in [0,128), as required by the CUDA encoder.
struct MetalNMMO3Weights {
  float *conv1, *conv2, *embedding, *projection;
};
struct MetalNMMO3Scratch {
  float *conv1;       // batch * 128 * 12
  float *conv2;       // batch * 128 * 2
  float *concat;      // batch * 1817
  float *out;         // batch * hidden
  float *dconcat;     // batch * 1817
  float *dconv1;      // batch * 128 * 12
};

// The caller retains obs, weights and scratch until backward is encoded, and
// must not reuse scratch for another forward in between. No input gradient is
// produced, matching the native encoder. Stream ordering follows MetalStream.
void metal_nmmo3_forward(const MetalNMMO3Weights &w, MetalNMMO3Scratch &s,
                         const float *obs, int batch, int obs_size, int hidden,
                         cudaStream_t stream);
void metal_nmmo3_backward(const MetalNMMO3Weights &w, MetalNMMO3Scratch &s,
                          const float *obs, const float *dout,
                          const MetalNMMO3Weights &dw, int batch, int obs_size,
                          int hidden, cudaStream_t stream);

// Attention parameter order: conv1 [128,1475], conv2 [128,1152],
// token [70,16], query/key/value [16,16] each, embedding [128,32],
// projection [hidden,1865]. The common scratch concat/dconcat must instead
// contain batch*1865 floats. Other common scratch sizes are unchanged.
struct MetalNMMO3AttentionWeights {
  float *conv1, *conv2, *token, *query, *key, *value, *embedding, *projection;
};
struct MetalNMMO3AttentionScratch {
  MetalNMMO3Scratch common;
  float *candidates; // batch * 16 * 8
  float *x;          // batch * 16 * 16
  float *q;          // batch * 16 * 16
  float *k;          // batch * 16 * 16
  float *v;          // batch * 16 * 16
  float *prob;       // batch * 16
  float *readout;    // batch * 48
  float *dq;         // batch * 16 * 16
  float *dk;         // batch * 16 * 16
  float *dv;         // batch * 16 * 16
  float *dx;         // batch * 16 * 16
};
void metal_nmmo3_attention_forward(const MetalNMMO3AttentionWeights &w,
    MetalNMMO3AttentionScratch &s, const float *obs, int batch, int obs_size,
    int hidden, cudaStream_t stream);
void metal_nmmo3_attention_backward(const MetalNMMO3AttentionWeights &w,
    MetalNMMO3AttentionScratch &s, const float *obs, const float *dout,
    const MetalNMMO3AttentionWeights &dw, int batch, int obs_size, int hidden,
    cudaStream_t stream);

#endif
