#ifndef PUFFERLIB_METAL_OSRS_H
#define PUFFERLIB_METAL_OSRS_H

#include "metal_platform.h"

// Parameter registration order matches CUDA: global, inventory l1/l2,
// equipment l1/l2, then (for colosseum/inferno) NPC l1/l2. All weights
// are row-major [output,input], bias-free; l1 output width is 16.
// Initialize each tensor with puf_kaiming_init(gain=sqrt(2)), consecutive seeds.
// Observations and parameters may be float32 or float16, but must share a
// dtype. All tensors must be contiguous and registered with the Metal context.
// Observations are [batch,obs_size], output and upstream gradient [batch,hidden].
// Item codes are rounded to nearest even after multiplication by 256 and must
// be in [0,201); NPC codes must be in [0,type_onehot].
// The caller retains observations, weights and activations until backward ends.
// Backward overwrites weight gradients and produces no observation gradient.
enum MetalOsrsEncounter { METAL_OSRS_COLOSSEUM, METAL_OSRS_INFERNO,
                          METAL_OSRS_ZULRAH, METAL_OSRS_PVP };

struct MetalOsrsBranch {
  PufTensor l1_w, l2_w;       // [16,features], [hidden,16]
  PufTensor flat;             // [batch,records,features]
  PufTensor z1, h1, grad_z1;  // [batch,records,16], grad_z1 training only
  PufTensor argmax;           // int32 [batch,hidden], also required for rollout
  PufTensor l1_wgrad, l2_wgrad; // same shapes as weights, training only
};
struct MetalOsrsEncoder {
  MetalOsrsEncounter encounter;
  int batch, obs_size, hidden;
  PufTensor global_w;         // [hidden,obs_size]
  PufTensor output;           // [batch,hidden]
  PufTensor global_wgrad;     // [hidden,obs_size], training only
  MetalOsrsBranch inventory;  // 28 records, 14 features, obs offset 52
  MetalOsrsBranch equipment;  // 11 records, 14 features, obs offset 80
  MetalOsrsBranch npc;        // colosseum: 24x34; inferno: 14x26
};

// Forward always saves flat/z1/h1/argmax for backward. Forward and backward
// must use the same stream or be externally synchronized. Output/gradients
// are overwritten; no accumulation across backward calls is performed.
void metal_osrs_forward(MetalOsrsEncoder &encoder, const PufTensor &obs,
                        cudaStream_t stream);
void metal_osrs_backward(MetalOsrsEncoder &encoder, const PufTensor &obs,
                         const PufTensor &grad, cudaStream_t stream);

#endif
