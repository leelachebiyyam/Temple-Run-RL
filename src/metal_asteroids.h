#ifndef PUFFERLIB_METAL_ASTEROIDS_H
#define PUFFERLIB_METAL_ASTEROIDS_H

#include "metal_platform.h"

// Asteroids' CUDA encoder uses two bias-free matrices, input_w [16,9] and
// output_w [hidden,16], in that order. Initialize them with puf_kaiming_init
// gains sqrt(2) and 1 respectively, using consecutive seeds. Tensors must be
// contiguous and share f32 or f16 precision, except argmax (32-bit integer).
// The caller owns all tensors and keeps obs alive until forward completes.
struct MetalAsteroidsEncoder {
  int batch, hidden;
  PufTensor input_w, output_w;
  PufTensor points;         // [batch,26,9]
  PufTensor entity;         // [batch,26,16], pre-ReLU
  PufTensor output;         // [batch,hidden]
  PufTensor argmax;         // [batch,hidden], int32 (also required for rollout)
  PufTensor grad_entity;    // [batch,26,16], training only
  PufTensor input_wgrad;    // [16,9], training only
  PufTensor output_wgrad;   // [hidden,16], training only
};

// GPU work is enqueued on stream. Forward must precede backward on the same
// stream (or the caller must synchronize); backward overwrites weight gradients.
void metal_asteroids_forward(MetalAsteroidsEncoder &encoder,
                             const PufTensor &obs, cudaStream_t stream);
void metal_asteroids_backward(MetalAsteroidsEncoder &encoder,
                              const PufTensor &grad, cudaStream_t stream);

#endif
