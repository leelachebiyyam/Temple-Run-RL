#ifndef PUFFERLIB_METAL_CRAFTAX_H
#define PUFFERLIB_METAL_CRAFTAX_H

#include "metal_platform.h"

// CUDA parameter order: embed_w [154,16], then proj_w [hidden,1635].
// Initialize with puf_normal_init(gain=1), then puf_kaiming_init(gain=1),
// using consecutive seeds. All tensors are contiguous float32 Metal buffers.
// Observations [batch,843] contain 99 cells of eight category IDs followed by
// 51 scalar values. Per-channel IDs must be in [0,64), [0,8), [0,2), then
// five times [0,16); they are converted to integers by truncation.
// Caller retains the unchanged observation and forward activations until
// backward completes. concat is overwritten by backward. No input gradient
// is produced, matching the native encoder.
struct MetalCraftaxEncoder {
  int batch, hidden;
  PufTensor embed_w;      // [154,16]
  PufTensor proj_w;       // [hidden,1635]
  PufTensor concat;       // [batch,1635]
  PufTensor output;       // [batch,hidden]
  PufTensor embed_wgrad;  // [154,16], training only
  PufTensor proj_wgrad;   // [hidden,1635], training only
};

// Work is enqueued on stream. Forward and backward must use the same stream,
// or the caller must synchronize them. Backward overwrites both gradients.
void metal_craftax_forward(MetalCraftaxEncoder &encoder,
                           const PufTensor &obs, cudaStream_t stream);
void metal_craftax_backward(MetalCraftaxEncoder &encoder,
                            const PufTensor &obs, const PufTensor &grad,
                            cudaStream_t stream);

#endif
