#ifndef PUFFERLIB_METAL_MINIMAL_H
#define PUFFERLIB_METAL_MINIMAL_H

#include "metal_platform.h"

// All tensors are contiguous float32 and owned by the caller. Observations have
// shape [batch,66]: two self features followed by 16 four-feature points.
// Parameters are registered in the order listed; initialize with consecutive
// puf_kaiming_init seeds and gains sqrt(2), 1. Gradients are overwritten.
struct MetalMinimalEncoder {
  int batch, hidden;
  PufTensor input_w;       // [16,6]
  PufTensor output_w;      // [hidden,16]
  PufTensor points;        // [batch,16,6]
  PufTensor entity;        // [batch,16,16], before ReLU
  PufTensor output;        // [batch,hidden]
  PufTensor argmax;        // [batch,hidden], int32
  PufTensor grad_entity;   // [batch,16,16], training only
  PufTensor input_wgrad;   // [16,6], training only
  PufTensor output_wgrad;  // [hidden,16], training only
};

void metal_minimal_forward(MetalMinimalEncoder &e, const PufTensor &obs, cudaStream_t stream);
void metal_minimal_backward(MetalMinimalEncoder &e, const PufTensor &grad, cudaStream_t stream);

// Attention variant. Register parameters in the order win,wq,wk,wv,wout;
// gains sqrt(2),1,1,1,1 with consecutive seeds. All scratch tensors below are
// required for both forward and backward; gradients are overwritten.
struct MetalMinimalAttentionEncoder {
  int batch, hidden;
  PufTensor win, wq, wk, wv, wout; // [16,6], 3 x [16,16], [hidden,16]
  PufTensor points;                 // [batch,16,6]
  PufTensor x, q, k, v;            // each [batch,16,16]
  PufTensor prob;                   // [batch,16,16] (query,key)
  PufTensor pooled;                 // [batch,16]
  PufTensor output;                 // [batch,hidden]
  PufTensor dpool;                  // [batch,16]
  PufTensor ds, dq, dk, dv, dx;    // each [batch,16,16]
  PufTensor win_g, wq_g, wk_g, wv_g, wout_g; // same shapes as weights
};

// Forward and backward must run in order on the same stream, or be explicitly
// synchronized. Preserve weights and forward scratch until backward completes.
void metal_minimal_attention_forward(MetalMinimalAttentionEncoder &e, const PufTensor &obs, cudaStream_t stream);
void metal_minimal_attention_backward(MetalMinimalAttentionEncoder &e, const PufTensor &grad, cudaStream_t stream);

#endif
