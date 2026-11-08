#ifndef PUFFER_METAL_PROTEIN_H
#define PUFFER_METAL_PROTEIN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { SPACE_LINEAR, SPACE_LOG, SPACE_POW2, SPACE_LOGIT } SpaceType;
typedef struct { SpaceType type; float min, max, scale; int is_integer; } Space;
typedef struct { Space *spaces; int num, cost_idx, optimize_direction; } SweepSpace;
typedef struct {
    float score_loss, cost_loss, predicted_score, predicted_cost, rating;
    int n_pareto, n_gp_obs, n_candidates, is_random;
} ProteinSweepInfo;
typedef struct ProteinSweep {
    SweepSpace *space;
    int num_random_samples, suggestions_per_pareto, gp_training_iter;
    float gp_learning_rate;
    int optimizer_reset_frequency, gp_max_obs, infer_batch_size;
    int use_success_prob, prune_pareto, use_logit;
    float global_search_scale, max_suggestion_cost, expansion_rate;
    float cost_random_suggestion, early_stop_quantile;
    int success_cap, failure_cap, top_k;
    unsigned long long rng_seed;
} ProteinSweep;

float space_normalize(const Space *space, float value);
float space_unnormalize(const Space *space, float normalized);
ProteinSweep *protein_sweep_create(ProteinSweep config);
void protein_sweep_destroy(ProteinSweep *sweep);
void protein_sweep_observe(ProteinSweep *sweep, const float *normalized_params,
                           float score, float cost, int is_failure);
int protein_sweep_should_stop(const ProteinSweep *sweep, float score, float cost);
ProteinSweepInfo protein_sweep_suggest(ProteinSweep *sweep, float *out,
                                      float fixed_cost_norm);
#ifdef __cplusplus
}
#endif
#endif
