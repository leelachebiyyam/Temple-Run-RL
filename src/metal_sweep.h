#pragma once
#include "metal_protein.h"
#include <spawn.h>
#include <sys/wait.h>
#include <fcntl.h>
extern char **environ;

static void sweep_metal(Ini *ini, const char *exe) {
    const char *goal = puf_ini_get_str(ini, "sweep", "goal");
    if (!goal || (strcmp(goal, "maximize") && strcmp(goal, "minimize"))) {
        fprintf(stderr, "sweep.goal must be maximize or minimize\n"); exit(1);
    }
    std::vector<Space> spaces;
    std::vector<std::string> keys;
    int cost_idx = -1;
    for (int i = 0; i < ini->num_sections; ++i) {
        Dict *dict = &ini->sections[i];
        if (strncmp(dict->name, "sweep.", 6)) continue;
        const char *path = dict->name + 6, *dot = strrchr(path, '.');
        if (!dot || dot == path || !dot[1]) { fprintf(stderr, "invalid sweep section %s\n", dict->name); exit(1); }
        const char *dist = dict_get_str(dict, "distribution");
        Space sp = {};
        if (!strcmp(dist, "uniform")) sp.type = SPACE_LINEAR;
        else if (!strcmp(dist, "int_uniform")) { sp.type = SPACE_LINEAR; sp.is_integer = 1; }
        else if (!strcmp(dist, "uniform_pow2")) { sp.type = SPACE_POW2; sp.is_integer = 1; }
        else if (!strcmp(dist, "log_normal")) sp.type = SPACE_LOG;
        else if (!strcmp(dist, "logit_normal")) sp.type = SPACE_LOGIT;
        else { fprintf(stderr, "invalid sweep distribution: %s\n", dist); exit(1); }
        sp.min = dict_get(dict, "min"); sp.max = dict_get(dict, "max");
        const char *scale = dict_get_str(dict, "scale");
        if (!strcmp(scale, "auto")) sp.scale = 0.5f;
        else if (!strcmp(scale, "time")) {
            if (sp.min <= 0 || sp.max <= 0 || sp.min == sp.max) { fprintf(stderr, "invalid sweep time scale\n"); exit(1); }
            sp.scale = 1.0f / (log2f(sp.max) - log2f(sp.min));
        } else sp.scale = dict_get(dict, "scale");
        if (!strcmp(path, "train.total_timesteps")) cost_idx = (int)spaces.size();
        spaces.push_back(sp); keys.emplace_back(path);
    }
    int max_runs = (int)puf_ini_get(ini, "sweep", "max_runs");
    int downsample = (int)puf_ini_get(ini, "sweep", "downsample");
    int train_gpus = (int)puf_ini_get(ini, "train", "gpus"), sweep_gpus = (int)puf_ini_get(ini, "sweep", "gpus");
    const char *metric_dist = puf_ini_get_str(ini, "sweep", "metric_distribution");
    if (spaces.empty() || max_runs < 1 || downsample < 1 || downsample > 64 || train_gpus != 1 ||
        (sweep_gpus != -1 && sweep_gpus != 1) || !metric_dist || (strcmp(metric_dist, "linear") && strcmp(metric_dist, "logit"))) {
        fprintf(stderr, "invalid Metal sweep configuration (one GPU, at least one parameter, 1..64 points required)\n"); exit(1);
    }
    SweepSpace space = {spaces.data(), (int)spaces.size(), cost_idx, !strcmp(goal, "minimize") ? -1 : 1};
    float max_cost = puf_ini_get(ini, "sweep", "max_suggestion_cost");
    ProteinSweep cfg = {};
    cfg.space = &space; cfg.num_random_samples = 10; cfg.suggestions_per_pareto = 256;
    cfg.gp_training_iter = 50; cfg.gp_learning_rate = 0.001f; cfg.optimizer_reset_frequency = 50;
    cfg.gp_max_obs = 750; cfg.infer_batch_size = 4096; cfg.use_success_prob = downsample == 1;
    cfg.prune_pareto = puf_ini_get(ini, "sweep", "prune_pareto"); cfg.use_logit = !strcmp(metric_dist, "logit");
    cfg.global_search_scale = 1.0f; cfg.max_suggestion_cost = max_cost; cfg.expansion_rate = 0.1f;
    cfg.cost_random_suggestion = -0.8f; cfg.early_stop_quantile = puf_ini_get(ini, "sweep", "early_stop_quantile");
    cfg.success_cap = std::max(max_runs * downsample * 2, 8192); cfg.failure_cap = 1024; cfg.top_k = 5; cfg.rng_seed = 73;
    ProteinSweep *protein = protein_sweep_create(cfg);
    if (!protein) { fprintf(stderr, "failed to create Protein sweep\n"); exit(1); }
    std::vector<float> sample(spaces.size());
    int completed = 0, failed = 0, run = 0;
    while (completed < max_runs) {
        ProteinSweepInfo info = {};
        if (!run) {
            for (size_t i = 0; i < keys.size(); ++i) {
                size_t dot = keys[i].rfind('.');
                std::string section = keys[i].substr(0, dot), key = keys[i].substr(dot + 1);
                sample[i] = space_normalize(&spaces[i], puf_ini_get(ini, section.c_str(), key.c_str()));
                if (!std::isfinite(sample[i]) || sample[i] < -1 || sample[i] > 1) {
                    fprintf(stderr, "default sweep value outside range: %s\n", keys[i].c_str()); exit(1);
                }
            }
        } else info = protein_sweep_suggest(protein, sample.data(), NAN);
        if (run) for (size_t i = 0; i < keys.size(); ++i) {
            double x = space_unnormalize(&spaces[i], sample[i]);
            const std::string &key = keys[i];
            if (spaces[i].is_integer || key == "train.total_timesteps" ||
                    key == "train.horizon" || key == "train.minibatch_size" ||
                    key == "policy.hidden_size" || key == "policy.num_layers" ||
                    key == "vec.total_agents" || key == "vec.num_buffers" || key == "vec.num_threads")
                x = std::round(x);
            char value[64]; snprintf(value, sizeof(value), "%.17g", x);
            puf_ini_put(ini, key.c_str(), value);
        }
        int pipefd[2]; if (pipe(pipefd)) { perror("pipe"); exit(1); }
        char value[128];
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        snprintf(value, sizeof(value), "sweep_%lld_%04d", (long long)now, run);
        puf_ini_put(ini, "base.run_id", value);
        snprintf(value, sizeof(value), "%d", pipefd[1]); puf_ini_put(ini, "base.result_fd", value);
        std::vector<std::string> args; args.emplace_back(exe); args.emplace_back("train");
        for (int s = 0; s < ini->num_sections; ++s) {
            Dict *dict = &ini->sections[s];
            for (int k = 0; k < dict->size; ++k) {
                DictItem *item = &dict->items[k]; char number[64];
                const char *val = item->str;
                if (!val) { snprintf(number, sizeof(number), "%.17g", item->value); val = number; }
                args.push_back(std::string("--") + dict->name + "." + item->key + "=" + val);
            }
        }
        std::vector<char*> argv; for (auto &arg : args) argv.push_back(&arg[0]); argv.push_back(nullptr);
        posix_spawn_file_actions_t actions; posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addclose(&actions, pipefd[0]);
        posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
        pid_t pid; int err = posix_spawnp(&pid, exe, &actions, nullptr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions); close(pipefd[1]);
        if (err) { errno = err; perror("posix_spawn"); exit(1); }
        int status; while (waitpid(pid, &status, 0) < 0) { if (errno != EINTR) { perror("waitpid"); exit(1); } }
        MetalTrainResult result; char *dst = (char*)&result; size_t received = 0;
        while (received < sizeof(result)) {
            ssize_t n = read(pipefd[0], dst + received, sizeof(result) - received);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            received += (size_t)n;
        }
        close(pipefd[0]);
        if (received != sizeof(result) || !WIFEXITED(status) || WEXITSTATUS(status) || result.points < 1 || result.points > 64) {
            fprintf(stderr, "sweep worker run=%d failed; marking sample bad\n", run);
            protein_sweep_observe(protein, sample.data(), NAN, max_cost, 1);
            if (++failed > 1000) { fprintf(stderr, "too many failed sweep workers\n"); exit(1); }
        } else {
            for (int i = 0; i < result.points; ++i) protein_sweep_observe(protein, sample.data(), result.scores[i], result.costs[i], 0);
            printf("sweep run=%d score=%.4f cost=%.2f steps=%.0f random=%d gp_obs=%d pareto=%d\n", run, result.score, result.cost, result.steps, info.is_random, info.n_gp_obs, info.n_pareto);
            ++completed;
        }
        ++run;
    }
    protein_sweep_destroy(protein);
}
