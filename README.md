# Temple-Run-RL

An open-source reinforcement-learning environment for **Temple Run 2**-style endless
running, built as a CPU environment for [PufferLib](https://github.com/PufferAI/PufferLib)
and trained on **Apple Silicon** with the Metal backend.

This repository contains the environment source (`ocean/templerun2/`), the PufferLib
build integration (`build.sh`, `config/`), the shared Apple Metal trainer runtime
(`src/metal_*`), unit tests, and the asset-exporter scripts (`tools/`). It does **not**
ship any Temple Run 2 game assets, decompiled code, or a pre-trained policy — see
*Assets* below.

The four gestures are **left / right / jump / slide**, plus **idle**. The policy runs at
30 Hz (one swipe decision every two 60 Hz physics frames) against a 238-float structured
observation and a 5-way discrete action space.

## Status and honesty about approximations

This is a research/hackathon project, not a finished product:

- It is **not** a pixel-perfect or rules-perfect copy of Temple Run 2. The endless run,
  obstacle, coin and powerup systems are approximations of the shipped game's behaviour.
- Unity UI, ads, IAP, audio, missions and the upgrade economy are **not** reproduced.
- Zone progression is simplified to category-based track-piece selection.
- The trained policy is **not** reliable and still dies. Do not read a successful demo
  clip as a solved game.
- The graphical demo requires assets you must supply yourself (see *Assets*); the
  environment can run headless without them.

See `ocean/templerun2/README.md` for a detailed, file-by-file description of the
simulation and what was recovered/approximated.

## Rewards

The reward function is deliberately simple and shaped to keep values inside the Metal
trainer's `[-1, 1]` clipping range:

| Signal | Config key | Default value |
|---|---|---|
| Obstacle cleared without a hit | `reward_clear` | `0.125` |
| Coin picked up (per coin value) | `reward_coin` | `0.01875` |
| Death | `death_penalty` | `1.0` (i.e. `-1`) |
| Swipe / gesture | `swipe_cost` | `0.0` |

The clear/coin scale preserves an intended **8:1** death-to-clear ratio. **There is no
penalty for swiping itself** (`swipe_cost = 0`); deaths and collisions are penalised,
while obstacle clears and coin pickups are rewarded. Distance, powerups and turns affect
the in-game *score* but add no RL reward. Values live in `config/templerun2.ini`.

## Requirements

Training on Apple Metal is **Apple Silicon only**:

- **macOS 26 or newer on an arm64 (Apple Silicon) Mac.** `./build.sh templerun2 --metal`
  refuses to run on Intel Macs or non-Darwin platforms.
- **Xcode command-line tools** (`xcode-select --install`) for `clang` / `clang++` and the
  Metal SDK.
- **Homebrew `libomp`**: `brew install libomp` (the build reads `brew --prefix libomp`).
- **raylib 5.5**: downloaded automatically by `build.sh` into `raylib-5.5_macos/`.
- Python 3 + `UnityPy` **only** if you want to run the asset exporters in `tools/`
  against your own extracted data.

The Metal path does not require CUDA, TensorFlow or PyTorch.

## Build and run

```sh
# Metal training executable -> ./puffer  (Apple Silicon, macOS 26+)
./build.sh templerun2 --metal

# Train with the provided config (reward/level/reward hyperparameters in config/templerun2.ini)
./puffer train

# Interactive raylib playback/standalone -> ./build/cpu_templerun2 (needs assets for visuals)
./build.sh templerun2 build/cpu_templerun2 --cpu
```

Evaluation can select the highest-probability legal action instead of sampling with
`--base.eval_deterministic=1`. Other flags such as `--base.mingru_fp16=1` (half-precision
steering GEMMs, float32 masters) are described in the code headers.

Only the `templerun2` environment is set up in this repository. `build.sh` still contains
branches for other PufferLib environments whose sources are not vendored here, so
`./build.sh <other-env>` will not work.

## Assets (you must supply your own)

**No Temple Run 2 assets are included in this repository** — no APKs, no decompiled
files, no extracted models, textures, audio or animation clips. Temple Run 2 and its
assets are the property of Imangi Studios, and this project is not affiliated with or
endorsed by them.

To run the graphical environment you must supply a **lawful copy of the assets from a
game you own** and place them under `resources/templerun2/` following the layout expected
by the renderer (see the exporter scripts in `tools/` and the `resources/templerun2/...`
paths referenced by `ocean/templerun2/templerun2_render.h`, `templerun2_character.h`,
`templerun2_atmosphere.h` and `templerun2_hud.h`). Without those assets there is **no
ready-made graphical demo**: the renderer falls back to procedural placeholders, and the
exporters in `tools/` will not run because they need your own extracted game data.

The exporter scripts (`tools/export_track_models.py`, `tools/export_character.py`,
`tools/export_props.py`, `tools/gen_colliders_table.py`, `tools/gen_model_table.py`,
`tools/obj_to_glb.py`) are our own code and only *describe* how to read your local
extracted data (`recon/...`). No such data is committed.

## Demo video and narration script

The [75-second gameplay video and screenshots](https://github.com/leelachebiyyam/Temple-Run-RL/releases/tag/hackathon-demo)
show the actual agent, without manual controls or scripted obstacle avoidance.
Download the [MP4](https://github.com/leelachebiyyam/Temple-Run-RL/releases/download/hackathon-demo/Temple-Run-RL-demo.mp4)
and see [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for a short narration you can record yourself.
The clip is silent, runs at normal speed, and includes the policy's mistakes.

## Repository layout

```
build.sh                     PufferLib build integration (adds the --metal target)
config/default.ini           shared trainer defaults (Metal additions)
config/templerun2.ini        environment + training hyperparameters
ocean/templerun2/            Temple Run 2 environment source (C, header-only modules)
ocean/templerun2/README.md   detailed environment notes and approximations
ocean/osrs/osrs_item_obs_generated.h   generated table required by the shared Metal runtime
src/metal_*.h/.mm            shared Apple Metal trainer runtime (see below)
src/puf_types.h, src/metal_tensor.h    tensor/type definitions used by the runtime
src/ini.h, src/pufferenv.h   environment API + ini parser (upstream PufferLib)
src/puffercpu.c              standalone CPU eval loop
tests/test_templerun2_*.c    environment + gap/collision unit tests
tools/export_*.py            our asset-exporter scripts (source only)
vendor/                      small upstream PufferLib vendored headers
DEMO_SCRIPT.md               hackathon narration script
LICENSE                      MIT (upstream PufferLib licence)
```

## Source dependency decisions

- The tree is built from an upstream PufferLib MIT baseline (a `git archive` of the
  PufferLib commit this work is based on) with a focused overlay.
- Included from the shared Metal runtime: `metal_trainer.mm`, `metal_platform.{h,mm}`,
  `metal_kernels.mm`, `metal_shader_src.h`, `metal_env.h`, `metal_custom_encoder.h`,
  `metal_eval.h`, `metal_selfplay.h`, `metal_sweep.h`, `metal_tensor.h`, and the encoder
  implementations `metal_{asteroids,minimal,craftax,nmmo3,osrs,protein}.{h,mm}` plus
  `puf_types.h` and `ocean/osrs/osrs_item_obs_generated.h` (a generated table the OSRS
  encoder references at compile time).
- Excluded as out of scope for a CPU-environment Metal build: the GPU-environment path
  (`src/metal_gpu_env.h` and the `breakout`/`admiral`/`brotato` `*_metal` headers), the
  NetHack Metal path (`src/metal_nethack*`), CUDA sources (`*.cu`), resource/asset bundles,
  and the other PufferLib environment ports.

## Upstream attribution

This project is built on **[PufferLib](https://github.com/PufferAI/PufferLib)** by
PufferAI, used under the MIT License. The upstream licence text is preserved verbatim in
`LICENSE`. Portions of `src/`, `vendor/`, `build.sh`, `config/default.ini` and the
environment API are PufferLib code (Copyright © 2022 PufferAI). The Temple Run 2
environment, exporter scripts and tests are additions by this project's authors.

Temple Run 2 is a trademark of Imangi Studios. This repository is an independent,
unaffiliated research project and ships none of Imangi's assets or code.
