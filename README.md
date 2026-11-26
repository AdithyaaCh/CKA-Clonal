# CKA with clonal selection

This repository tests whether CLONALG improves CKA's selection of previously learned components in continual reinforcement learning. It compares CLONALG with argmax and random baselines in Atari Freeway, Space Invaders, and Meta-World push-wall. See [the robotics README](robotic/README.md) for the Meta-World setup and run commands.

## Results

CLONALG led the reported Freeway averages. Random search or initialization led the corresponding Space Invaders comparisons and had the highest final return on Meta-World push-wall.

### Atari

**Sequential transfer.** Mean episodic return in steps 450,000–500,000 of each 500,000-step visit, averaged equally over visits 2–9 at seed 0. [Return CSVs](data/)

| Game | CLONALG | Argmax | Random |
| --- | ---: | ---: | ---: |
| Freeway | **23.11** | 22.99 | 22.77 |
| Space Invaders | 434.85 | 432.83 | **478.82** |

**Alpha search.** Mean held-out episodic return before and after PPO, averaged over visits 7 and 9 and three seeds per game. [Alpha search results](long_runs/atari_alpha_search/)

| Game | CLONALG before → after | Random search before → after |
| --- | ---: | ---: |
| Freeway | 10.77 → **23.34** | 10.64 → 23.11 |
| Space Invaders | 330.50 → 502.46 | 336.09 → **521.94** |

**Vector retention.** Mean reward per 1,000 steps after PPO, averaged over four held-out modes and three seeds per game. [Vector retention results](long_runs/atari_vector_retention/)

| Game | CLONALG | Random | Recent | Cosine merge |
| --- | ---: | ---: | ---: | ---: |
| Freeway | **27.06** | 24.37 | 21.72 | 22.93 |
| Space Invaders | 2,137.07 | **2,162.88** | 2,148.92 | 2,076.54 |

### Meta-World push-wall

Final evaluation after 300,000 target training steps at seed 0. [Comparison JSON](robotic/agents/robotic_push_seed0_300k_cpu/comparison_seed_0.json)

| Method | Mean return | Success rate |
| --- | ---: | ---: |
| CLONALG | 18.24 | 0% |
| Argmax | 21.37 | 0% |
| Random initialization | **92.45** | 0% |

These are descriptive results from the saved runs. The sequential Atari and Meta-World comparisons each use one seed. The [Atari returns](data/) include other run profiles; the archived 100,000-step profile contains partial runs. [Atari comparison summaries](long_runs/) hold the per-visit JSON files, and the [early returns plot](early_returns.png) shows smoke-test results.

## Run the Atari experiments

```bash
./bootstrap_gpu_env.sh
export PYTHON_BIN="$PWD/.venv_gpu/bin/python"
./start_gpu_runs.sh --sequence-500k
./status_gpu_runs.sh --sequence-500k
```

The bootstrap script uses Python 3.9–3.12 if already installed. If the server only has a newer Python, it installs a local Python 3.12 with `uv`, then creates `.venv_gpu`. It installs PyTorch 2.5.1 with CUDA 12.1. If the server needs CUDA 11.8, use `TORCH_CUDA=cu118 ./bootstrap_gpu_env.sh`. If CUDA PyTorch is already installed in a server environment, set `PYTHON_BIN` to that environment's Python and run `./setup_gpu_env.sh` instead of bootstrapping.

The 500,000-step profile runs ten visits per game at seed 0, comparing CLONALG, argmax, and random initialization after two shared visits. Use a separate checkout for reruns because training writes to the same paths as the bundled CSVs and summaries. Use `./stop_gpu_runs.sh --sequence-500k` to stop both jobs.

Once that profile has generated the source checkpoints in `agents/`, run the follow-up experiments with new output directories. Repeat these commands with `--game SpaceInvaders` for the second game.

```bash
"$PYTHON_BIN" run_atari_alpha_search.py --game Freeway --output-dir long_runs/alpha_search_rerun
"$PYTHON_BIN" run_atari_vector_retention.py --game Freeway --output-dir long_runs/vector_retention_rerun
```
