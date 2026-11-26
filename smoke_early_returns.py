"""Short Atari task-2 PPO comparison from a shared task-0/1 checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from clonalg_init import SearchConfig, VectorStores, weights_to_beta, search, project, projected_distance
from run_ppo import make_env


HERE = Path(__file__).resolve().parent


def train(env_name, mode, seed, tag, steps, prev=(), init=None, temperature=1.0,
          device="cpu", num_steps=32, update_epochs=1):
    name = f"{env_name}_{mode}_CKA-RL_{seed}"
    path = HERE / "agents" / env_name / tag / name
    returns = HERE / "data" / env_name / tag / "CKA-RL" / str(mode) / "returns.csv"
    if path.exists() and returns.exists() and init is None:
        return path, returns
    cmd = [sys.executable, "run_ppo.py", "--method-type", "CKA-RL", "--env-id", f"ALE/{env_name}-v5",
           "--mode", str(mode), "--task-id", str(mode), "--seed", str(seed), "--tag", tag,
           "--total-timesteps", str(steps), "--num-envs", "8", "--num-steps", str(num_steps),
           "--update-epochs", str(update_epochs), "--pool-size", "10", "--no-track"]
    if device == "cpu":
        cmd.append("--no-cuda")
    if prev:
        cmd += ["--prev-units", *map(str, prev)]
    if init is not None:
        cmd += ["--alpha-beta-init-file", str(init), "--alpha-temperature", str(temperature)]
    with open(HERE / f"{tag}_mode{mode}.log", "w") as log:
        subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return path, returns


def collect(env_name, mode, encoder, seed, count=256):
    envs = gym.vector.SyncVectorEnv([make_env(f"ALE/{env_name}-v5", i, False, "smoke", mode) for i in range(8)])
    envs.single_action_space.seed(seed)
    states, _ = envs.reset(seed=seed)
    chunks = []
    device = next(encoder.parameters()).device
    with torch.no_grad():
        for _ in range((count + 7) // 8):
            actions = np.array([envs.single_action_space.sample() for _ in range(8)])
            following, _, _, _, _ = envs.step(actions)
            a = encoder(torch.as_tensor(np.asarray(states), device=device).float() / 255)
            b = encoder(torch.as_tensor(np.asarray(following), device=device).float() / 255)
            chunks.append(torch.cat([a, b - a], 1))
            states = following
    envs.close()
    return torch.cat(chunks)[:count]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", choices=["Freeway", "SpaceInvaders"], default="Freeway")
    p.add_argument("--base-steps", type=int, default=8192)
    p.add_argument("--early-steps", type=int, default=16384)
    p.add_argument("--probe", type=int, default=256)
    p.add_argument("--out", type=Path, default=HERE / "smoke_early_results.json")
    args = p.parse_args()
    shared_tag = ("smokeearly" if args.seed == 0 else f"smokeearly_{args.seed}") if args.env == "Freeway" else f"smokeearly_space_{args.seed}_{args.base_steps}"
    base, _ = train(args.env, 0, args.seed, shared_tag, args.base_steps)
    previous, _ = train(args.env, 1, args.seed, shared_tag, args.base_steps, [base])
    encoder = torch.load(base / "encoder.pt", map_location="cpu").eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    store = VectorStores(args.probe)
    for mode in (0, 1):
        store.add(collect(args.env, mode, encoder, args.seed, args.probe))
    probe = collect(args.env, 2, encoder, args.seed, args.probe)
    cfg = SearchConfig(antibody_length=128, population=24, generations=20, selected=5,
                       projections=32, chunk_size=8)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    found = search(store, probe, cfg, g)
    combined = torch.cat(list(store.samples.values()))
    pp, qp = project(combined, probe, cfg.projections, g)
    pure = torch.stack([torch.randint(args.probe, (128,), generator=g),
                        args.probe + torch.randint(args.probe, (128,), generator=g)])
    distance = projected_distance(pp, qp, pure)
    weights = {"random": torch.softmax(torch.randn(2, generator=g), 0),
               "argmax": torch.nn.functional.one_hot(distance.argmin(), 2).float(),
               "clonalg": found["weights"]}
    result = {"seed": args.seed, "probe_distances": distance.tolist(), "clonalg_seconds": found["seconds"],
              "methods": {}}
    for method, weight in weights.items():
        # Agent vector order is [mode 1, mode 0] (new vector is prepended).
        beta = weights_to_beta(weight.flip(0), 1.0, 0.02)
        tag = f"early_{args.env}_{method}_{args.seed}_{args.base_steps}"
        init_file = HERE / f"{tag}_beta.pt"
        torch.save(beta, init_file)
        _, csv_path = train(args.env, 2, args.seed, tag, args.early_steps, [base, previous], init_file)
        frame = pd.read_csv(csv_path)
        values = frame.loc[(frame.global_step > 0) & (frame.global_step <= args.early_steps), "episodic_return"]
        result["methods"][method] = {"weights_mode0_mode1": weight.tolist(),
                                      "early_return_mean": float(values.mean()),
                                      "episodes": int(len(values)), "returns_csv": str(csv_path)}
        print(method, result["methods"][method], flush=True)
    args.out.write_text(json.dumps(result, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
