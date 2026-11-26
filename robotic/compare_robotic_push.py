"""Equal-budget CKA-RL random, argmax, and CLONALG on Meta-World robotics.

Run from this directory with a Python environment containing the pinned requirements.
The first two source tasks are trained once and shared by both target-task runs.
"""

import argparse
import json
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
SEQUENCES = {
    "robotic-push": ["push-v2", "reach-wall-v2", "push-wall-v2"],
    "robotic-pick-place": ["pick-place-v2", "reach-wall-v2", "pick-place-wall-v2"],
}


def run_task(task_id, seed, tag, steps, source_paths, alpha_init, probe_steps, args):
    save_dir = HERE / "agents" / tag
    cmd = [sys.executable, str(HERE / "run_sac.py"),
           "--model-type", "cka-rl", "--task-id", str(task_id),
           "--task-sequence", args.sequence, "--seed", str(seed),
           "--tag", tag, "--save-dir", str(save_dir),
           "--total-timesteps", str(steps), "--probe-steps", str(probe_steps),
           "--random-actions-end", str(args.random_actions_end),
           "--learning-starts", str(args.learning_starts),
           "--buffer-size", str(args.buffer_size), "--batch-size", str(args.batch_size),
           "--eval-every", str(args.eval_every), "--num-evals", str(args.num_evals),
           "--alpha-init", alpha_init]
    if source_paths:
        cmd += ["--prev-units", *(str(path) for path in source_paths)]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)
    return save_dir / f"task_{task_id}__cka-rl__run_sac__{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", choices=SEQUENCES, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-steps", type=int, default=100_000)
    parser.add_argument("--target-steps", type=int, default=100_000)
    parser.add_argument("--probe-steps", type=int, default=256)
    parser.add_argument("--random-actions-end", type=int, default=1000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--num-evals", type=int, default=5)
    parser.add_argument("--tag", default="robotic_push_probe")
    args = parser.parse_args()
    if args.require_cuda:
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA GPU unavailable; refusing to run on CPU")
    if args.source_steps <= args.learning_starts or args.target_steps <= args.probe_steps + args.learning_starts:
        parser.error("Source and target step budgets must exceed warmup and probe steps")

    source_paths = []
    for task_id in range(2):
        expected = HERE / "agents" / (args.tag + "_sources") / f"task_{task_id}__cka-rl__run_sac__{args.seed}"
        if (expected / "evals.json").is_file() and (expected / "transition_features.pt").is_file():
            path = expected
            print(f"Reusing completed source task {task_id}: {path}", flush=True)
        else:
            path = run_task(task_id, args.seed, args.tag + "_sources", args.source_steps,
                            source_paths, "Randn", 0, args)
        source_paths.append(path)

    results = {}
    for method in ("Randn", "Argmax", "Clonalg"):
        expected = HERE / "agents" / (args.tag + "_" + method.lower()) / f"task_2__cka-rl__run_sac__{args.seed}"
        if (expected / "evals.json").is_file() and (expected / "mixture_init.json").is_file():
            path = expected
            print(f"Reusing completed {method} target: {path}", flush=True)
        else:
            path = run_task(2, args.seed, args.tag + "_" + method.lower(),
                            args.target_steps, source_paths, method, args.probe_steps, args)
        results[method] = {
            "evals": json.loads((path / "evals.json").read_text()),
            "init": json.loads((path / "mixture_init.json").read_text()),
            "path": str(path),
        }

    out = HERE / "agents" / args.tag / f"comparison_seed_{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sequence": SEQUENCES[args.sequence],
                               "source_steps_each": args.source_steps,
                               "target_steps_each": args.target_steps,
                               "probe_steps_each": args.probe_steps,
                               "seed": args.seed, "methods": results}, indent=2))
    print(f"Results saved to {out}")
    for method, record in results.items():
        first, last = record["evals"][0], record["evals"][-1]
        print(f"{method}: first return={first['return']:.2f}, success={first['success']:.2f}; "
              f"final return={last['return']:.2f}, success={last['success']:.2f}")


if __name__ == "__main__":
    main()
