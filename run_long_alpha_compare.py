"""Long, resumable Argmax versus CLONALG CKA-RL runs over every Atari mode."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from clonalg_init import SearchConfig, VectorStores, project, projected_distance, search, weights_to_beta
from smoke_early_returns import collect
from task_utils import TASKS


HERE = Path(__file__).resolve().parent


def checkpoint_dir(name, task_id, seed, tag):
    return HERE / 'agents' / name / tag / f'{name}_{task_id}_CKA-RL_{seed}'


def train(env, task_id, mode, seed, tag, steps, previous=(), beta=None, require_cuda=False,
          async_envs=False, complete_step_budget=False):
    name = env.split('/')[1].split('-')[0]
    path = checkpoint_dir(name, task_id, seed, tag)
    done = path / 'actor.pt'
    if done.is_file():
        print(f'REUSE {done}', flush=True)
        return path
    cmd = [sys.executable, str(HERE / 'run_ppo.py'), '--method-type', 'CKA-RL',
           '--env-id', env, '--mode', str(mode), '--task-id', str(task_id),
           '--seed', str(seed), '--tag', tag, '--total-timesteps', str(steps),
           '--num-envs', '8', '--num-steps', '128', '--update-epochs', '4',
           '--pool-size', '20', '--no-track']
    if not require_cuda:
        cmd.append('--no-cuda')
    if async_envs:
        cmd.append('--async-envs')
    if complete_step_budget:
        cmd.append('--complete-step-budget')
    if previous:
        cmd += ['--prev-units', *(str(p) for p in previous)]
    if beta is not None:
        cmd += ['--alpha-beta-init-file', str(beta)]
    if beta is not None or complete_step_budget:
        cmd += ['--probe-steps', '256']
    log_dir = HERE / 'long_runs' / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'task_{task_id}_mode_{mode}.log'
    print(f'START {name} task {task_id} mode {mode} tag={tag} log={log_path}', flush=True)
    with log_path.open('a') as log:
        log.write(f'\nCOMMAND {" ".join(cmd)}\n')
        log.flush()
        subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, check=True)
    if not done.is_file():
        raise RuntimeError(f'Run exited without checkpoint: {path}')
    print(f'DONE {name} task {task_id} mode {mode} tag={tag}', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=TASKS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--modes', type=int, nargs='+', help='Explicit sequence of Atari modes')
    parser.add_argument('--async-envs', action='store_true', help='Step Atari environments in CPU worker processes')
    parser.add_argument('--complete-step-budget', action='store_true',
                        help='Charge the probe to every visit and train on the full remaining PPO budget')
    parser.add_argument('--include-random', action='store_true', help='Compare against seeded random alpha weights')
    parser.add_argument('--require-cuda', action='store_true',
                        help='Fail before training unless a CUDA GPU is available')
    parser.add_argument('--tag', default='long_alpha')
    parser.add_argument('--reuse-tag', help='Tag of the completed checkpoint prefix to reuse')
    parser.add_argument('--reuse-visits', type=int, default=0,
                        help='Number of initial visits to load from --reuse-tag')
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        parser.error('CUDA GPU unavailable; refusing to run on CPU')
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    modes = args.modes if args.modes is not None else TASKS[args.env] * args.repeats
    if len(modes) < 2 or any(mode not in TASKS[args.env] for mode in modes):
        parser.error('Provide at least two valid modes for the selected game')
    if len(modes) > 20:
        parser.error('At most 20 visits are supported without vector merging')
    if not 0 <= args.reuse_visits <= len(modes):
        parser.error('--reuse-visits must be between zero and the total number of visits')
    if bool(args.reuse_tag) != bool(args.reuse_visits) or args.reuse_tag == args.tag:
        parser.error('Set --reuse-tag and --reuse-visits together and use distinct tags')
    name = args.env.split('/')[1].split('-')[0]
    common_tag = f'{args.tag}_{name}_seed{args.seed}_shared'
    histories = {}
    histories['Argmax'] = []
    histories['Clonalg'] = []
    if args.include_random:
        histories['Random'] = []
    reused = {}
    for task_id in range(args.reuse_visits):
        for method in histories:
            suffix = 'shared' if task_id < 2 else method.lower()
            tag = f'{args.reuse_tag}_{name}_seed{args.seed}_{suffix}'
            path = checkpoint_dir(name, task_id, args.seed, tag)
            if not all((path / file).is_file() for file in ('actor.pt', 'encoder.pt')):
                parser.error(f'Missing completed checkpoint: {path}')
            reused[task_id, method] = path

    for task_id, mode in enumerate(modes[:2]):
        path = reused.get((task_id, 'Argmax'))
        if path is None:
            path = train(args.env, task_id, mode, args.seed, common_tag, args.steps,
                         histories['Argmax'], require_cuda=args.require_cuda, async_envs=args.async_envs,
                         complete_step_budget=args.complete_step_budget)
        for history in histories.values():
            history.append(path)

    encoder = torch.load(histories['Argmax'][0] / 'encoder.pt', map_location='cpu').eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    stores = VectorStores(cap=256)
    for mode in modes[:2]:
        stores.add(collect(name, mode, encoder, args.seed, 256))

    for task_id, mode in enumerate(modes[2:], start=2):
        probe = collect(name, mode, encoder, args.seed, 256)
        if task_id < args.reuse_visits:
            for method, history in histories.items():
                history.append(reused[task_id, method])
            stores.add(probe)
            continue
        config = SearchConfig(antibody_length=128, population=24, generations=20,
                              selected=5, projections=32, chunk_size=8,
                              per_id_cap=256, alpha_floor=0.02)
        generator = torch.Generator().manual_seed(args.seed + task_id)
        result = search(stores, probe, config, generator)
        pooled = torch.cat(list(stores.samples.values()))
        projected_pool, projected_probe = project(pooled, probe, config.projections,
                                                   torch.Generator().manual_seed(args.seed + task_id))
        pure = torch.stack([i * 256 + torch.randint(256, (128,), generator=generator)
                            for i in range(task_id)])
        distances = projected_distance(projected_pool, projected_probe, pure)
        winner = int(distances.argmin())
        argmax = torch.nn.functional.one_hot(torch.tensor(winner), task_id).float()
        weights = {'Argmax': argmax, 'Clonalg': result['weights']}
        meta = {'env': args.env, 'task_id': task_id, 'mode': mode, 'seed': args.seed,
                'steps_per_visit': args.steps, 'repeats': args.repeats,
                'argmax_source': winner, 'source_distances': distances.tolist(),
                'clonalg_weights': result['weights'].tolist(), 'clonalg_search_seconds': result['seconds']}
        if args.modes is not None:
            meta['mode_sequence'] = modes
            del meta['repeats']
        if args.include_random:
            random_generator = torch.Generator().manual_seed(args.seed + task_id)
            weights['Random'] = torch.randn(task_id, generator=random_generator).softmax(dim=0)
            meta['random_weights'] = weights['Random'].tolist()
        if args.reuse_tag:
            meta.update(reuse_tag=args.reuse_tag, reuse_visits=args.reuse_visits)

        for method, weight in weights.items():
            tag = f'{args.tag}_{name}_seed{args.seed}_{method.lower()}'
            beta_dir = HERE / 'long_runs' / tag
            beta_dir.mkdir(parents=True, exist_ok=True)
            beta_file = beta_dir / f'task_{task_id}_mode_{mode}_beta.pt'
            # CKA prepends new vectors, so reverse chronological source weights.
            torch.save(weights_to_beta(weight.flip(0), 1.0, 0.02), beta_file)
            path = train(args.env, task_id, mode, args.seed, tag, args.steps,
                         histories[method], beta_file, require_cuda=args.require_cuda,
                         async_envs=args.async_envs, complete_step_budget=args.complete_step_budget)
            histories[method].append(path)
        out = HERE / 'long_runs' / f'{args.tag}_{name}_seed{args.seed}_task{task_id}_mode{mode}.json'
        out.write_text(json.dumps(meta, indent=2))
        stores.add(probe)
        print(f'FINISHED COMPARISON {name} task {task_id} mode {mode}', flush=True)


if __name__ == '__main__':
    main()
