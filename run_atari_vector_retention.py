"""Compare four-vector retention strategies on previously seen Atari modes."""

import argparse
from contextlib import ExitStack, contextmanager
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
from loguru import logger
import numpy as np
import torch

from models import CkaRlAgent
from run_atari_alpha_search import HERE, NUM_ENVS, PPO_STEPS, SEEDS, checkpoint, evaluate, set_beta, write_json
from run_ppo import make_env


BANK_SIZE = 4
LAST_VISIT = 8
REVISIT_MODE = 3
SEARCH_MODES = (0, 1)
HOLDOUT_MODES = {'Freeway': (0, 1, 3, 4), 'SpaceInvaders': (0, 1, 3, 5)}
CANDIDATES = 16
STEPS_PER_MODE = 16_384


@contextmanager
def environments(game, mode):
    env_fns = [make_env(f'ALE/{game}-v5', i, False, 'retention', mode=mode,
                        episodic_life=False, clip_reward=False) for i in range(NUM_ENVS)]
    if game == 'Freeway':
        envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True, context='forkserver')
    else:
        envs = gym.vector.SyncVectorEnv(env_fns)
    try:
        yield envs
    finally:
        envs.close()


def read_policy(game, envs, bank=None):
    return CkaRlAgent(envs, checkpoint(game, 0, 'shared'), checkpoint(game, LAST_VISIT, 'argmax'),
                      pool_size=20 if bank is None else BANK_SIZE, alpha_init='Uniform',
                      map_location='cuda', actor_vector_bank=bank).to('cuda').eval()


def select_bank(bank, indices):
    return [{name: tensor[list(indices)].cpu() for name, tensor in layer.items()} for layer in bank]


def cosine_merge(bank):
    """Merge the same pair across layers using cosine similarity of complete vectors."""
    count = len(bank[0]['weight'])
    flattened = torch.cat([tensor.flatten(1).cpu() for layer in bank for tensor in layer.values()], dim=1)
    groups = [(i,) for i in range(count)]
    while len(groups) > BANK_SIZE:
        centroids = torch.stack([flattened[list(group)].mean(dim=0) for group in groups])
        normalized = torch.nn.functional.normalize(centroids)
        similarities = normalized @ normalized.T
        invalid = torch.ones_like(similarities, dtype=torch.bool).tril()
        pair = int(similarities.masked_fill(invalid, -torch.inf).argmax())
        i, j = divmod(pair, len(groups))
        merged = groups[i] + groups[j]
        groups = [group for k, group in enumerate(groups) if k not in (i, j)] + [merged]
    merged_bank = [{name: torch.stack([tensor[list(group)].mean(dim=0).cpu() for group in groups])
                    for name, tensor in layer.items()} for layer in bank]
    return groups, merged_bank


def one_swap(indices, size):
    return {tuple(sorted(set(indices) - {old} | {new})) for old in indices
            for new in range(size) if new not in indices}


def find_subset(method, initial, universe, count, rng, score):
    history = list(initial)
    tried = {tuple(row['indices']) for row in history}
    for _ in range(3):
        if method == 'clonal':
            parents = sorted(history, key=lambda row: row['score'], reverse=True)[:2]
            proposals = [one_swap(row['indices'], count) for row in parents for _ in range(2)]
        else:
            proposals = [universe] * 4
        for pool in proposals:
            options = sorted(pool - tried)
            indices = options[int(rng.integers(len(options)))]
            tried.add(indices)
            result = score(indices)
            history.append(result)
            print(f"{method} {len(history)}/{CANDIDATES}: {result['score']:.4f} {indices}", flush=True)
    assert len(tried) == CANDIDATES
    return history


def search_banks(game, seed, path):
    with ExitStack() as stack:
        envs = {mode: stack.enter_context(environments(game, mode)) for mode in SEARCH_MODES}
        agent = read_policy(game, envs[SEARCH_MODES[0]])
        count = agent.num_vectors
        if count != LAST_VISIT + 1:
            raise RuntimeError('Unexpected number of stored vectors in the source checkpoint')
        bank = agent.actor_vectors
        selection_file = path / 'selection.json'
        if selection_file.is_file():
            selection = json.loads(selection_file.read_text())
        else:
            set_beta(agent, torch.zeros(count))
            reference = {mode: evaluate(env, agent, STEPS_PER_MODE, 10_000 + mode * 1000 + seed,
                                        require_complete_games=False)
                         for mode, env in envs.items()}

            def score(indices):
                beta = torch.full((count,), -torch.inf)
                beta[list(indices)] = 0.0
                set_beta(agent, beta)
                scores = {mode: evaluate(env, agent, STEPS_PER_MODE, 10_000 + mode * 1000 + seed,
                                         require_complete_games=False)
                          for mode, env in envs.items()}
                ratios = [scores[mode]['reward_per_1000_steps'] /
                          max(1.0, reference[mode]['reward_per_1000_steps'])
                          for mode in envs]
                return {'indices': indices, 'score': float(np.mean(ratios)), 'modes': scores}

            universe = set(combinations(range(count), BANK_SIZE))
            ordered = sorted(universe)
            rng = np.random.default_rng(seed)
            initial = []
            for i in rng.choice(len(ordered), 4, replace=False):
                initial.append(score(ordered[i]))
                print(f"shared {len(initial)}/4: {initial[-1]['score']:.4f}", flush=True)
            selection = {}
            for method in ('clonal', 'random'):
                history = find_subset(method, initial, universe, count, np.random.default_rng(seed), score)
                winner = max(history, key=lambda row: row['score'])
                selection[method] = {'selected': winner['indices'], 'candidates': history}
            selection['search_reference'] = reference
            write_json(selection_file, selection)

    groups, merged_bank = cosine_merge(bank)
    recent = tuple(range(BANK_SIZE))
    banks = {method: select_bank(bank, selection[method]['selected']) for method in ('clonal', 'random')}
    banks.update(recent=select_bank(bank, recent), cosine_merge=merged_bank)
    selection.update(recent={'selected': recent}, cosine_merge={'groups': groups})
    return selection, banks


def heldout(game, seed, bank=None, trained=None):
    scores = {}
    agent = None
    for mode in HOLDOUT_MODES[game]:
        with environments(game, mode) as envs:
            if agent is None:
                if trained is None:
                    agent = read_policy(game, envs, bank)
                else:
                    agent = CkaRlAgent.load(trained, envs, load_critic=False, map_location='cuda').to('cuda').eval()
            scores[mode] = evaluate(envs, agent, STEPS_PER_MODE, 20_000 + mode * 1000 + seed,
                                    require_complete_games=False)
    return scores


def train(game, seed, path, method, bank_file):
    tag = f'{game}_{seed}_{method}'
    model_dir = path / 'agents' / game / tag / f'{game}_{LAST_VISIT + 1}_CKA-RL_{seed}'
    if all((model_dir / file).is_file() for file in ('encoder.pt', 'actor.pt', 'critic.pt')):
        return model_dir
    command = [sys.executable, str(HERE / 'run_ppo.py'), '--method-type', 'CKA-RL',
               '--env-id', f'ALE/{game}-v5', '--mode', str(REVISIT_MODE), '--task-id', str(LAST_VISIT + 1),
               '--seed', str(seed), '--tag', tag, '--total-timesteps', str(PPO_STEPS),
               '--complete-step-budget', '--num-envs', str(NUM_ENVS), '--pool-size', str(BANK_SIZE),
               '--prev-units', str(checkpoint(game, 0, 'shared')), str(checkpoint(game, LAST_VISIT, 'argmax')),
               '--actor-vector-bank', str(bank_file), '--alpha-init', 'Uniform', '--cuda', '--no-track']
    if game == 'Freeway':
        command.append('--async-envs')
    with (path / f'{method}_ppo.log').open('a') as log:
        subprocess.run(command, cwd=path, stdout=log, stderr=subprocess.STDOUT, check=True)
    return model_dir


def run_case(game, seed, path):
    result_file = path / 'results.json'
    if result_file.is_file():
        return
    path.mkdir(parents=True, exist_ok=True)
    print(f'START {game} seed {seed}', flush=True)
    torch.manual_seed(seed)
    selection, banks = search_banks(game, seed, path)
    reference_file = path / 'full_bank_holdout.json'
    if not reference_file.is_file():
        write_json(reference_file, heldout(game, seed))
    for method, bank in banks.items():
        bank_file = path / f'{method}_bank.pt'
        before_file = path / f'{method}_holdout.json'
        if not before_file.is_file():
            torch.save(bank, bank_file)
            before = heldout(game, seed, bank=bank)
            write_json(before_file, before)
            print(f"HOLDOUT {game} {seed} {method}: {before[REVISIT_MODE]['reward_per_1000_steps']:.2f} reward/1k", flush=True)
    results = {}
    for method in banks:
        method_file = path / f'{method}_result.json'
        if method_file.is_file():
            results[method] = json.loads(method_file.read_text())
            continue
        bank_file = path / f'{method}_bank.pt'
        before = json.loads((path / f'{method}_holdout.json').read_text())
        trained = train(game, seed, path, method, bank_file)
        after = heldout(game, seed, trained=trained)
        print(f"POST-PPO {game} {seed} {method}: {after[REVISIT_MODE]['reward_per_1000_steps']:.2f} reward/1k", flush=True)
        results[method] = {'selection': selection[method], 'before_ppo': before, 'after_ppo': after}
        write_json(method_file, results[method])
    write_json(result_file, {'game': game, 'seed': seed, 'bank_size': BANK_SIZE,
                             'search_modes': SEARCH_MODES, 'holdout_modes': HOLDOUT_MODES[game],
                             'search_steps_each': CANDIDATES * len(SEARCH_MODES) * STEPS_PER_MODE,
                             'ppo_steps': PPO_STEPS, 'reference': json.loads(reference_file.read_text()),
                             'methods': results})
    print(f'DONE {game} seed {seed}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', required=True, choices=HOLDOUT_MODES)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'long_runs' / 'atari_vector_retention')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required for this experiment')
    logger.remove()
    for seed in SEEDS:
        run_case(args.game, seed, args.output_dir.resolve() / args.game / f'seed_{seed}')


if __name__ == '__main__':
    main()
