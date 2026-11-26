"""Compare return-guided clonal and random alpha search on frozen Atari policies."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
from loguru import logger
import numpy as np
import torch

from clonalg_init import weights_to_beta
from models import CkaRlAgent
from run_ppo import make_env


HERE = Path(__file__).resolve().parent
SOURCE_TAG = 'gpu500k_seq10'
TARGETS = {'Freeway': {7: 7, 9: 3}, 'SpaceInvaders': {7: 9, 9: 3}}
SEEDS = (101, 202, 303)
NUM_ENVS = 8
CANDIDATE_STEPS = 32_768
CANDIDATES = 16
HOLDOUT_STEPS = 65_536
PPO_STEPS = 500_000


def write_json(path, data):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(data, indent=2) + '\n')
    pending.replace(path)


def checkpoint(game, visit, method):
    tag = f'{SOURCE_TAG}_{game}_seed0_{method}'
    return HERE / 'agents' / game / tag / f'{game}_{visit}_CKA-RL_0'


def evaluate(envs, agent, steps, seed, *, require_complete_games=True):
    """Average the raw returns of complete games over a fixed rollout budget."""
    generator = torch.Generator(device='cuda').manual_seed(seed)
    obs, _ = envs.reset(seed=seed)
    returns = []
    reward_sum = 0.0
    with torch.inference_mode():
        for _ in range(steps // NUM_ENVS):
            hidden = agent.network(torch.as_tensor(obs, device='cuda').float() / 255.0)
            probs = agent.actor(hidden).softmax(dim=-1)
            actions = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            obs, rewards, _, _, infos = envs.step(actions.cpu().numpy())
            reward_sum += rewards.sum()
            returns.extend(float(info['episode']['r']) for info in infos.get('final_info', ())
                           if info is not None and 'episode' in info)
    if require_complete_games and not returns:
        raise RuntimeError('The rollout budget produced no complete games')
    return {'mean_return': float(np.mean(returns)) if returns else None, 'episode_returns': returns,
            'reward_per_1000_steps': float(1000 * reward_sum / steps)}


def beta_from_logits(logits):
    weights = torch.from_numpy(logits).float().softmax(dim=0)
    return weights_to_beta(weights, temperature=1.0, floor=0.02)


def set_beta(agent, beta):
    with torch.no_grad():
        agent.alpha.copy_(beta)
        agent.alpha_scale.fill_(1.0)


def search(method, envs, agent, seed, initial):
    """Share four initial draws, then spend twelve more evaluations per method."""
    rng = np.random.default_rng(seed)
    rng.standard_normal((len(initial), agent.num_vectors))
    history = list(initial)
    for sigma in (0.75, 0.40, 0.25):
        if method == 'clonal':
            parents = sorted(history, key=lambda row: row['mean_return'], reverse=True)[:2]
            parents = np.repeat([row['logits'] for row in parents], 2, axis=0)
            batch = parents + rng.normal(0.0, sigma, size=parents.shape)
        else:
            batch = rng.standard_normal((4, agent.num_vectors))
        for logits in batch:
            set_beta(agent, beta_from_logits(logits))
            result = evaluate(envs, agent, CANDIDATE_STEPS, 10_000 + seed)
            history.append({**result, 'logits': logits.tolist()})
            print(f'{method} {len(history)}/{CANDIDATES}: '
                  f"{result['mean_return']:.2f}", flush=True)
    assert len(history) == CANDIDATES
    return history


def train(game, visit, mode, seed, path, base, latest, method):
    tag = f'{game}_{visit}_{seed}_{method}'
    model_dir = path / 'agents' / game / tag / f'{game}_{visit}_CKA-RL_{seed}'
    if all((model_dir / file).is_file() for file in ('encoder.pt', 'actor.pt', 'critic.pt')):
        return model_dir
    command = [sys.executable, str(HERE / 'run_ppo.py'), '--method-type', 'CKA-RL',
               '--env-id', f'ALE/{game}-v5', '--mode', str(mode), '--task-id', str(visit),
               '--seed', str(seed), '--tag', tag, '--total-timesteps', str(PPO_STEPS),
               '--complete-step-budget', '--num-envs', str(NUM_ENVS), '--pool-size', '20',
               '--prev-units', str(base), str(latest), '--alpha-beta-init-file',
               str(path / f'{method}.pt'), '--cuda', '--no-track']
    if game == 'Freeway':
        command.append('--async-envs')
    with (path / f'{method}_ppo.log').open('a') as log:
        subprocess.run(command, cwd=path, stdout=log, stderr=subprocess.STDOUT, check=True)
    return model_dir


def run_case(game, visit, mode, seed, path):
    summary_path = path / 'results.json'
    if summary_path.exists():
        return
    path.mkdir(parents=True, exist_ok=True)
    print(f'START {game} visit {visit} mode {mode} seed {seed}', flush=True)
    base, latest = checkpoint(game, 0, 'shared'), checkpoint(game, visit - 1, 'argmax')
    env_fns = [make_env(f'ALE/{game}-v5', i, False, 'evaluation', mode=mode,
                        episodic_life=False, clip_reward=False) for i in range(NUM_ENVS)]
    if game == 'Freeway':
        envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True, context='forkserver')
    else:
        envs = gym.vector.SyncVectorEnv(env_fns)
    try:
        torch.manual_seed(seed)
        agent = CkaRlAgent(envs, base, latest, pool_size=20, map_location='cuda').to('cuda').eval()
        selection_path = path / 'selection.json'
        if selection_path.exists():
            selections = json.loads(selection_path.read_text())
        else:
            # Both methods get the same four initial observations. Run them once.
            initial = []
            for logits in np.random.default_rng(seed).standard_normal((4, agent.num_vectors)):
                set_beta(agent, beta_from_logits(logits))
                result = evaluate(envs, agent, CANDIDATE_STEPS, 10_000 + seed)
                initial.append({**result, 'logits': logits.tolist()})
                print(f"shared {len(initial)}/4: {result['mean_return']:.2f}", flush=True)
            selections = {}
            for method in ('clonal', 'random'):
                history = search(method, envs, agent, seed, initial)
                best = max(history, key=lambda row: row['mean_return'])
                selections[method] = {'candidates': history, 'selected_logits': best['logits']}
            write_json(selection_path, selections)

        for method, result in selections.items():
            beta = beta_from_logits(np.asarray(result['selected_logits']))
            torch.save(beta, path / f'{method}.pt')
            set_beta(agent, beta)
            result['before_ppo'] = evaluate(envs, agent, HOLDOUT_STEPS, 20_000 + seed)
            print(f"HOLDOUT {game} {visit} {seed} {method}: {result['before_ppo']['mean_return']:.2f}",
                  flush=True)
            trained_dir = train(game, visit, mode, seed, path, base, latest, method)
            trained = CkaRlAgent.load(trained_dir, envs, load_critic=False, map_location='cuda')
            result['after_ppo'] = evaluate(envs, trained.to('cuda').eval(), HOLDOUT_STEPS, 30_000 + seed)
            print(f"POST-PPO {game} {visit} {seed} {method}: {result['after_ppo']['mean_return']:.2f}",
                  flush=True)
        write_json(summary_path, {
            'game': game, 'visit': visit, 'mode': mode, 'seed': seed,
            'base': str(base), 'latest': str(latest), 'candidate_steps': CANDIDATE_STEPS,
            'ppo_steps': PPO_STEPS, 'holdout_steps': HOLDOUT_STEPS, 'methods': selections,
        })
        print(f'DONE {game} visit {visit} seed {seed}', flush=True)
    finally:
        envs.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', required=True, choices=TARGETS)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'long_runs' / 'atari_alpha_search')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required for this experiment')
    logger.remove()
    for seed in SEEDS:
        for visit, mode in TARGETS[args.game].items():
            case = args.output_dir.resolve() / args.game / f'visit_{visit}_seed_{seed}'
            run_case(args.game, visit, mode, seed, case)


if __name__ == '__main__':
    main()
