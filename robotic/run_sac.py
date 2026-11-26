# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import pathlib
import sys
import json
from torch.utils.tensorboard import SummaryWriter
from typing import Literal, Optional, Tuple
from models import shared, SimpleAgent, CompoNetAgent, PackNetAgent, ProgressiveNetAgent, CkaRlAgent, MaskNetAgent, CbpAgent, CReLUsAgent
from tasks import get_task
from utils.AdamGnT import AdamGnT
from clonalg_init import SearchConfig, VectorStores, search, weights_to_beta, project, projected_distance
from stable_baselines3.common.buffers import ReplayBuffer
from models.cbp_modules import GnT


@dataclass
class Args:
    model_type: Literal["simple", "finetune", "componet", "packnet", "prognet", "cka-rl", "masknet", "cbpnet", "crelus"]
    """The name of the NN model to use for the agent"""
    save_dir: Optional[str] = None
    """If provided, the model will be saved in the given directory"""
    prev_units: Tuple[pathlib.Path, ...] = ()
    """Paths to the previous models. Not required when model_type is `simple` or `packnet` or `prognet`"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cw-sac"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    task_id: int = 0
    """ID number of the task"""
    task_sequence: Literal["cw10", "robotic-push", "robotic-pick-place"] = "cw10"
    """CW10 order or a focused three-task robotic arm sequence."""
    eval_every: int = 10_000
    """Evaluate the agent in determinstic mode every X timesteps"""
    num_evals: int = 10
    """Number of times to evaluate the agent"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    random_actions_end: int = 10_000
    """timesteps to take actions randomly"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    tag: str = "Debug"
    """experiment tag"""
    pool_size: int = 9
    """pool size"""
    fuse_heads: bool = True
    """fuse output heads with knowledge vectors"""
    delta_theta_mode: str = "T"
    """'T' = no merge on save, 'TAT' = merge alpha into weight on save"""
    alpha_init: str = "Randn"
    """Randn / Uniform / Major / Argmax / Clonalg"""
    alpha_major: float = 0.6
    """alpha value for Major init"""
    alpha_factor: float = 1e-3
    """alpha factor for Uniform init"""
    fix_alpha: bool = False
    """freeze alpha (untrainable)"""
    encoder_from_base: bool = False
    """load encoder from base_dir"""
    alpha_learning_rate: float = 1e-3
    """the learning rate of alpha optimizer (only for cka-rl)"""
    anneal_alpha_lr: bool = True
    """Toggle learning rate annealing for alpha optimizer"""
    probe_steps: int = 0
    """Random new-task interactions charged to the task budget before SAC training."""
    feature_store_size: int = 512
    """Number of random transition features saved for later task matching."""


def transition_features(obs, next_obs, rewards):
    """Compact task signature from a random-action transition."""
    return np.concatenate((obs, next_obs - obs, rewards[:, None]), axis=1).astype(np.float32)


def clonalg_initial_beta(prev_units, probe, expected_vectors, seed, device):
    """Prior tasks are stored oldest first; CKA vectors are newest first."""
    if expected_vectors != len(prev_units):
        raise ValueError("CLONALG initialization currently requires one vector per prior task; use a task before pool merging")
    stores = VectorStores()
    for path in reversed(prev_units):
        samples = torch.load(path / "transition_features.pt", map_location="cpu")
        stores.add(samples)
    config = SearchConfig(antibody_length=128, population=16, generations=12,
                          selected=4, projections=64, chunk_size=8, per_id_cap=512)
    generator = torch.Generator(device=device).manual_seed(seed)
    result = search(stores, torch.as_tensor(probe, device=device), config, generator)
    beta = weights_to_beta(result["weights"], config.temperature, config.alpha_floor)
    return beta, {"weights": result["weights"].cpu().tolist(),
                  "distance": result["distance"], "search_seconds": result["seconds"]}


def argmax_initial_beta(prev_units, probe, expected_vectors, seed, device):
    """Choose the single prior task with smallest projected distribution distance."""
    if expected_vectors != len(prev_units):
        raise ValueError("Argmax initialization requires one vector per prior task")
    generator = torch.Generator(device=device).manual_seed(seed)
    samples = [torch.load(path / "transition_features.pt", map_location="cpu").to(device)
               for path in reversed(prev_units)]
    count = min(128, *(len(x) for x in samples))
    if count < 2:
        raise ValueError("Each source task needs at least two transition features")
    pool = torch.cat(samples)
    projected_pool, projected_probe = project(pool, torch.as_tensor(probe, device=device), 64, generator)
    pure_rows = []
    offset = 0
    for source in samples:
        pure_rows.append(offset + torch.randperm(len(source), device=device, generator=generator)[:count])
        offset += len(source)
    distances = projected_distance(projected_pool, projected_probe, torch.stack(pure_rows), chunk_size=8)
    winner = int(distances.argmin())
    weights = torch.zeros(len(samples), device=device)
    weights[winner] = 1
    beta = weights_to_beta(weights, temperature=1.0, floor=0.02)
    return beta, {"weights": weights.cpu().tolist(), "distances": distances.cpu().tolist(), "winner": winner}

def make_env(task_id, sequence="cw10"):
    def thunk():
        env = get_task(task_id, sequence=sequence)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc = shared(
            np.array(envs.observation_space.shape).prod()
            + np.prod(envs.action_space.shape)
        )
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc(x)
        x = self.fc_out(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, envs, model):
        super().__init__()
        self.model = model

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x, **kwargs):
        mean, log_std = self.model(x, **kwargs)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, **kwargs):
        mean, log_std = self(x, **kwargs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


@torch.no_grad()
def eval_agent(agent, test_env, num_evals, global_step, writer, device, seed=None):
    total_return = 0.0
    successful_episodes = 0
    for episode in range(num_evals):
        obs, _ = test_env.reset(seed=seed if episode == 0 else None)
        episode_return = 0.0
        episode_success = False
        while True:
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            mean, _ = agent(obs)
            action = torch.tanh(mean) * agent.action_scale + agent.action_bias
            obs, reward, termination, truncation, info = test_env.step(
                action[0].cpu().numpy()
            )

            episode_return += reward
            episode_success |= bool(info["success"])

            if termination or truncation:
                successful_episodes += episode_success
                total_return += episode_return
                break
    avg_ep_ret = total_return / num_evals
    avg_success = successful_episodes / num_evals
    print(f"\nTEST: ep_ret={avg_ep_ret}, success={avg_success}\n")
    writer.add_scalar("charts/test_episodic_return", avg_ep_ret, global_step)
    writer.add_scalar("charts/test_success", avg_success, global_step)
    return {"step": global_step, "return": float(avg_ep_ret), "success": float(avg_success)}


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"task_{args.task_id}__{args.model_type}__{args.exp_name}__{args.seed}"
    print(f"\n*** Run name: {run_name} ***\n")

    writer = SummaryWriter(f"runs/{args.tag}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"*** Device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.task_id, args.task_sequence)])
    envs.single_action_space.seed(args.seed)
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # select the model to use as the agent
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
    print(obs_dim)
    print(act_dim)
    print(f"*** Loading model `{args.model_type}` ***")
    if args.model_type in ["finetune", "componet"]:
        assert (
            len(args.prev_units) > 0
        ), f"Model type {args.model_type} requires at least one previous unit"

    if args.model_type == "simple":
        model = SimpleAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)

    elif args.model_type == "finetune":
        model = SimpleAgent.load(
            args.prev_units[0], map_location=device, reset_heads=True
        ).to(device)

    elif args.model_type == "componet":
        model = CompoNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=args.prev_units,
            map_location=device,
        ).to(device)
    elif args.model_type == "packnet":
        packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)
        if len(args.prev_units) == 0:
            model = PackNetAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                task_id=args.task_id,
                total_task_num=20,
                device=device,
            ).to(device)
        else:
            model = PackNetAgent.load(
                args.prev_units[0],
                task_id=args.task_id + 1,
                restart_heads=True,
                freeze_bias=True,
                map_location=device,
            ).to(device)
    elif args.model_type == "prognet":
        model = ProgressiveNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=args.prev_units,
            map_location=device,
        ).to(device)
    elif args.model_type == "cka-rl":
        base_dir = args.prev_units[0] if len(args.prev_units) > 0 else None
        latest_dir = args.prev_units[-1] if len(args.prev_units) > 0 else None
        model = CkaRlAgent(
            base_dir=base_dir,
            latest_dir=latest_dir,
            obs_dim=obs_dim,
            act_dim=act_dim,
            fuse_shared=False,
            fuse_heads=args.fuse_heads,
            pool_size=args.pool_size,
            encoder_from_base=args.encoder_from_base,
            delta_theta_mode=args.delta_theta_mode,
            alpha_init=args.alpha_init,
            alpha_major=args.alpha_major,
            alpha_factor=args.alpha_factor,
            fix_alpha=args.fix_alpha,
        )
    elif args.model_type == 'masknet':
        if len(args.prev_units) == 0:
            model = MaskNetAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                num_tasks=20,
            ).to(device)
        else:
            model = MaskNetAgent.load(
                args.prev_units[0],
                map_location=device,
            ).to(device)
        model.set_task(args.task_id, new_task=True)
    elif args.model_type == 'cbpnet':
        if len(args.prev_units) == 0:
            model = CbpAgent(
                obs_dim=obs_dim,
                act_dim=act_dim
                ).to(device)
        else:
            model = CbpAgent.load(
                args.prev_units[0],
                map_location=device,
            ).to(device)
    elif args.model_type == "crelus":
        if len(args.prev_units) == 0:
            model = CReLUsAgent(
                obs_dim=obs_dim,
                act_dim=act_dim
                ).to(device)
        else:
            model = CReLUsAgent.load(
                args.prev_units[0],
                map_location=device,
            ).to(device)
        
    actor = Actor(envs, model).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    ao_exist = False  # separate alpha optimizer exists?
    if args.model_type == "cka-rl":
        main_params = [p for n, p in actor.named_parameters() if p.requires_grad and "alpha" not in n]
        alpha_params = [p for n, p in actor.named_parameters() if p.requires_grad and "alpha" in n]
        actor_optimizer = optim.Adam(main_params, lr=args.policy_lr)
        if len(alpha_params) > 0:
            ao_exist = True
            alpha_optimizer = optim.Adam(alpha_params, lr=args.alpha_learning_rate, eps=1e-5)
            print(f"Alpha optimizer created, LR={args.alpha_learning_rate}, params={len(alpha_params)}")
    else:
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    if args.model_type == 'cbpnet':
        actor_optimizer = AdamGnT(actor.parameters(), lr=args.policy_lr, eps=1e-5)
        GnT = GnT(net=actor.model.fc.net, opt=actor_optimizer,replacement_rate=1e-3, decay_rate=0.99, device=device,
                    maturity_threshold=1000, util_type="contribution")
        
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    if args.probe_steps < 0 or args.probe_steps >= args.total_timesteps:
        raise ValueError("probe_steps must be between 0 and total_timesteps")
    if args.alpha_init in ("Clonalg", "Argmax") and args.model_type != "cka-rl":
        raise ValueError("Clonalg/Argmax alpha initialization requires model-type cka-rl")
    if args.alpha_init in ("Clonalg", "Argmax") and args.task_id >= 2 and args.probe_steps == 0:
        raise ValueError("Clonalg/Argmax initialization needs --probe-steps > 0")

    feature_store = []
    init_info = None

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    probe_features = []
    for _ in range(args.probe_steps):
        probe_actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        probe_next, probe_rewards, probe_done, probe_truncated, _ = envs.step(probe_actions)
        if not (probe_done | probe_truncated).any():
            probe_features.append(transition_features(obs, probe_next, probe_rewards)[0])
        obs = probe_next
    if args.alpha_init in ("Clonalg", "Argmax") and args.task_id >= 2:
        if not probe_features:
            raise ValueError("No valid probe transitions were collected")
        initializer = clonalg_initial_beta if args.alpha_init == "Clonalg" else argmax_initial_beta
        beta, init_info = initializer(args.prev_units, np.stack(probe_features),
                                      model.num_vectors, args.seed, device)
        with torch.no_grad():
            model.alpha.copy_(beta.to(model.alpha.device))
        init_info["method"] = args.alpha_init
        print(f"{args.alpha_init} alpha initialization: {init_info}")
    elif args.alpha_init == "Randn" and args.task_id >= 2 and args.probe_steps > 0:
        with torch.no_grad():
            weights = torch.softmax(model.alpha * model.alpha_scale, dim=0)
        init_info = {"method": "Randn", "weights": weights.cpu().tolist()}
    eval_env = get_task(args.task_id, sequence=args.task_sequence)
    eval_history = []
    if args.task_id >= 2 and args.probe_steps > 0:
        eval_history.append(eval_agent(actor, eval_env, args.num_evals, args.probe_steps,
                                       writer, device, seed=args.seed + 10000))
    for global_step in tqdm(range(args.probe_steps, args.total_timesteps)):
        # ALGO LOGIC: put action logic here
        if global_step < args.random_actions_end:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            if args.model_type == "componet" and global_step % 1000 == 0:
                actions, _, _ = actor.get_action(
                    torch.Tensor(obs).to(device),
                    writer=writer,
                    global_step=global_step,
                )
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if len(feature_store) < args.feature_store_size and not (terminations | truncations).any():
            feature_store.append(transition_features(obs, next_obs, rewards)[0])

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # print(
                #     f"global_step={global_step}, episodic_return={info['episode']['r']}, success={info['success']}"
                # )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                writer.add_scalar("charts/success", info["success"], global_step)

                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                if args.anneal_alpha_lr and ao_exist:
                    frac = max(0.0, 1.0 - (global_step - args.learning_starts) / (args.total_timesteps - args.learning_starts))
                    alpha_optimizer.param_groups[0]["lr"] = frac * args.alpha_learning_rate
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    if ao_exist:
                        alpha_optimizer.zero_grad()
                    actor_loss.backward()
                    if args.model_type == "packnet":
                        if global_step >= packnet_retrain_start:
                            # can be called multiple times, only the first counts
                            actor.model.start_retraining()
                        actor.model.before_update()
                    actor_optimizer.step()
                    if ao_exist:
                        alpha_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if ao_exist:
                    writer.add_scalar(
                        "charts/alpha_learning_rate", alpha_optimizer.param_groups[0]["lr"], global_step
                    )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )
            if args.model_type == 'cbpnet':
                # print("cbpnet: selective initailization")
                GnT.gen_and_test(actor.model.fc.get_activations())
        if args.eval_every > 0 and (global_step + 1) % args.eval_every == 0:
            eval_history.append(eval_agent(actor, eval_env, args.num_evals, global_step + 1,
                                           writer, device, seed=args.seed + 10000))
    if not eval_history or eval_history[-1]["step"] != args.total_timesteps:
        eval_history.append(eval_agent(actor, eval_env, args.num_evals, args.total_timesteps,
                                       writer, device, seed=args.seed + 10000))

    envs.close()
    eval_env.close()
    writer.close()

    if args.save_dir is not None:
        print(f"Saving trained agent in `{args.save_dir}` with name `{run_name}`")
        actor.model.save(dirname=f"{args.save_dir}/{run_name}")
        save_path = pathlib.Path(args.save_dir) / run_name
        if feature_store:
            torch.save(torch.as_tensor(np.stack(feature_store)), save_path / "transition_features.pt")
        if init_info is not None:
            (save_path / "mixture_init.json").write_text(json.dumps(init_info, indent=2))
        (save_path / "evals.json").write_text(json.dumps(eval_history, indent=2))
