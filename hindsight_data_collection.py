import os
import argparse
import torch
import numpy as np
import gym
from learner import Learner
from torchkit.pytorch_utils import set_gpu_mode
from data_collection_config import args_point_robot_barrier, args_point_robot_rand_params
from utils import helpers as utl, offline_utils as off_utl
from torchkit import pytorch_utils as ptu
from algorithms.sac import SAC
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from utils.tb_logger import TBLogger
from models.policy import TanhGaussianPolicy
from environments.make_env import make_env
import math


def load_agent(args, agent_path):
    q1_network = FlattenMlp(input_size=args.obs_dim + args.action_dim,
                            output_size=1,
                            hidden_sizes=args.dqn_layers)
    q2_network = FlattenMlp(input_size=args.obs_dim + args.action_dim,
                            output_size=1,
                            hidden_sizes=args.dqn_layers)
    policy = TanhGaussianPolicy(obs_dim=args.obs_dim,
                                action_dim=args.action_dim,
                                hidden_sizes=args.policy_layers)
    agent = SAC(
        policy,
        q1_network,
        q2_network,

        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.soft_target_tau,

        entropy_alpha=args.entropy_alpha,
        automatic_entropy_tuning=args.automatic_entropy_tuning,
        alpha_lr=args.alpha_lr
    ).to(ptu.device)
    agent.load_state_dict(torch.load(agent_path))
    return agent


def collect_rollout_per_policy(args, policy_dir, task):
    files_list = os.listdir(policy_dir)
    agent_path = os.path.join(policy_dir, sorted(files_list)[0])

    env = make_env(args.env_name,
                        args.max_rollouts_per_task,
                        seed=args.seed,
                        n_tasks=1,
                        modify_init_state_dist=args.modify_init_state_dist
                        if 'modify_init_state_dist' in args else False,
                        on_circle_init_state=args.on_circle_init_state
                        if 'on_circle_init_state' in args else True)
    unwrapped_env = env.unwrapped
    unwrapped_env.goals = np.array([task])

    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        args.action_dim = 1
    else:
        args.action_dim = env.action_space.shape[0]
    args.obs_dim = env.observation_space.shape[0]
    args.max_trajectory_len = unwrapped_env._max_episode_steps
    args.max_trajectory_len *= args.max_rollouts_per_task
    args.act_space = env.action_space

    agent = load_agent(args, agent_path)

    policy_storage = MultiTaskPolicyStorage(
        max_replay_buffer_size=int(args.policy_buffer_size),
        obs_dim=args.obs_dim,
        action_space=env.action_space,
        tasks=[0],
        trajectory_len=args.max_trajectory_len,
        num_reward_arrays=1,
        reward_types=[],
    )

    collect_rollouts_per_task(0, agent, policy_storage, env, args.num_rollouts)

    return policy_storage


def save_buffer(policy_storage,output_dir):
    size = policy_storage.task_buffers[0].size()
    np.save(os.path.join(output_dir, 'obs'), policy_storage.task_buffers[0]._observations[:size])
    np.save(os.path.join(output_dir, 'actions'), policy_storage.task_buffers[0]._actions[:size])
    np.save(os.path.join(output_dir, 'rewards'), policy_storage.task_buffers[0]._rewards[:size])
    np.save(os.path.join(output_dir, 'next_obs'), policy_storage.task_buffers[0]._next_obs[:size])
    np.save(os.path.join(output_dir, 'terminals'), policy_storage.task_buffers[0]._terminals[:size])


def collect_rollouts_per_task(task_idx, agent, policy_storage, env, num_rollouts):
    for rollout in range(num_rollouts):
        obs = ptu.from_numpy(env.reset(task_idx))
        obs = obs.reshape(-1, obs.shape[-1])
        done_rollout = False

        while not done_rollout:
            action, _, _, _ = agent.act(obs=obs)   # SAC
            # observe reward and next obs
            next_obs, reward, done, info = utl.env_step(env, action.squeeze(dim=0))
            done_rollout = False if ptu.get_numpy(done[0][0]) == 0. else True

            # add data to policy buffer - (s+, a, r, s'+, term)
            term = env.unwrapped.is_goal_state() if "is_goal_state" in dir(env.unwrapped) else False
            rew_to_buffer = ptu.get_numpy(reward.squeeze(dim=0))
            policy_storage.add_sample(task=0,#task_idx,
                                      observation=ptu.get_numpy(obs.squeeze(dim=0)),
                                      action=ptu.get_numpy(action.squeeze(dim=0)),
                                      reward=rew_to_buffer,
                                      terminal=np.array([term], dtype=float),
                                      next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)))

            # set: obs <- next_obs
            obs = next_obs.clone()


def collect_rollout_per_goal(args, task, agents):
    models_dir = './trained_agents'
    output_dir = os.path.join(args.save_data_path, "task_{}".format(task))
    os.makedirs(output_dir)
    for i, agent_dir in enumerate(agents):
        agent_save_dir = os.path.join(output_dir,"policy_{}".format(i))
        os.makedirs(agent_save_dir)
        subdir = os.path.join(models_dir, agent_dir,'models')
        policy_storage = collect_rollout_per_policy(args, subdir, task)
        save_buffer(policy_storage, agent_save_dir)


def collect_hindsight_data():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-type', default='point_robot_wind')
    #parser.add_argument('--env-type', default='escape_room')

    args, rest_args = parser.parse_known_args()
    env = args.env_type

    if env == 'point_robot_wind':
        args = args_point_robot_rand_params.get_args(rest_args)
    elif env == 'escape_room':
        args = args_point_robot_barrier.get_args(rest_args)

    # necessary args because we use VAE functions
    args.main_data_dir = args.main_save_dir
    args.trajectory_len = 50
    args.num_trajs_per_task = None

    args.num_rollouts = 10

    set_gpu_mode(torch.cuda.is_available())

    if hasattr(args, 'save_buffer') and args.save_buffer:
        os.makedirs(args.main_save_dir, exist_ok=True)

    _, goals = off_utl.load_dataset(data_dir=args.save_dir, args=args, arr_type='numpy')

    args.save_dir = "hindsight_data"
    args.save_data_path = os.path.join(args.main_data_dir, args.env_name, args.save_dir)
    os.makedirs(args.save_data_path)
    models_dir = './trained_agents'
    all_dirs = os.listdir(models_dir)
    for i, goal in enumerate(goals):
        print("start collect rollouts for task number ", i+1)
        collect_rollout_per_goal(args, goal, all_dirs)


if __name__ == '__main__':
    collect_hindsight_data()
