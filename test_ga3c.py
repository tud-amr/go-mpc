import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
print(os.getcwd())
sys.path.append(os.getcwd()+'/../gym-collision-avoidance')
import sys
import argparse
import pkg_resources
import importlib
import warnings
import scipy.io as sio
import yaml
from tqdm import tqdm
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import stable_baselines
from stable_baselines.common import set_global_seeds
np.random.seed(1)
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, SAC, TD3
try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None
if mpi4py is None:
    DDPG, TRPO = None, None
else:
    from stable_baselines import DDPG, TRPO
# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer
from gym_collision_avoidance.scripts.utils import get_latest_run_id, get_saved_hyperparams, find_saved_model
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env
from gym_collision_avoidance.envs.config import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from mpc_rl_collision_avoidance.policies.MPCPolicy import MPCPolicy
from mpc_rl_collision_avoidance.algorithms.sac.sacmpc import SACMPC
from mpc_rl_collision_avoidance.algorithms.ppo2.ppo2mpc import PPO2MPC
from mpc_rl_collision_avoidance.utils.compute_performance_results import *

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3,
    'sac-mpc': SACMPC,
    'ppo2-mpc': PPO2MPC
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='gym-collision-avoidance')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='mpc_rl_collision_avoidance/logs')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2-mpc',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--scenario', help='Testing scenario', default='',
                        type=str, required=False)
    parser.add_argument('-n', '--n-episodes', help='number of episodes', default=100,
                        type=int)
    parser.add_argument('--n-envs', help='number of environments', default=1,
                        type=int)
    parser.add_argument('--n-agents', help='number of agents', default=1,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=29,
                        type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--no-render', action='store_true', default=True,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--load-best', action='store_true', default=False,
                        help='Load best model instead of last model if available')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--record', action='store_true', default=False,
                        help='Save episode images and gifs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--policy', help='Ego agent policy', default='MPCPolicy', type=str)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = dir_path + '/' + folder
    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print('Loading latest experiment, id={}'.format(args.exp_id))

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
    else:
        log_path = os.path.join(folder, algo)


    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    model_path = find_saved_model(algo, log_path, env_id, load_best=args.load_best)

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        args.n_envs = 1

    set_global_seeds(args.seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    log_dir = args.reward_log if args.reward_log != '' else None

    ####### Gym-collision-avodiance Environment - Swap Scenario
    Config.TRAIN_SINGLE_AGENT = True
    Config.ANIMATE_EPISODES = args.record
    Config.SHOW_EPISODE_PLOTS = False
    Config.SAVE_EPISODE_PLOTS = args.record
    Config.EVALUATE_MODE = True

    env, one_env = create_env()
    if args.scenario != "":
        one_env.scenario = [args.scenario]

    one_env.number_of_agents = args.n_agents
    env.unwrapped.envs[0].env.ego_policy = args.policy
    obs = env.reset()

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not args.stochastic

    # Save plot trajectories
    plot_save_dir = log_path + '/figs/ga3c/'
    os.makedirs(plot_save_dir, exist_ok=True)
    one_env.plot_save_dir = plot_save_dir

    total_reward = 0
    step = 0
    done = False

    episode_stats = []

    for ep_id in tqdm(range(args.n_episodes)):
        actions = []
        agents = env.unwrapped.envs[0].env.agents
        number_of_agents = len(one_env.agents)-1
        agents[0].policy.x_error_weight_ = 1.0
        agents[0].policy.y_error_weight_ = 1.0
        agents[0].policy.cost_function_weight = 0.0
        episode_step = 0
        state = None
        total_reward = 0
        while not done:

            # Take a step in the environment, record reward/steps for logging
            obs, rewards, done, which_agents_done = env.step([None])

            total_reward += rewards[0]
            step += 1
            episode_step += 1

        # After end of episode, store some statistics about the environment
        # Some stats apply to every gym env...

        generic_episode_stats = {
            'total_reward': total_reward,
            'steps': step,
            'actions': actions
        }

        agents = one_env.prev_episode_agents
        time_to_goal = np.array([a.t for a in agents])
        extra_time_to_goal = np.array([a.t - a.straight_line_time_to_reach_goal for a in agents])

        collision = agents[0].in_collision
        timeout = agents[0].ran_out_of_time
        all_at_goal = np.array(
                np.all([a.is_at_goal for a in agents])).tolist()
        any_stuck = np.array(
                np.any([not a.in_collision and not a.is_at_goal for a in agents])).tolist()
        outcome = "collision" if collision else "all_at_goal" if all_at_goal else "stuck"
        if len(agents) > 1:
            specific_episode_stats = {
                    'num_agents': len(agents),
                    'time_to_goal': time_to_goal,
                    'total_time_to_goal': np.sum(time_to_goal),
                    'extra_time_to_goal': extra_time_to_goal,
                    'collision': collision,
                    'stuck': timeout,
                    'all_at_goal': all_at_goal,
                    'any_stuck': any_stuck,
                    'outcome': outcome,
                    'ego_agent_traj': agents[0].global_state_history[:episode_step]
            }
        else:
            specific_episode_stats = {
                    'num_agents': len(agents),
                    'time_to_goal': time_to_goal,
                    'total_time_to_goal': np.sum(time_to_goal),
                    'extra_time_to_goal': extra_time_to_goal,
                    'collision': collision,
                    'stuck': timeout,
                    'ego_agent_traj': agents[0].global_state_history[:episode_step],
                    'all_at_goal': all_at_goal,
                    'any_stuck': any_stuck,
                    'outcome': outcome
            }

        # Merge all stats into a single dict
        episode_stats.append({**generic_episode_stats, **specific_episode_stats})
        done = False
        one_env.test_case_index = ep_id +1

    episode_stats_dict = {
        "all_episodes_stats": episode_stats
    }
    results_file = stats_path + '_GA3C_model_'+str(args.exp_id)+'_'+str(args.n_agents)+'_agents_perf_results.mat'
    sio.savemat(results_file, episode_stats_dict)

    perf_results = process_statistics(episode_stats)

    with open(os.path.join(stats_path, 'model_'+str(args.exp_id)+'_'+str(args.n_agents)+'_ga3c_agents_perf_results.yml'), 'w') as f:
        yaml.dump(perf_results, f)

if __name__ == '__main__':
    main()
