import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import uuid
import difflib
import argparse
import importlib
import warnings
from pprint import pprint
from collections import OrderedDict
import sys
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
print(os.getcwd())
try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from mpc_rl_collision_avoidance.external.stable_baselines.common import set_global_seeds
from mpc_rl_collision_avoidance.external.stable_baselines.common.cmd_util import make_atari_env
from mpc_rl_collision_avoidance.external.stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from mpc_rl_collision_avoidance.external.stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from mpc_rl_collision_avoidance.external.stable_baselines.common.schedules import constfn
from mpc_rl_collision_avoidance.external.stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from mpc_rl_collision_avoidance.external.stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, SAC, TD3
from mpc_rl_collision_avoidance.algorithms.ppo2.ppo2mpc import PPO2MPC
if mpi4py is None:
    DDPG, TRPO = None, None
else:
    from stable_baselines import DDPG, TRPO
from gym_collision_avoidance.scripts.utils import *
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from mpc_rl_collision_avoidance.policies.MPCPolicy import MPCPolicy
from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class, find_saved_model
from utils.hyperparams_opt import hyperparam_optimization


ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    #'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3,
    'ppo2-mpc': PPO2MPC,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="gym-collision-avoidance", help='environment ID')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2-mpc',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                        type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                        type=int)
    parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',
                        default=0, type=int)
    parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation',
                        default=5, type=int)
    parser.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)',
                        default=50000, type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--tensorboard_log', help='Tensorboard folder', type=str, default='./logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,
                        help='Run hyperparameters search')
    parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=1)
    parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,
                        default='tpe', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str,
                        default='median', choices=['halving', 'median', 'none'])
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[],
                        help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict,
                        help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    parser.add_argument('-uuid', '--uuid', action='store_true', default=False,
                        help='Ensure that the run has a unique ID')
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = '_{}'.format(uuid.uuid4()) if args.uuid else ''
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1)

    set_global_seeds(args.seed)

    if args.trained_agent != "":
        valid_extension = args.trained_agent.endswith('.pkl') or args.trained_agent.endswith('.zip')
        assert valid_extension and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .zip/.pkl file"

    rank = 0
    if mpi4py is not None and MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))

        args.seed += rank
        if rank != 0:
            args.verbose = 0
            args.tensorboard_log = ''

    tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, env_id)

    is_atari = False
    if 'NoFrameskip' in env_id:
        is_atari = True

    print("=" * 10, env_id, "=" * 10)
    print("Seed: {}".format(args.seed))

    # Load hyperparameters from yaml file
    with open(os.getcwd()+'/mpc_rl_collision_avoidance/hyperparams/{}.yml'.format(args.algo), 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif is_atari:
            hyperparams = hyperparams_dict['atari']
        else:
            raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

    if args.hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(args.hyperparams)

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    algo_ = args.algo
    # HER is only a wrapper around an algo
    if args.algo == 'her':
        algo_ = saved_hyperparams['model_class']
        assert algo_ in {'sac', 'ddpg', 'dqn', 'td3'}, "{} is not compatible with HER".format(algo_)
        # Retrieve the model class
        hyperparams['model_class'] = ALGOS[saved_hyperparams['model_class']]
        if hyperparams['model_class'] is None:
            raise ValueError('{} requires MPI to be installed'.format(algo_))

    if args.verbose > 0:
        pprint(saved_hyperparams)

    n_envs = hyperparams.get('n_envs', 1)

    if args.verbose > 0:
        print("Using {} environments".format(n_envs))

    # Create learning rate schedules for ppo2 and sac
    if algo_ in ["ppo2-mpc","ppo2", "sac", "td3"]:
        for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constfn(float(hyperparams[key]))
            else:
                raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))

    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        if args.verbose:
            print("Overwriting n_timesteps with n={}".format(args.n_timesteps))
        n_timesteps = args.n_timesteps
    else:
        n_timesteps = int(hyperparams['n_timesteps'])

    normalize = False
    normalize_kwargs = {}
    if 'normalize' in hyperparams.keys():
        normalize = hyperparams['normalize']
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams['normalize']

    # Convert to python object if needed
    if 'policy_kwargs' in hyperparams.keys() and isinstance(hyperparams['policy_kwargs'], str):
        hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

    # Delete keys so the dict can be pass to the model constructor
    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    log_path = "{}/{}/".format(args.log_folder, args.algo)
    save_path = os.path.join(log_path, "{}_{}{}".format(env_id, get_latest_run_id(log_path, env_id) + 1, uuid_str))
    args.tensorboard_log = save_path + "/tf_log"
    params_path = "{}/{}".format(save_path, env_id)
    os.makedirs(params_path, exist_ok=True)

    callbacks = []
    if args.save_freq > 0:
        # Account for the number of parallel environments
        args.save_freq = max(args.save_freq // n_envs, 1)
        callbacks.append(CheckpointCallback(save_freq=args.save_freq,
                                            save_path=save_path, name_prefix='rl_model', verbose=1))

    ###### gym-collision-avoidance parameters #######

    Config.TRAIN_SINGLE_AGENT = True
    Config.ANIMATE_EPISODES = True
    Config.SHOW_EPISODE_PLOTS = False
    Config.TRAIN_MODE = True
    Config.SAVE_EPISODE_PLOTS = True
    env, one_env = create_env()

    print(env.observation_space.sample())
    #env = VecNormalize(env)
    print(env.observation_space.sample())
    #agents = tc.go_to_goal
    #agents = tc.train_agents_swap_circle
    #one_env.set_agents(agents)
    #init_obs = env.reset()

    # Create test env if needed, do not normalize reward
    eval_env = None
    if args.eval_freq > 0:
        # Account for the number of parallel environments
        args.eval_freq = max(args.eval_freq // n_envs, 1)

        # Do not normalize the rewards of the eval env
        old_kwargs = None
        if normalize:
            if len(normalize_kwargs) > 0:
                old_kwargs = normalize_kwargs.copy()
                normalize_kwargs['norm_reward'] = False
            else:
                normalize_kwargs = {'norm_reward': False}

        if args.verbose > 0:
            print("Creating test environment")

        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=params_path)
        eval_callback = EvalCallback(create_env(1, eval_env=True), callback_on_new_best=save_vec_normalize,
                                     best_model_save_path=save_path, n_eval_episodes=args.eval_episodes,
                                     log_path=save_path, eval_freq=args.eval_freq)
        callbacks.append(eval_callback)

        # Restore original kwargs
        if old_kwargs is not None:
            normalize_kwargs = old_kwargs.copy()


    # Stop env processes to free memory
    if args.optimize_hyperparameters and n_envs > 1:
        env.close()

    # Parse noise string for DDPG and SAC
    if algo_ in ['ddpg', 'sac','sac-mpc', 'td3'] and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        n_actions = env.action_space.shape[0]
        if 'adaptive-param' in noise_type:
            assert algo_ == 'ddpg', 'Parameter is not supported by SAC'
            hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                desired_action_stddev=noise_std)
        elif 'normal' in noise_type:
            if 'lin' in noise_type:
                hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                      sigma=noise_std * np.ones(n_actions),
                                                                      final_sigma=hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions),
                                                                      max_steps=n_timesteps)
            else:
                hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                sigma=noise_std * np.ones(n_actions))
        elif 'ornstein-uhlenbeck' in noise_type:
            hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
        else:
            raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
        print("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparams['noise_type']
        del hyperparams['noise_std']
        if 'noise_std_final' in hyperparams:
            del hyperparams['noise_std_final']

    if ALGOS[args.algo] is None:
        raise ValueError('{} requires MPI to be installed'.format(args.algo))

    if os.path.isfile(args.trained_agent):
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams['policy']

        model = ALGOS[args.algo].load(args.trained_agent, env=env,
                                      tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

        exp_folder = args.trained_agent[:-4]
        if normalize:
            print("Loading saved running average")
            stats_path = os.path.join(exp_folder, env_id)
            if os.path.exists(os.path.join(stats_path, 'vecnormalize.pkl')):
                env = VecNormalize.load(os.path.join(stats_path, 'vecnormalize.pkl'), env)
            else:
                # Legacy:
                env.load_running_average(exp_folder)

    elif args.optimize_hyperparameters:

        if args.verbose > 0:
            print("Optimizing hyperparameters")


        def create_model(*_args, **kwargs):
            """
            Helper to create a model with different hyperparameters
            """
            return ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log,
                                    verbose=0, **kwargs)


        data_frame = hyperparam_optimization(args.algo, create_model, create_env, n_trials=args.n_trials,
                                             n_timesteps=n_timesteps, hyperparams=hyperparams,
                                             n_jobs=args.n_jobs, seed=args.seed,
                                             sampler_method=args.sampler, pruner_method=args.pruner,
                                             verbose=args.verbose)

        report_name = "report_{}_{}-trials-{}-{}-{}_{}.csv".format(env_id, args.n_trials, n_timesteps,
                                                                args.sampler, args.pruner, int(time.time()))

        log_path = os.path.join(args.log_folder, args.algo, report_name)

        if args.verbose:
            print("Writing report to {}".format(log_path))

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        data_frame.to_csv(log_path)
        exit()
    else:
        # Train an agent from scratch
        model = ALGOS[args.algo](env=env, tensorboard_log=args.tensorboard_log, verbose=args.verbose, **hyperparams)

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}

    if len(callbacks) > 0:
        kwargs['callback'] = callbacks

    # Save hyperparams
    with open(os.path.join(params_path, 'config.yml'), 'w') as f:
        yaml.dump(saved_hyperparams, f)

    # Save Configuration Parameters
    with open(os.path.join(params_path, 'simulation_config.yml'), 'w') as f:
        var = {
        "MODEL_DESCRIPTION": " diff dt with new scenarios",
        "ENABLE_COLLISION_AVOIDANCE": Config.ENABLE_COLLISION_AVOIDANCE,
        "LSTM_HIDDEN_SIZE": Config.LSTM_HIDDEN_SIZE,
        "NUM_LAYERS":Config.NUM_LAYERS,
        "NUM_HIDDEN_UNITS":Config.NUM_HIDDEN_UNITS,
        "NETWORK":Config.NETWORK,
        "GAMMA":Config.GAMMA,
        "LEARNING_RATE":Config.LEARNING_RATE,
        "NUM_TEST_CASES":Config.NUM_TEST_CASES,
        "PLOT_EVERY_N_EPISODES":Config.PLOT_EVERY_N_EPISODES,
        "DT":Config.DT,
        "REWARD_AT_GOAL":Config.REWARD_AT_GOAL,
        "REWARD_COLLISION_WITH_AGENT":Config.REWARD_COLLISION_WITH_AGENT,
        "REWARD_INFEASIBLE": Config.REWARD_INFEASIBLE,
        "REWARD_COLLISION_WITH_WALL":Config.REWARD_COLLISION_WITH_WALL ,
        "REWARD_GETTING_CLOSE":Config.REWARD_GETTING_CLOSE,
        "REWARD_ENTERED_NORM_ZONE":Config.REWARD_ENTERED_NORM_ZONE,
        "REWARD_TIME_STEP":Config.REWARD_TIME_STEP,
        "REWARD_DISTANCE_TO_GOAL":Config.REWARD_DISTANCE_TO_GOAL,
        "REWARD_WIGGLY_BEHAVIOR":Config.REWARD_WIGGLY_BEHAVIOR,
        "WIGGLY_BEHAVIOR_THRESHOLD":Config.WIGGLY_BEHAVIOR_THRESHOLD,
        "COLLISION_DIST":Config.COLLISION_DIST,
        "GETTING_CLOSE_RANGE":Config.GETTING_CLOSE_RANGE,
        "CURRICULUM_LEARNING": Config.CURRICULUM_LEARNING ,
        "JOINT_MPC_RL_TRAINING": Config.JOINT_MPC_RL_TRAINING,
        "LASERSCAN_LENGTH":Config.LASERSCAN_LENGTH,
        "NUM_STEPS_IN_OBS_HISTORY":Config.NUM_STEPS_IN_OBS_HISTORY,
        "NUM_PAST_ACTIONS_IN_STATE":Config.NUM_PAST_ACTIONS_IN_STATE ,

        "NEAR_GOAL_THRESHOLD":Config.NEAR_GOAL_THRESHOLD,
        "MAX_TIME_RATIO":Config.MAX_TIME_RATIO,

        "SENSING_HORIZON":Config.SENSING_HORIZON,

        "RVO_TIME_HORIZON":Config.RVO_TIME_HORIZON,
        "RVO_COLLAB_COEFF":Config.RVO_COLLAB_COEFF,
        "RVO_ANTI_COLLAB_T":Config.RVO_ANTI_COLLAB_T,

        "MAX_NUM_AGENTS_IN_ENVIRONMENT":Config.MAX_NUM_AGENTS_IN_ENVIRONMENT,
        "MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT":Config.MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT,
        "MAX_NUM_OTHER_AGENTS_OBSERVED":Config.MAX_NUM_OTHER_AGENTS_OBSERVED,
        "scenario": env.unwrapped.envs[0].scenario
        }
        yaml.dump(var, f)

    print("Log path: {}".format(save_path))

    # Save plot trajectories
    plot_save_dir = save_path + '/figs/'
    os.makedirs(plot_save_dir, exist_ok=True)
    one_env.plot_save_dir = plot_save_dir

    try:
        #model = ALGOS[args.algo].load(model_path, env=env)
        model.learn(n_timesteps, **kwargs)
    except KeyboardInterrupt:
        pass


    # Only save worker of rank 0 when using mpi
    if rank == 0:
        print("Saving to {}".format(save_path))

        model.save("{}/{}".format(save_path, env_id))

    if normalize:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(os.path.join(params_path, 'vecnormalize.pkl'))
        # Deprecated saving:
        # env.save_running_average(params_path)
