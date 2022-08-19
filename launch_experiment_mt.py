"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianModularPolicy
from rlkit.torch.network_soft_module import FlattenModularGatedCascadeCondNet, MLPBase
from rlkit.torch.sac.mtsac import PEARLSoftActorCritic
from rlkit.torch.sac.mtagent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default_mt import default_config


def experiment(variant):

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    print([t.tasks for t in env.env_list])
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    variant['net']['base_type'] = MLPBase
    #encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    use_pure_one_hot = variant['algo_params']['use_pure_one_hot'] 
    print(variant['algo_params']['use_pure_one_hot'] )
    '''
    context_encoder = encoder_model(
        hidden_sizes=[128, 128, 128],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    '''
    qf1 = FlattenModularGatedCascadeCondNet(
        input_shape=obs_dim + action_dim + 1,
        em_input_shape=len(env.train_idx) if use_pure_one_hot else len(env.task_info_key.keys()),
        output_shape=1,
        **variant['net']
    )
    qf2 = FlattenModularGatedCascadeCondNet(
        #hidden_sizes=[net_size, net_size, net_size],
        input_shape=obs_dim + action_dim + 1,
        em_input_shape=len(env.train_idx) if use_pure_one_hot else len(env.task_info_key.keys()),
        output_shape=1,
        **variant['net']
    )
    vf = FlattenModularGatedCascadeCondNet(
        input_shape=obs_dim + 1,
        em_input_shape=len(env.train_idx) if use_pure_one_hot else len(env.task_info_key.keys()),
        output_shape=1,
        **variant['net']
    )
    modular_model = FlattenModularGatedCascadeCondNet(
        input_shape=obs_dim + 1,
        em_input_shape=len(env.train_idx) if use_pure_one_hot else len(env.task_info_key.keys()),
        output_shape=int(variant['net']['hidden_shapes'][-1]/2), #200
        **variant['net']
    )
    policy = TanhGaussianModularPolicy(
        modular_model=modular_model,
        last_hidden_size=int(variant['net']['hidden_shapes'][-1]/2),
        action_dim=action_dim)

    agent = PEARLAgent(
        latent_dim,
        policy,
        env,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(range(3)),
        eval_tasks=list(range(3)),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def setup_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--seed', default=0)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, seed, debug):
    setup_seed(seed)
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    variant['seed'] = seed
    experiment(variant)

if __name__ == "__main__":
    main()

