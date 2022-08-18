import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config

import os
import json
def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

variant = default_config
config = './configs/cheetah-ml3.json'
if config:
    with open(os.path.join(config)) as f:
        exp_params = json.load(f)
    variant = deep_update_dict(exp_params, variant)

env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
print(env.idx2task(2), env.idx2task(31), env.idx2task(32), env.task2idx(1,29), env.task2idx(2,0), env.train_idx, env.test_idx)