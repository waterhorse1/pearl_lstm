import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env
from rlkit.envs.ant import AntEnv
from .ant_dir2d import *
from .ant_goal import *
from .ant_mass import *
from .ant_vel import *

@register_env('ant-ml4')
class AntML4lEnv(AntEnv):

    def __init__(self, n_tasks=2, task={}, task_num=None, randomize_tasks=True, uniform_sample=True):     
        self.env_list = [AntDirEnv(n_tasks=task_num['dir']['all_num'],n_train_tasks=task_num['dir']['train_num'], randomize_tasks=randomize_tasks, uniform_sample=uniform_sample), AntGoalEnv(n_tasks=task_num['goal']['all_num'],n_train_tasks=task_num['goal']['train_num'], randomize_tasks=randomize_tasks, uniform_sample=uniform_sample), AntMassEnv(n_tasks=task_num['mass']['all_num'],n_train_tasks=task_num['mass']['train_num'], randomize_tasks=randomize_tasks, uniform_sample=uniform_sample), AntVelEnv(n_tasks=task_num['vel']['all_num'],n_train_tasks=task_num['vel']['train_num'], randomize_tasks=randomize_tasks, uniform_sample=uniform_sample)]
        self.task_num_list = np.array([task_num['dir']['all_num'], task_num['goal']['all_num'], task_num['mass']['all_num'], task_num['vel']['all_num']])
        self.train_task_num_list = np.array([task_num['dir']['train_num'], task_num['goal']['train_num'], task_num['mass']['train_num'], task_num['vel']['train_num']])
        self.task_idx = 0
        self.idx = 0
        self.get_train_test_idx()
        self.get_taskinfo()
        super(AntML4lEnv, self).__init__()

    def idx2task(self, idx):
        for task_idx, task_num in enumerate(self.task_num_list):
            if idx - task_num < 0:
                break
            else:
                idx = idx - task_num
        
        return task_idx, idx
    
    def task2idx(self, task_idx, idx):
        idx = self.task_num_list[:task_idx].sum() + idx
        return idx
    
    def get_train_test_idx(self):
        train_idx = []
        test_idx = []
        idx = 0
        for train_num, task_num in zip(self.train_task_num_list, self.task_num_list):
            train_idx.extend(list(range(idx, idx+ train_num)))
            test_idx.extend(list(range(idx+ train_num, idx+ task_num)))
            idx += task_num
        self.train_idx = train_idx
        self.test_idx = test_idx
    
    def sample_train(self, num):
        # sample tasks
        idx_list = []
        infos_list = []
        task_idx = np.random.choice(len(self.env_list), num, replace=True)
        for task_id in task_idx:
            idx, infos = self.env_list[task_id].sample_train(num=1)
            infos_list.extend(infos)
            idx_list.extend(self.task2idx(task_id, idx))
            
        return idx_list, infos_list
            
    def sample_test(self, num):
        # sample tasks
        idx_list = []
        infos_list = []
        task_idx = np.random.choice(len(self.env_list), num, replace=True)
        for task_id in task_idx:
            idx, infos = self.env_list[task_id].sample_test(num=1)
            infos_list.extend(infos)
            idx_list.extend(self.task2idx(task_id, idx))
            
        return idx_list, infos_list
    
    def step(self, action):
        return self.env_list[self.task_idx].step(action)

    def get_all_task_idx(self):
        return range(sum(self.task_num_list))
    
    def get_taskinfo(self):
        task_info = []
        for idx in range(sum(self.task_num_list)):
            task_idx, sidx = self.idx2task(idx)
            print(task_idx, sidx)
            task_info.append(self.env_list[task_idx].tasks[sidx])
        self.task_info = task_info
        self.task_info_key = dict()
        key_id = 0
        for idx in range(sum(self.task_num_list)):
            env_task_info =  self.task_info[idx]
            key = list(env_task_info.keys())[0]
            if key in self.task_info_key.keys():
                pass
            else:
                self.task_info_key[key] = key_id
                key_id += 1
        print(self.task_info, self.task_info_key)

    def reset_task(self, idx):
        task_idx, idx = self.idx2task(idx)
        self.env_list[task_idx].reset_task(idx)
        self.task_idx = task_idx
        self.idx = idx