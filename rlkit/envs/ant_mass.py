import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env
from rlkit.envs.ant import AntEnv

@register_env('ant-mass')
class AntMassEnv(AntEnv):

    def __init__(self, task={}, n_tasks=30, n_train_tasks=24, randomize_tasks=True, uniform_sample=False, **kwargs):
        self._task = task
        self.mass_scale = 0.8
        self._goal = 0.
        self.uniform_sample = uniform_sample
        self.randomize_tasks = randomize_tasks
        self.train_idx = list(range(n_train_tasks))
        self.test_idx = list(range(n_train_tasks, n_tasks))
        self.tasks = self.sample_tasks(n_tasks)
        super(AntMassEnv, self).__init__()
        self.original_mass = np.copy(self.model.body_mass)
        #super(AntMassEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, a):
        self.xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        reward_ctrl = -0.005 * np.square(a).sum()
        reward_run = (xposafter - self.xposbefore) / self.dt
        reward_contact = 0.0
        reward_survive = 0.05
        reward = reward_run + reward_ctrl + reward_contact + reward_survive
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            reward_forward=reward_run,
            reward_ctrl=reward_ctrl,
            reward_contact=reward_contact,
            reward_survive=reward_survive
        )
    def get_all_task_idx(self):
        return range(len(self.tasks))
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.xposbefore = self.get_body_com("torso")[0]
        #random_index = self.np_random.randint(len(self.mass_scale_set))
        #self.mass_scale = self.mass_scale_set[random_index]

        #random_index = self.np_random.randint(len(self.damping_scale_set))
        #self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()
        return self._get_obs()
    def change_env(self):
        mass = np.copy(self.original_mass)
        #damping = np.copy(self.original_damping)
        mass *= self.mass_scale
        #damping *= self.damping_scale

        self.model.body_mass[:] = mass
    def sample_tasks(self, num_tasks):
        if not self.uniform_sample:
            masses = np.random.uniform(0.2, 1.0, size=(num_tasks,))
        else:
            masses = np.linspace(0.2, 1.0, num_tasks)
        
        # 0 for padding
        tasks = [{'mass': np.array([mass, 0])} for mass in masses]
        
        if self.randomize_tasks:
            np.random.shuffle(tasks)
        return tasks
    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._mass = self._task['mass']
        self.mass_scale = self._mass
        self.reset()
        
    def sample_train(self, num):
        idx = np.random.choice(self.train_idx, num)
        infos = []
        for i in idx:
            infos.append(self.tasks[i])
        return idx, infos
    
    def sample_test(self, num):
        idx = np.random.choice(self.test_idx, num)
        infos = []
        for i in idx:
            infos.append(self.tasks[i])
        return idx, infos
    
    def get_all_task_idx(self):
        return range(len(self.tasks))
