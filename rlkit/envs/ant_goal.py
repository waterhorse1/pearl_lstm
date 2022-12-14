import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env
from rlkit.envs.ant import AntEnv

# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-goal')
class AntGoalEnv(AntEnv):
    def __init__(self, task={}, n_tasks=30, n_train_tasks=24, randomize_tasks=True, uniform_sample=False, **kwargs):
        self._task = task
        self._goal = 0.
        self.uniform_sample = uniform_sample
        self.randomize_tasks = randomize_tasks
        self.train_idx = list(range(n_train_tasks))
        self.test_idx = list(range(n_train_tasks, n_tasks))
        self.tasks = self.sample_tasks(n_tasks)
        super(AntGoalEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        #print(ob.shape)
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self, num_tasks):
        if not self.uniform_sample:
            a = np.random.random(num_tasks) * 2 * np.pi
            r = 3 * np.random.random(num_tasks) ** 0.5
        else:
            a = np.linspace(0.05, 1, num_tasks) * 2 * np.pi
            r = 3 * np.linspace(0.05, 1, num_tasks) ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        if self.randomize_tasks:
            np.random.shuffle(tasks)
        return tasks
    
    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
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

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
