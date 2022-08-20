import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env
from rlkit.envs.ant import AntEnv

@register_env('ant-vel')
class AntVelEnv(AntEnv):

    def __init__(self, task={}, n_tasks=36, n_train_tasks=30, randomize_tasks=True, uniform_sample=False, **kwargs):
        self._task = task
        self._goal = 0.0
        self.uniform_sample = uniform_sample
        self.randomize_tasks = randomize_tasks
        self.train_idx = list(range(n_train_tasks))
        self.test_idx = list(range(n_train_tasks, n_tasks))
        self.tasks = self.sample_tasks(int(np.sqrt(n_tasks)))
        super(AntVelEnv, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        #forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)
        forward_vel = torso_velocity[:2]/self.dt
        forward_reward = -1.0 * np.sum(np.abs(forward_vel - self._goal))

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def sample_tasks(self, num_tasks):
        assert num_tasks == 6
        if not self.uniform_sample:
            velocity_x = np.random.uniform(-2.5, 2.5, size=(num_tasks,))
            velocity_y = np.random.uniform(-2.5, 2.5, size=(num_tasks,))
        else:
            X,Y = np.mgrid[-2.5:2.5:6j, -2.5:2.5:6j]
            velocities = np.vstack((X.flatten(), Y.flatten())).T
        tasks = [{'velocity': velocity} for velocity in velocities]
        if self.randomize_tasks:
            np.random.shuffle(tasks)
        return tasks
    
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
    
    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_vel = self._task['velocity']
        self._goal = self._goal_vel
        self.reset()
