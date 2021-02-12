import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from gym import Env
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points


class PointEnvBarrier(Env):
    """
    2D point robot must navigate outside a ball, with door at different locations
    over a semi-circle
     - tasks sampled from unit half-circle
     - reward is 1 outside the ball and 0 inside
    """

    def __init__(self,
                 max_episode_steps=20,
                 n_tasks=2,
                 modify_init_state_dist=True,
                 **kwargs):

        self._max_episode_steps = max_episode_steps
        self.step_count = 0
        self.modify_init_state_dist = modify_init_state_dist

        # np.random.seed(1337)
        self.radius = 1.0   # radius of barrier
        self.door_width_angle = np.pi / 8   # 8 full doors with no "intersection" to cover semi-circle
        self.door_width = np.sqrt(1+1-2*np.cos(self.door_width_angle/2))
        angles = np.random.uniform(0, np.pi, size=n_tasks)
        # xs = radius * np.cos(angles)
        # ys = radius * np.sin(angles)
        # goals = np.stack([xs, ys], axis=1)
        # np.random.shuffle(goals)
        # goals = goals.tolist()

        self.goals = angles.tolist()
        # self.goals_cartesian = goals

        # construct sphere barrier
        self._barrier = Point(0, 0).buffer(1.0).difference(Point(0, 0).buffer(0.999))

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        if idx is not None:
            self._goal = np.array(self.goals[idx])
        self.reset()

    def set_goal(self, goal):
        self._goal = np.asarray(goal)

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        self.step_count = 0
        if self.modify_init_state_dist:     # at random inside upper half-ball
            # self._state = np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-0.5, 1.5)])
            random_angle = np.random.uniform(-np.pi, np.pi)
            self._state = np.random.uniform(0, self.radius) * np.array([np.sin(random_angle), np.cos(random_angle)])
        else:
            self._state = np.array([0, 0])
        return self._get_obs()

    def reset(self):
        self.step_count = 0
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        lookahead_state = self._state + action
        # check intersection of path with circle
        path = LineString([tuple(self._state), tuple(lookahead_state)])
        intersection = path.intersection(self._barrier)
        if not intersection.is_empty:
            # find nearest intersection point
            nearest_intersection = nearest_points(intersection, Point(self._state))[0]
            intersection_point = np.array([nearest_intersection.x, nearest_intersection.y])
            intersection_angle = np.arctan2(intersection_point[1], intersection_point[0])
            if (self._goal - (self.door_width_angle / 2)) <= intersection_angle \
                    <= (self._goal + (self.door_width_angle / 2)):
                self._state = lookahead_state
            else:
                self._state = (lookahead_state / np.linalg.norm(lookahead_state)) * 0.99
                # self._state = self._state + 0.999 * (intersection_point - self._state)
        else:
            self._state = lookahead_state

        reward = self.reward(self._state)

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True
        else:
            done = False

        ob = self._get_obs()
        return ob, reward, done, dict()

    def reward(self, state, action=None):
        return (np.linalg.norm(state) > self.radius).astype(np.float32)

    def is_goal_state(self):
        return np.linalg.norm(self._state) > self.radius

    def plot_env(self):
        ax = plt.gca()
        # plot half circle and goal position
        #angles = np.linspace(0, np.pi, num=100)
        x, y = np.cos(self._goal), np.sin(self._goal)
        #plt.plot(x, y, color='k')
        # fix visualization
        plt.axis('scaled')
        # ax.set_xlim(-1.25, 1.25)
        ax.set_xlim(-2, 2)
        # ax.set_ylim(-0.25, 1.25)
        ax.set_ylim(-1, 2)
        plt.xticks([])
        plt.yticks([])
        circle = plt.Circle((x, y), radius=self.door_width, alpha=0.3)
        circle1 = plt.Circle((0, 0), 1, edgecolor='black', facecolor='w')
        ax.add_artist(circle1)
        ax.add_artist(circle)


    def plot_behavior(self, observations, plot_env=True, **kwargs):
         if plot_env:  # whether to plot circle and goal pos..(maybe already exists)
             self.plot_env()
         # visualise behaviour, current position, goal
         plt.plot(observations[1:, 0], observations[1:, 1], **kwargs)
