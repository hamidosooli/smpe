from functools import partial
import pretrained
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim as gym_flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit

try:
    from gymnasium.spaces.utils import flatdim as gymnasium_flatdim
    from gymnasium.spaces.utils import flatten as gymnasium_flatten
except ImportError:
    gymnasium_flatdim = None
    gymnasium_flatten = None

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# from .traffic_junction import Traffic_JunctionEnv
# REGISTRY["traffic_junction"] = partial(env_fn, env=Traffic_JunctionEnv)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


def _space_flatdim(space):
    try:
        return gym_flatdim(space)
    except Exception:
        if gymnasium_flatdim is not None:
            return gymnasium_flatdim(space)
        if hasattr(space, "shape") and space.shape is not None:
            return int(np.prod(space.shape))
        if hasattr(space, "n"):
            return int(space.n)
        raise


def _space_flatten(space, obs):
    try:
        return spaces.flatten(space, obs)
    except Exception:
        if gymnasium_flatten is not None:
            return gymnasium_flatten(space, obs)
        return np.asarray(obs, dtype=np.float32).reshape(-1)


def _all_done(done):
    if isinstance(done, (list, tuple, np.ndarray)):
        return all(done)
    return bool(done)


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def reset(self, **kwargs):
        reset_output = super().reset(**kwargs)
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            observation, _ = reset_output
            return observation
        return reset_output

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        step_output = self.env.step(action)
        if len(step_output) == 5:
            observation, reward, terminated, truncated, info = step_output
            if isinstance(terminated, (list, tuple, np.ndarray)):
                done = [bool(t) or bool(tr) for t, tr in zip(terminated, truncated)]
            else:
                done = bool(terminated) or bool(truncated)
        else:
            observation, reward, done, info = step_output
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not _all_done(done)
            if isinstance(done, (list, tuple, np.ndarray)):
                done = len(observation) * [True]
            else:
                done = True
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            obs_dim = _space_flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(obs_dim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                _space_flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, **kwargs):
        self.episode_limit = time_limit
        self._episode_steps = 0

        try:
            self._env = gym.make(f"{key}", disable_env_checker=True)
        except TypeError:
            self._env = gym.make(f"{key}")

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = getattr(self._env, "n_agents", len(self._env.action_space))
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=_space_flatdim)
        self.longest_observation_space = max(
            self._env.observation_space, key=_space_flatdim
        )
        self.longest_observation_dim = _space_flatdim(self.longest_observation_space)

        self._seed = kwargs.get("seed")
        if self._seed is not None:
            try:
                self._env.reset(seed=self._seed)
            except TypeError:
                if hasattr(self._env, "seed"):
                    self._env.seed(self._seed)

    def _pad_obs(self, obs):
        return [
            np.pad(
                np.asarray(o, dtype=np.float32).reshape(-1),
                (0, self.longest_observation_dim - np.asarray(o).size),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        step_output = self._env.step(actions)
        if len(step_output) == 5:
            self._obs, reward, terminated, truncated, _ = step_output
            if isinstance(terminated, (list, tuple, np.ndarray)):
                if isinstance(truncated, (list, tuple, np.ndarray)):
                    done = [bool(t) or bool(tr) for t, tr in zip(terminated, truncated)]
                else:
                    done = [bool(t) or bool(truncated) for t in terminated]
            else:
                done = bool(terminated) or bool(truncated)
        else:
            self._obs, reward, done, _ = step_output

        self._episode_steps += 1
        if self._episode_steps >= self.episode_limit:
            if isinstance(done, (list, tuple, np.ndarray)):
                done = [True] * len(done)
            else:
                done = True

        self._obs = self._pad_obs(self._obs)

        if isinstance(reward, (list, tuple, np.ndarray)):
            reward_sum = float(np.sum(reward))
        else:
            reward_sum = float(reward)

        return reward_sum, _all_done(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.longest_observation_dim

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * self.longest_observation_dim

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = _space_flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (_space_flatdim(self.longest_action_space) - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return _space_flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._episode_steps = 0
        reset_output = self._env.reset()
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            self._obs, _ = reset_output
        else:
            self._obs = reset_output
        self._obs = self._pad_obs(self._obs)
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}


REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
