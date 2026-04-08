from pathlib import Path
import warnings

try:
    import gym
    from gym.envs.registration import registry
except ImportError:
    import gymnasium as gym
    from gymnasium.envs.registration import registry
import numpy as np

try:
    import vmas
except ImportError:
    vmas = None
    warnings.warn(
        "VMAS is not installed. Install with `pip install 'vmas[gymnasium]'` to use vmas-* tasks."
    )


class VMASWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, env_name, render_mode="rgb_array", **kwargs):
        if vmas is None:
            raise ImportError(
                "VMAS is not installed. Install with `pip install 'vmas[gymnasium]'`."
            )

        kwargs = dict(kwargs)
        kwargs.pop("render_mode", None)
        self.render_mode = render_mode

        self._env = vmas.make_env(
            env_name,
            num_envs=1,
            continuous_actions=False,
            dict_spaces=False,
            terminated_truncated=True,
            wrapper="gymnasium",
            **kwargs,
        )
        self._env.render_mode = self.render_mode

        self.n_agents = self._env.unwrapped.n_agents
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def seed(self, seed=None):
        self._env.reset(seed=seed)
        return [seed]

    def reset(self, seed=None, options=None):
        reset_output = self._env.reset(seed=seed, options=options)
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            return reset_output
        return reset_output, {}

    def step(self, actions):
        obs, reward, terminated, truncated, info = self._env.step(actions)
        if isinstance(terminated, (list, tuple, np.ndarray)):
            terminated = bool(np.all(terminated))
        if isinstance(truncated, (list, tuple, np.ndarray)):
            truncated = bool(np.all(truncated))
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def _register_vmas_envs():
    if vmas is None:
        return

    envs = Path(vmas.__path__[0]).glob("scenarios/**/*.py")
    for env in envs:
        name = env.stem
        if "__" in name:
            continue

        env_id = f"vmas-{name}"
        if env_id in registry:
            continue

        gym.register(
            env_id,
            entry_point=f"{__name__}:VMASWrapper",
            kwargs={"env_name": name},
        )


_register_vmas_envs()
