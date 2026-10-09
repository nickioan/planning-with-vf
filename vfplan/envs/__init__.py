from vfplan.envs.walker3d import TASKS, Walker3DPlanningEnv

# Episode lengths in control steps (60 Hz).
# Data collection, VF training and the obstacle-task evaluation use 1500-step episodes;
# the platform, hurdle and composition evaluations use 350-step episodes, long enough
# to traverse the first few objects.
EPISODE_STEPS = 1500
SHORT_EPISODE_STEPS = 350


class TimeLimit:
    """Ends episodes after `max_episode_steps` steps (sets info["TimeLimit.truncated"])."""

    def __init__(self, env, max_episode_steps):
        self.env = env
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    @property
    def unwrapped(self):
        return self.env

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_env(task, render=False, max_episode_steps=EPISODE_STEPS, **env_kwargs):
    """Creates a `Walker3DPlanningEnv` with a time limit."""
    return TimeLimit(Walker3DPlanningEnv(task=task, render=render, **env_kwargs), max_episode_steps)


__all__ = ["TASKS", "Walker3DPlanningEnv", "make_env", "TimeLimit", "EPISODE_STEPS", "SHORT_EPISODE_STEPS"]
