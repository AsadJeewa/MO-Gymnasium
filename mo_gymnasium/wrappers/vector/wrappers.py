"""Vector wrappers."""
import time
from copy import deepcopy
from typing import Any, Iterator

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.vector import SyncVectorEnv
from gymnasium.vector.utils import concatenate, iterate
from gymnasium.vector.vector_env import ArrayType, VectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics


# class MONormalizeReward(gym.Wrapper, gym.utils.RecordConstructorArgs):
#     """Wrapper to normalize the reward component at index idx. Does not touch other reward components."""
#
#     def __init__(self, env: gym.Env, idx: int, gamma: float = 0.99, epsilon: float = 1e-8):
#         """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
#
#         Args:
#             env (env): The environment to apply the wrapper
#             idx (int): the index of the reward to normalize
#             epsilon (float): A stability parameter
#             gamma (float): The discount factor that is used in the exponential moving average.
#         """
#         gym.utils.RecordConstructorArgs.__init__(self, idx=idx, gamma=gamma, epsilon=epsilon)
#         gym.Wrapper.__init__(self, env)
#         self.idx = idx
#         self.num_envs = getattr(env, "num_envs", 1)
#         self.is_vector_env = getattr(env, "is_vector_env", False)
#         self.return_rms = RunningMeanStd(shape=())
#         self.returns = np.zeros(self.num_envs)
#         self.gamma = gamma
#         self.epsilon = epsilon
#
#     def step(self, action: ActType):
#         """Steps through the environment, normalizing the rewards returned.
#
#         Args:
#             action: action to perform
#         Returns: obs, normalized_rewards, terminated, truncated, infos
#         """
#         obs, rews, terminated, truncated, infos = self.env.step(action)
#         # Extracts the objective value to normalize
#         to_normalize = rews[self.idx]
#         if not self.is_vector_env:
#             to_normalize = np.array([to_normalize])
#         self.returns = self.returns * self.gamma + to_normalize
#         # Defer normalization to gym implementation
#         to_normalize = self.normalize(to_normalize)
#         self.returns[terminated] = 0.0
#         if not self.is_vector_env:
#             to_normalize = to_normalize[0]
#         # Injecting the normalized objective value back into the reward vector
#         rews[self.idx] = to_normalize
#         return obs, rews, terminated, truncated, infos
#
#     def normalize(self, rews):
#         """Normalizes the rewards with the running mean rewards and their variance.
#
#         Args:
#             rews: rewards
#         Returns: the normalized reward
#         """
#         self.return_rms.update(self.returns)
#         return rews / np.sqrt(self.return_rms.var + self.epsilon)
#
#
class MOSyncVectorEnv(SyncVectorEnv):
    """Vectorized environment that serially runs multiple environments."""

    def __init__(
        self,
        env_fns: Iterator[callable],
        copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: env constructors
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.
        """
        SyncVectorEnv.__init__(self, env_fns, copy=copy)
        # Just overrides the rewards memory to add the number of objectives
        self.reward_space = self.envs[0].unwrapped.reward_space
        self._rewards = np.zeros(
            (
                self.num_envs,
                self.reward_space.shape[0],
            ),
            dtype=np.float32,
        )

    def step(self, actions: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        actions = iterate(self.action_space, actions)

        observations, infos = [], {}
        for i, action in enumerate(actions):
            if self._autoreset_envs[i]:
                env_obs, env_info = self.envs[i].reset()

                self._rewards[i] = np.zeros(self.reward_space.shape[0])  # This overrides Gymnasium's implem
                self._terminations[i] = False
                self._truncations[i] = False
            else:
                (env_obs, self._rewards[i], self._terminations[i], self._truncations[i], env_info,) = self.envs[
                    i
                ].step(action)

            observations.append(env_obs)
            infos = self._add_info(infos, env_info, i)

        # Concatenate the observations
        self._observations = concatenate(self.single_observation_space, observations, self._observations)
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            deepcopy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )


class MORecordEpisodeStatistics(RecordEpisodeStatistics):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of any episode within the vectorized env, the statistics of the episode
    will be added to ``info`` using the key ``episode``, and the ``_episode`` key
    is used to indicate the environment index which has a terminated or truncated episode.

     For a vectorized environments the output will be in the form of (be careful to first wrap the env into vector before applying MORewordStatistics)::

        >>> infos = { # doctest: +SKIP
        ...     "episode": {
        ...         "r": "<array of cumulative reward for each done sub-environment (2d array, shape (num_envs, dim_reward))>",
        ...         "dr": "<array of discounted reward for each done sub-environment (2d array, shape (num_envs, dim_reward))>",
        ...         "l": "<array of episode length for each done sub-environment (array)>",
        ...         "t": "<array of elapsed time since beginning of episode for each done sub-environment (array)>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(
        self,
        env: VectorEnv,
        gamma: float = 1.0,
        buffer_length: int = 100,
        stats_key: str = "episode",
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            gamma: The discount factor
            buffer_length: The size of the buffers :attr:`return_queue`, :attr:`length_queue` and :attr:`time_queue`
            stats_key: The info key to save the data
        """
        gym.utils.RecordConstructorArgs.__init__(self, buffer_length=buffer_length, stats_key=stats_key)
        RecordEpisodeStatistics.__init__(self, env, buffer_length=buffer_length, stats_key=stats_key)
        self.reward_dim = self.env.unwrapped.reward_space.shape[0]
        self.rewards_shape = (self.num_envs, self.reward_dim)
        self.gamma = gamma

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)

        # CHANGE: Here we just override the standard implementation to extend to MO
        self.episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)
        self.disc_episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)

        return obs, info

    def step(self, actions: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        assert isinstance(
            infos, dict
        ), f"`vector.RecordEpisodeStatistics` requires `info` type to be `dict`, its actual type is {type(infos)}. This may be due to usage of other wrappers in the wrong order."

        self.episode_returns[self.prev_dones] = 0
        self.episode_lengths[self.prev_dones] = 0
        self.episode_start_times[self.prev_dones] = time.perf_counter()
        self.episode_returns[~self.prev_dones] += rewards[~self.prev_dones]
        self.episode_lengths[~self.prev_dones] += 1

        # CHANGE: The discounted returns are also computed here
        self.disc_episode_returns += rewards * np.repeat(self.gamma**self.episode_lengths, self.reward_dim).reshape(
            self.episode_returns.shape
        )

        self.prev_dones = dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if self._stats_key in infos or f"_{self._stats_key}" in infos:
                raise ValueError(f"Attempted to add episode stats when they already exist, info keys: {list(infos.keys())}")
            else:
                # CHANGE to handle the vectorial reward and do deepcopies
                episode_return = np.zeros(self.rewards_shape, dtype=np.float32)
                disc_episode_return = np.zeros(self.rewards_shape, dtype=np.float32)

                for i in range(self.num_envs):
                    if dones[i]:
                        episode_return[i] = np.copy(self.episode_returns[i])
                        disc_episode_return[i] = np.copy(self.disc_episode_returns[i])

                episode_time_length = np.round(time.perf_counter() - self.episode_start_times, 6)
                infos[self._stats_key] = {
                    "r": np.where(dones, self.episode_returns, np.zeros(self.rewards_shape, dtype=np.float32)),
                    "dr": np.where(dones, self.disc_episode_returns, np.zeros(self.rewards_shape, dtype=np.float32)),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(dones, episode_time_length, 0.0),
                }
                infos[f"_{self._stats_key}"] = dones

            self.episode_count += num_dones

            for i in np.where(dones):
                self.time_queue.extend(episode_time_length[i])
                self.return_queue.extend(self.episode_returns[i])
                self.length_queue.extend(self.episode_lengths[i])

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
