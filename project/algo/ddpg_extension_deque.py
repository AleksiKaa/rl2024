from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
from .ddpg_agent import DDPGAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path
from collections import deque


def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


class DDPGExtension(DDPGAgent):
    def __init__(self, config=None, n_step=10):
        super(DDPGExtension, self).__init__(config)

        self.name = "ddpg_extension"
        self.n_step = n_step
        self.random_transition = 5000  # Hyperparameter to be tuned
        self.exp_buffer = deque()  # Queue for storing experiences

    def lnss_reward(self):
        # Unpack transition
        state, action, next_state, reward, not_done = self.exp_buffer.popleft()

        # Calculate discounted rewards G
        gamma = self.gamma  # For discounting future timesteps
        discounted_reward = reward
        for _, _, _, r_i, _ in self.exp_buffer:
            discounted_reward += r_i * gamma
            gamma *= self.gamma

        # Update discounting coeffiecient according to LNSS paper
        lnss_discount = (self.gamma - 1) / (gamma - 1)
        discounted_reward = lnss_discount * discounted_reward

        # Return current state transition with discounted reward
        return (
            state,
            action,
            next_state,
            discounted_reward,
            not_done,
        )

    def train_iteration(self):
        # start = time.perf_counter()
        # Run actual training
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()

        # Loop until finished
        while not done:
            # Sample action from policy
            action, _ = self.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))
            done_bool = float(done) if timesteps < self.max_episode_steps else 0

            # Record to experience buffer
            self.exp_buffer.append((obs, action, next_obs, reward, done_bool))

            # Enough data collected in temporary buffer, update permanent buffer
            if len(self.exp_buffer) >= self.n_step:
                # Calculate LNSS reward with state transition
                state_0, action_0, state_1, r_prime, not_done_1 = self.lnss_reward()

                # Append r^prime to experience replay buffer
                self.record(state_0, action_0, state_1, r_prime, not_done_1)

            if timesteps >= self.max_episode_steps:
                done = True

            # Need to calculate rest of rewards
            if done:
                while len(self.exp_buffer):
                    # Calculate LNSS reward with state transition
                    state_0, action_0, state_1, r_prime, not_done_1 = self.lnss_reward()
                    # Append r^prime to experience replay buffer
                    self.record(state_0, action_0, state_1, r_prime, not_done_1)

            # Update observation, episode rewards and timestep
            obs = next_obs.copy()
            reward_sum += reward
            timesteps += 1

        # Update the policy after one episode
        # s = time.perf_counter()
        info = self.update()
        # e = time.perf_counter()

        # Return stats of training
        info.update({"episode_length": timesteps, "ep_reward": reward_sum})
        return info
