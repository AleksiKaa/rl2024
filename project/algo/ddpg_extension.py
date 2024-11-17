from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
from .ddpg_agent import DDPGAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path


def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


class DDPGExtension(DDPGAgent):
    def __init__(self, config=None, n_step=10):
        super(DDPGExtension, self).__init__(config)

        self.name = "ddpg_extension"
        self.n_step = n_step
        self.random_transition = 5000  # Hyperparameter to be tuned

    def discount_rewards(self, rewards, r_0):
        # Calculate discounted rewards G
        gamma = self.gamma  # For discounting future timesteps
        discounted_reward = r_0
        for r_i in rewards:
            discounted_reward += r_i * gamma
            gamma *= self.gamma

        # Update discounting coeffiecient according to LNSS paper
        lnss_discount = (self.gamma - 1) / (gamma - 1)
        discounted_reward = lnss_discount * discounted_reward

        return discounted_reward

    def lnss_reward(self, batch, exp_buffer_ptr):
        # Unpack batch
        state = batch.state
        action = batch.action
        next_state = batch.next_state
        reward = batch.reward
        not_done = batch.not_done

        r_0 = reward[exp_buffer_ptr]
        reward_buffer = reward[exp_buffer_ptr + 1 :]
        discounted_reward = self.discount_rewards(reward_buffer, r_0)

        # Return current state transition with discounted reward
        return (
            state[exp_buffer_ptr],
            action[exp_buffer_ptr],
            next_state[exp_buffer_ptr],
            discounted_reward,
            not_done[exp_buffer_ptr],
        )

    def train_iteration(self):
        # start = time.perf_counter()
        # Run actual training
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()

        # Buffer for LNSS
        exp_buffer = ReplayBuffer(
            self.observation_space_dim,
            self.action_dim,
            int(float(self.max_episode_steps)),
        )
        exp_buffer_ptr = 0

        # Loop until finished
        while not done:

            # Sample action from policy
            action, _ = self.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))
            done_bool = float(done) if timesteps < self.max_episode_steps else 0

            # Record to replay buffer
            exp_buffer.add(obs, action, next_obs, reward, done_bool)

            # Get transitions from temp buffer
            batch = exp_buffer.get_all(device=self.device)

            # Enough data collected in temporary buffer, update permanent buffer
            if exp_buffer.size >= self.n_step:
                # Calculate LNSS reward with state transition
                state_0, action_0, state_1, r_prime, not_done_1 = self.lnss_reward(
                    batch, exp_buffer_ptr
                )

                # Update buffer pointer
                exp_buffer_ptr += 1

                # Finally append r^prime to experience replay buffer
                self.record(state_0, action_0, state_1, r_prime, not_done_1)

            if timesteps >= self.max_episode_steps:
                done = True

            if done:
                # Need to calculate rest of rewards
                while exp_buffer_ptr < exp_buffer.size:
                    # Calculate LNSS reward with state transition
                    state_0, action_0, state_1, r_prime, not_done_1 = self.lnss_reward(
                        batch, exp_buffer_ptr
                    )

                    # Update buffer pointer
                    exp_buffer_ptr += 1

                    # Finally append r^prime to experience replay buffer
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
