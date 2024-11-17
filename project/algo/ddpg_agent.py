from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path


def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


class DDPGAgent(BaseAgent):
    def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "ddpg"
        state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action
        self.lr = self.cfg.lr

        # Policy
        self.pi = Policy(state_dim, self.action_dim, self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))

        # Critic
        self.q = Critic(state_dim, self.action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=float(self.lr))

        self.buffer = ReplayBuffer(
            state_dim, self.action_dim, max_size=int(float(self.cfg.buffer_size))
        )

        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau

        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0
        self.random_transition = 5000  # collect 5k random data for better exploration
        self.max_episode_steps = self.cfg.max_episode_steps

    def update(self):
        """After collecting one trajectory, update the pi and q for #transition times:"""
        info = {}
        update_iter = (
            self.buffer_ptr - self.buffer_head
        )  # update the network once per transition

        if self.buffer_ptr > self.random_transition:  # update once we have enough data
            for _ in range(update_iter):
                info = self._update()

        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
        return info

    @torch.no_grad()
    def get_action(self, observation, evaluation=False, expl_noise=0.3):
        if observation.ndim == 1:
            observation = observation[None]  # add the batch dimension
        x = torch.from_numpy(observation).float().to(self.device)

        if (
            self.buffer_ptr < self.random_transition and not evaluation
        ):  # collect random trajectories for better exploration.
            action = torch.rand(self.action_dim, device=self.device)
        else:
            action = self.pi(x)

            if not evaluation:
                action += torch.normal(
                    torch.zeros(action.size()),
                    torch.full(action.size(), self.max_action * expl_noise),
                ).to(self.device)
                action.clip(-self.max_action, self.max_action)

        return action, {}  # just return a positional value

    def calculate_target(self, batch):
        next_state = batch.next_state
        reward = batch.reward
        not_done = batch.not_done

        with torch.no_grad():
            next_action = self.pi_target(next_state)
            q_tar = self.q_target(next_state, next_action)
            target_Q = reward + self.gamma * not_done * q_tar
        return target_Q

    def calculate_critic_loss(self, current_Q, target_Q):
        return F.mse_loss(current_Q, target_Q)

    def calculate_actor_loss(self, batch):
        state = batch.state
        actor_loss = -torch.mean(self.q(state, self.pi(state)))
        return actor_loss

    def _update(
        self,
    ):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)
        state = batch.state
        action = batch.action
        # next_state = batch.next_state
        # reward = batch.reward
        # not_done = batch.not_done

        current_Q = self.q(state, action)

        target_Q = self.calculate_target(batch)
        critic_loss = self.calculate_critic_loss(current_Q, target_Q)

        # optimize the critic
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        actor_loss = self.calculate_actor_loss(batch)

        # optimize the actor
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        # update the target q and pi using soft_update_params()
        self.soft_update_params(self.pi, self.pi_target, self.tau)
        self.soft_update_params(self.q, self.q_target, self.tau)

        return {}

    def train_iteration(self):
        # start = time.perf_counter()
        # Run actual training
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:

            # Sample action from policy
            action, _ = self.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)
            done_bool = float(done) if timesteps < self.max_episode_steps else 0
            self.record(obs, action, next_obs, reward, done_bool)

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        # s = time.perf_counter()
        info = self.update()
        # e = time.perf_counter()

        # Return stats of training
        info.update(
            {
                "episode_length": timesteps,
                "ep_reward": reward_sum,
            }
        )

        end = time.perf_counter()
        return info

    def train(self):
        if self.cfg.save_logging:
            L = cu.Logger()  # create a simple logger to record stats
        start = time.perf_counter()
        total_step = 0
        run_episode_reward = []
        log_count = 0

        for ep in range(self.cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = self.train_iteration()
            train_info.update({"episodes": ep})
            total_step += train_info["episode_length"]
            train_info.update({"total_step": total_step})
            run_episode_reward.append(train_info["ep_reward"])

            if total_step > self.cfg.log_interval * log_count:
                average_return = sum(run_episode_reward) / len(run_episode_reward)
                if not self.cfg.silent:
                    print(
                        f"Episode {ep} Step {total_step} finished. Average episode return: {average_return}"
                    )
                if self.cfg.save_logging:
                    train_info.update({"average_return": average_return})
                    L.log(**train_info)
                run_episode_reward = []
                log_count += 1

        if self.cfg.save_model:
            self.save_model()

        logging_path = str(self.logging_dir) + "/logs"
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()

        end = time.perf_counter()
        train_time = (end - start) / 60
        print("------ Training Finished ------")
        print(f"Total traning time is {train_time}mins")

    def soft_update_params(self, m, m_target, tau):
        """Update slow-moving average of online network (target network) at rate tau."""
        with torch.no_grad():
            for p, p_target in zip(m.parameters(), m_target.parameters()):
                p_target.data.lerp_(p.data, tau)

    def record(self, state, action, next_state, reward, done):
        """Save transitions to the buffer."""
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)

    def load_model(self):
        # define the save path, do not modify
        filepath = str(self.model_dir) + "/model_parameters_" + str(self.seed) + ".pt"

        d = torch.load(filepath)
        self.q.load_state_dict(d["q"])
        self.q_target.load_state_dict(d["q_target"])
        self.pi.load_state_dict(d["pi"])
        self.pi_target.load_state_dict(d["pi_target"])

    def save_model(self):
        # define the save path, do not modify
        filepath = str(self.model_dir) + "/model_parameters_" + str(self.seed) + ".pt"

        torch.save(
            {
                "q": self.q.state_dict(),
                "q_target": self.q_target.state_dict(),
                "pi": self.pi.state_dict(),
                "pi_target": self.pi_target.state_dict(),
            },
            filepath,
        )
        print("Saved model to", filepath, "...")
