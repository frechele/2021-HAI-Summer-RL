import torch
from torch import nn, optim
from torch.distributions import Categorical

import numpy as np
import gym

LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.net(x)


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, device):
        self.device = device

        self.actor = ActorNet().to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

        self.critic = CriticNet().to(device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

    def get_action(self, state):
        policy = self.actor(state.to(self.device))

        m = Categorical(policy)
        action = m.sample()

        return action.item(), m.log_prob(action)

    def train(self, state, log_prob, next_state, reward, done):
        state, next_state = state.to(self.device), next_state.to(self.device)

        value, next_value = self.critic(state), self.critic(next_state)

        if done:
            target = torch.tensor(reward).to(device)
        else:
            target = reward + DISCOUNT_FACTOR * next_value

        adv = target - value

        self.actor_opt.zero_grad()
        actor_loss = -log_prob * adv.detach()
        actor_loss.backward()
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        critic_loss = (target.detach() - value) ** 2
        critic_loss.backward()
        self.critic_opt.step()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make('CartPole-v0')
    agent = Agent(device)

    total_rewards = []

    while True:
        obs = env.reset()
        obs = torch.FloatTensor(obs)

        total_reward = 0

        while True:
            action, log_prob = agent.get_action(obs)
            next_obs, reward, done, _ = env.step(action)

            total_reward += reward

            next_obs = torch.FloatTensor(next_obs)

            agent.train(obs, log_prob, next_obs, reward, done)

            if done:
                break

            obs = next_obs

        total_rewards.append(total_reward)

        m_reward = np.mean(total_rewards[-30:])
        print("done %d games, reward %.3f, avg_reward %.3f" %
              (len(total_rewards), total_reward, m_reward))

        if m_reward > 195:
            print("Solved in %d games!" % len(total_rewards))
            break
