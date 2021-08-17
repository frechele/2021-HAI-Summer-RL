import torch
from torch import nn, optim
from torch.distributions import Categorical

import numpy as np
import gym

LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, device):
        self.device = device
        self.net = PolicyNet().to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

        self.log_probs = []
        self.rewards = []

    def get_action(self, state):
        policy = self.net(state.to(self.device))

        m = Categorical(policy)
        action = m.sample()

        self.log_probs.append(m.log_prob(action))
        return action.item()

    def train(self):
        R = 0
        losses = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + DISCOUNT_FACTOR * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)

        for log_prob, R in zip(self.log_probs, returns):
            losses.append(-log_prob * R)

        self.opt.zero_grad()

        loss = sum(losses)
        loss.backward()

        self.opt.step()

        self.log_probs = []
        self.rewards = []


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
            action = agent.get_action(obs)
            next_obs, reward, done, _ = env.step(action)

            agent.rewards.append(reward)
            total_reward += reward

            if done:
                break

            obs = torch.FloatTensor(next_obs)

        total_rewards.append(total_reward)
        agent.train()

        m_reward = np.mean(total_rewards[-30:])
        print("done %d games, reward %.3f, avg_reward %.3f" %
              (len(total_rewards), total_reward, m_reward))

        if m_reward > 195:
            print("Solved in %d games!" % len(total_rewards))
            break
