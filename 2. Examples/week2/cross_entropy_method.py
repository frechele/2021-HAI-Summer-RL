import gym
from collections import namedtuple
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim


HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []

    obs = env.reset()
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = F.softmax(net(obs_v), dim=1)
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)

        next_obs, reward, done, _ = env.step(action)

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))

            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs


def filter_batch(batch, percentile):
    disc_rewards = list(
        map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound


if __name__ == "__main__":
    env = DiscreteOneHotWrapper(gym.make('FrozenLake-v0', is_slippery=False))

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = nn.Sequential(
        nn.Linear(obs_size, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, n_actions)
    )

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_bound = filter_batch(
            full_batch + batch, PERCENTILE)

        if not full_batch:
            continue

        obs_v = torch.FloatTensor(obs)
        acts_v = torch.LongTensor(acts)
        full_batch = full_batch[-500:]

        optimizer.zero_grad()
        action_scores_v = net(obs_v)

        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (
            iter_no, loss_v.item(), reward_mean, reward_bound, len(full_batch)))

        if reward_mean > 0.8:
            print("Solved!")
            break
