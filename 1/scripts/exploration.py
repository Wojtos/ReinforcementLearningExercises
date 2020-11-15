from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import sys
import random as rand

"""
Implement the different exploration strategies.

  * mab is a MAB (MultiArmedBandit) object as defined below
  * epsilon is a scalar, which influences the amount of random actions
  * schedule is a callable decaying epsilon

You can get the approximated Q-values via mab.bandit_q_values and the different
counters for the bandits via mab.bandit_counters. mab.no_actions gives you the number
of arms.
"""


class Bandit:
    def __init__(self, bias, q_value=0, counter=0):
        self.bias = bias
        self.q_value = q_value
        self.counter = counter

    def pull(self):
        self.counter += 1
        reward = np.clip(self.bias + np.random.uniform(), 0, 1)
        self.q_value = self.q_value + 1 / self.counter * (reward - self.q_value)
        return reward


class MAB:
    def __init__(self, best_action_value, no_rounds, *bandits):
        self.bandits = bandits
        self._no_actions = len(bandits)
        self.step_counter = 0
        self.best_action_value = best_action_value
        self.no_rounds = no_rounds

    def pull(self, action):
        self.step_counter += 1
        return self.bandits[action].pull(), self.bandits[action].q_value

    def run(self, exploration_strategy, **strategy_parameters):
        regrets = []
        rewards = []
        for i in range(no_rounds):
            if (i + 1) % 100 == 0:
                print("\rRound {}/{}".format(i + 1, self.no_rounds), end="")
                sys.stdout.flush()
            action = exploration_strategy(self, **strategy_parameters)
            reward, q = self.pull(action)
            regret = self.best_action_value - q
            regrets.append(regret)
            rewards.append(reward)
        return regrets, rewards

    @property
    def bandit_counters(self):
        return np.array([bandit.counter for bandit in self.bandits])

    @property
    def bandit_q_values(self):
        return np.array([bandit.q_value for bandit in self.bandits])

    @property
    def no_actions(self):
        return self._no_actions


def plot(regrets):
    for strategy, regret in regrets.items():
        total_regret = np.cumsum(regret)
        plt.ylabel('Total Regret')
        plt.xlabel('Rounds')
        plt.plot(np.arange(len(total_regret)), total_regret, label=strategy)
    plt.legend()
    plt.savefig('regret.pdf', bbox_inches='tight')


def random(mab: MAB):
    return rand.randrange(0, mab.no_actions)
    # We just choose random index from all available actions. Results of this strategy are the worst.


def epsilon_greedy(mab: MAB, epsilon):
    if rand.random() < epsilon:
        return rand.randrange(0, mab.no_actions)
    else:
        return mab.bandit_q_values.argmax()
    # Now that you have implemented the strategy, don't forget to comment in the strategy below


def decaying_epsilon_greedy(mab: MAB, epsilon_init):
    epsilon = (1 - (mab.step_counter / mab.no_actions)) * epsilon_init
    if rand.random() < epsilon:
        return rand.randrange(0, mab.no_actions)
    else:
        return mab.bandit_q_values.argmax()
    # Now that you have implemented the strategy, don't forget to comment in the strategy below


def ucb(mab: MAB, c):
    ucb_q_values = np.array(
        [mab.bandit_q_values[i] + c * np.sqrt(np.log(mab.step_counter) / (mab.bandit_counters[i]))
         if mab.step_counter and mab.bandit_counters[i] else mab.bandit_q_values[i]
         for i in range(mab.no_actions)])
    return ucb_q_values.argmax()
    # Now that you have implemented the strategy, don't forget to comment in the strategy below


def softmax(mab: MAB, tau):
    z = mab.bandit_q_values / tau
    reduction_axes = tuple(range(1, len(z.shape)))
    e_z = np.exp(z - np.max(z))
    softmaxed_z = e_z / e_z.sum(axis=reduction_axes, keepdims=True)
    cumsum_softmaxed_z = np.cumsum(softmaxed_z)
    random_number = rand.random()
    if random_number <= cumsum_softmaxed_z[0]:
        return 0
    for i in range(1, np.size(cumsum_softmaxed_z)):
        if cumsum_softmaxed_z[i - 1] < random_number <= cumsum_softmaxed_z[i]:
            return i

    # Now that you have implemented the strategy, don't forget to comment in the strategy below


if __name__ == '__main__':
    no_rounds = 1000000
    epsilon = 0.5
    epsilon_init = 0.6
    tau = 0.01
    c = 1.0
    num_actions = 10
    biases = [1.0 / k for k in range(5, 5 + num_actions)]
    best_action_value = 0.7

    strategies = {}
    strategies[random] = {}
    strategies[epsilon_greedy] = {'epsilon': epsilon}
    strategies[decaying_epsilon_greedy] = {'epsilon_init': epsilon_init}
    strategies[ucb] = {'c': c}
    strategies[softmax] = {'tau': tau}

    average_total_returns = {}
    total_regrets = {}

    for strategy, parameters in strategies.items():
        print(strategy.__name__)
        bandits = [Bandit(bias, 1 - bias) for bias in biases]
        mab = MAB(best_action_value, *bandits)
        total_regret, average_total_return = mab.run(strategy, **parameters)
        print("\n")
        average_total_returns[strategy.__name__] = average_total_return
        total_regrets[strategy.__name__] = total_regret
    plot(total_regrets)
