import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import random
import check_test
from plot_utils import plot_values
env = gym.make('CliffWalking-v0')
def epsilon_greedy(Q, state, nA, eps):

    if random.random() > eps:  # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:  # otherwise, select an action randomly
        return random.choice(np.arange(env.action_space.n))

def q_learning(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        epsilon = 1.0 / i_episode
        state = env.reset()
        action = epsilon_greedy(Q, state, env.nA, epsilon)

        while True:
            next_state, reward, done, info = env.step(action)
            if (not done):
                # next_action = epsilon_greedy(Q, next_state, env.nA, epsilon)
                Q[state][action] = Q[state][action] + alpha * (
                            reward + (gamma * np.max(Q[next_state])) - Q[state][action])
                state = next_state
                action = epsilon_greedy(Q, next_state, env.nA, epsilon)
            if (done):
                Q[state][action] = Q[state][action] + alpha * (reward + (0 - Q[state][action]))
                break

    return Q

Q_sarsamax = q_learning(env, 5000, .01)

# print the estimated optimal policy
policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
check_test.run_check('td_control_check', policy_sarsamax)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsamax)

# plot the estimated optimal state-value function
plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])