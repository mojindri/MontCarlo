import sys
import gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values,plot_policy
env = gym.make('Blackjack-v0')
#test
'''
for i_episode in range(3):
    state = env.reset()
    while True:
        print (state)
        action = env.action_space.sample()
        state,reward, done,info = env.step(action)
        if done:
            print ("Game ENd! Reward: " , reward)
            print("You Won \n")  if reward > 0 else print ("You Lost \n")
            break
'''
def generate_episode_from_Q(env,Q,epsilon,nA):
    episode=[]
    state = env.reset()
    while True:
        action = np.random.choice(np.arange(nA), p = gets_prob(Q[state],epsilon,nA)) if state in Q else env.action_space.sample()
        next_state,reward, done, info = env.step(action)
        episode.append((state,action,reward))
        state = next_state
        if done:
            break
    return episode

def gets_prob(Q_S, epsilon,nA):
    policy_s =np.ones(nA)  * epsilon/nA
    best_a  = np.argmax(Q_S)
    policy_s[best_a]  = 1- epsilon + (epsilon/nA)
    return policy_s

def update_Q(env,episode, Q,alpha, gamma):
    states, actions, rewards = zip (*episode)
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha * (sum(rewards[i:] * discounts[:-(1+i)]) - old_Q)
    return  Q

def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # set the value of epsilon
        epsilon = max(epsilon*eps_decay, eps_min)
        # generate an episode by following epsilon-greedy policy
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        # update the action-value function estimate using the episode
        Q = update_Q(env, episode, Q, alpha, gamma)
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q

policy, Q = mc_control(env, 500000, 0.2)