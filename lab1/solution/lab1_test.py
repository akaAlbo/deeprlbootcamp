from __future__ import print_function

import numpy as np


A_lin = np.zeros(16)
print(A_lin)

A = np.arange(16).reshape(4, 4)
prob = [(0.1, np.random.randint(0, 15), float(np.random.randint(2))),
        (0.1, np.random.randint(0, 15), float(np.random.randint(2))),
        (0.1, np.random.randint(0, 15), float(np.random.randint(2)))]

P = {num: [action for action in prob] for num in range(0, 16)}

l1 = [0.1, 11, 1.0]
l2 = [[0.1, 11, 1.0], [0.2, 22, 2.0]]

print(l1)
print(l2)
for one, two, three in l2:
	print(one)
	print(two)
	print(three)
	print('#' * 20)

print('\n' + '#' * 35 + '\n')

print([P[4][0]])
for probability, next_state, reward in [P[0][0]]:
	print(probability)
	print(next_state)
	print(reward)
print('\n' + '#' * 35 + '\n')


print(prob)
print('Probability |', 'State\' |', 'Reward(s\')')
print('\n' + '#' * 35 + '\n')

# Reward
print('Probability:', prob[0][0])
print('State:', prob[0][1])
print('Reward:', prob[0][2])


# P[s][a] = [(prob, nextstate, reward), (prob, nextstate, reward), (prob, nextstate, reward)]

# max (P[s][a][0 = s'][0 = prob] * (P[s][a][0][3 = reward] + GAMMA * Vprev[s']))
# max (P[s][a][1 = s'][0 = prob] * (P[s][a][1][3 = reward] + GAMMA * Vprev[s']))
# max (P[s][a][2 = s'][0 = prob] * (P[s][a][2][3 = reward] + GAMMA * Vprev[s']))
# max (P[s][a][3 = s'][0 = prob] * (P[s][a][3][3 = reward] + GAMMA * Vprev[s']))

# for s in range(0, mdp.nS):
#	for s_next in range(0,3):
#		for a in range(0, mdp.nA):
#    		prob = mdp.P[s][a][s_next][0]
#			reward = mdp.P[s][a][s_next][3]
#			future = GAMMA * Vprev[s_next]
