from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sys
from sklearn.utils import shuffle
import utils
from scipy.spatial import distance
from automata import Automata
from scipy.special import softmax
import pandas as pd


tf.compat.v1.disable_eager_execution()

df = pd.read_csv('long_states_df.csv',nrows=2000)

states_df = utils.process_df1(df['states'])
states_lg = utils.process_df1(df['logit_seq'])

print("states_df = ",states_df)

# np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
state_list_sm = softmax(states_lg, axis=1)
fig, ax = plt.subplots()
ax.scatter(states_df[:,0], states_df[:,1], c=state_list_sm[:, 1],s=10 )
# plt.show()
print('df = ',df)
df['state_list_sm'] = list(state_list_sm)
print('df = ',df)
df.to_csv('long_states_df_sm.csv', index=False, header=True)



def drawArrow(A, B, label, color='red'):
	# print('label = ',label)
	ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
			 head_width=0.02, length_includes_head=True, label=label, color=color)
	ax.annotate(label, ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2))

df = pd.read_csv('short_states.csv')
print('df = ', df)
short_states = utils.process_df(df['state'])
seq = df['digit'].values
print('short states = ',short_states)

# for i in range(len(short_states) - 1):
# 	drawArrow(short_states[i], short_states[i + 1], label=seq[i],color='blue')

plt.show()
print('-----generate dictionaries----')
id_to_p = {}
p_to_id = {}
for i,state in enumerate(states_df):
	id_to_p[i] = (tuple(state),state_list_sm[i])
	p_to_id[tuple(state)] = i

state_points = {}

print(id_to_p)
print(p_to_id)

print('----clustering----')

from sklearn.cluster import KMeans

no_of_clusters = 5
kmeans = KMeans(n_clusters=no_of_clusters)
kmeans.fit(states_df)
y_kmeans = kmeans.predict(states_df)
print('y_kmeans = ',y_kmeans)
plt.scatter(states_df[:, 0], states_df[:, 1], c=y_kmeans , s=10, cmap='viridis')
centers_pos = kmeans.cluster_centers_
ax.plot(centers_pos[:, 0], centers_pos[:, 1], 'kx', markersize=15, color='blue')

plt.show()

print('----generate series----')

series_dict = {}

def gen_series(pid):
	max_id = pid
	points = []
	digits = []
	res = ""
	while True:
		max_id+=1
		p_next= id_to_p[max_id][0]
		points.append(p_next)
		digits.append(seq[max_id])
		if state_list_sm[max_id][1] >= 0.9:
			break
	clusters = kmeans.predict(points)
	for i,c in enumerate(clusters):
		res += str(digits[i]) + "-" + str(c)
	return res

pid = 1
print('[db] res for pid = ',gen_series(pid))
print('Done')