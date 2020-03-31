from __future__ import print_function, division
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sys
from sklearn.utils import shuffle
import utils
from scipy.spatial import distance
from automata import Automata
from scipy.special import softmax
import pandas as pd


# tf.compat.v1.disable_eager_execution()

df = pd.read_csv('long_states_df.csv',nrows=1000)

states_df = utils.process_df1(df['states'])
states_lg = utils.process_df1(df['logit_seq'])
digit_seq = df['words'].values
ind_seq = df['ind_seq'].values
reset_seq = df['reset_bool'].values

print("states_df = ",states_df.shape)

# np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
state_list_sm = softmax(states_lg, axis=1)
fig, ax = plt.subplots()
ax.scatter(states_df[:,0], states_df[:,1], c=state_list_sm[:, 1],s=10 )
# plt.show()
# print('df = ',df)
df['state_list_sm'] = list(state_list_sm)
# print('df = ',df)
df.to_csv('long_states_df_sm.csv', index=False, header=True)



def drawArrow(ax, A, B, label, color='red'):
	# print('label = ',label)
	ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
			 head_width=0.02, length_includes_head=True, label=label, color=color)
	ax.annotate(label, ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2))

df_short = pd.read_csv('short_states.csv')
print('df_short = ', df_short)
short_states = utils.process_df(df_short['state'])
seq = df_short['digit'].values
print('short states = ',short_states)

for i in range(len(short_states) - 1):
	drawArrow(ax,short_states[i], short_states[i + 1], label=seq[i],color='blue')

plt.show()
print('-----generate dictionaries----')
id_to_p = {}
p_to_id = {}
for i,state in enumerate(states_df):
	id_to_p[i] = (tuple(state),state_list_sm[i],reset_seq[i],ind_seq[i])
	p_to_id[tuple(state)] = i

state_points = {}

print(id_to_p)
print(p_to_id)


print('----generate series----')

series_dict = {}

def gen_series(pid):
	if id_to_p[pid][3] == 1:
		if state_list_sm[pid][0] > 0.5:
			f_cluster = "P"
		else:
			f_cluster = "N"

		return f_cluster


pid = 37
print('[db] res for pid = ',gen_series(pid))

seq_dict = {i:gen_series(i) for i in range(len(states_df) - 2)}


print('---visualizing P and N points----')

P_vis_points = []
N_vis_points = []
P_point_ids = []
N_point_ids = []
for k,v in seq_dict.items():
	if v == 'P':
		P_vis_points.append(id_to_p[k][0])
		P_point_ids.append(k)
	if v == 'N':
		N_vis_points.append(id_to_p[k][0])
		N_point_ids.append(k)

P_points = np.array(P_vis_points)
N_points = np.array(N_vis_points)

fig, ax = plt.subplots()
		# ax.scatter(states_df[:, 0], states_df[:, 1], c=state_list_sm[:, 1], s=10)
ax.scatter(P_points[:, 0], P_points[:, 1], c='blue', s=10, cmap='viridis')
ax.scatter(N_points[:, 0], N_points[:, 1], c='red', s=10, cmap='viridis')
plt.show()
print('Done')

print('----build automata----')

automat = {}
digits = ['0','1']
def gen_automat(prefix, pids):
	if len(pids) == 0:
		return
	automat[prefix] = pids
	prev_pids = {}
	for d in digits:
		prev_pids[d] = []
	for pid in pids:
		if reset_seq[pid] == 1:
			continue
		prev_pid = pid - 1
		prev_pids[str(digit_seq[pid])].append(prev_pid)

	for d in prev_pids:
		gen_automat(prefix + "." + d,prev_pids[d])

gen_automat('P',P_point_ids)
gen_automat('N',N_point_ids)

print('automat = ',automat)

for k in automat:
	print(str(k) + ":" + str(automat[k]))

print('Done')

