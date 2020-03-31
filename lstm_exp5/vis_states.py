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

state_list_sm = softmax(states_lg, axis=1)
fig, ax = plt.subplots()
ax.scatter(states_df[:,0], states_df[:,1], c=state_list_sm[:, 1],s=10 )
# plt.show()



def drawArrow(A, B, label, color='red'):
	# print('label = ',label)
	ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
			 head_width=0.02, length_includes_head=True, label=label, color=color)
	ax.annotate(label, ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2))

df = pd.read_csv('short_states.csv')
print('df = ', df)
short_states = [[0,0]]
short_states.extend(utils.process_df(df['state']))
seq = df['digit'].values
print('short states = ',short_states)

for i in range(len(short_states) - 1):
	drawArrow(short_states[i], short_states[i + 1], label=seq[i],color='blue')

plt.show()
print('-----generate dictionaries----')
# d1 = {}
# for i,state in enumerate(states_df):
# 	d1[i] = tuple(state)



print('Done')