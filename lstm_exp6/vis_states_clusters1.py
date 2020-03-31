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

df = pd.read_csv('long_states_df.csv',nrows=5000)

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

# for i in range(len(short_states) - 1):
# 	drawArrow(short_states[i], short_states[i + 1], label=seq[i],color='blue')

# plt.show()
print('-----generate dictionaries----')
id_to_p = {}
p_to_id = {}
for i,state in enumerate(states_df):
	id_to_p[i] = (tuple(state),state_list_sm[i],reset_seq[i],ind_seq[i])
	p_to_id[tuple(state)] = i

state_points = {}

print(id_to_p)
print(p_to_id)

print('----clustering----')

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
#
# distortions = []
# K = range(1,20)
#
# distortions = []
# inertias = []
# mapping1 = {}
# mapping2 = {}
#
#
# for k in K:
# 	kmeans = KMeans(n_clusters=k)
# 	kmeans.fit(states_df)
# 	distortions.append(sum(np.min(cdist(states_df, kmeans.cluster_centers_,
# 										'euclidean'), axis=1)) / states_df.shape[0])
# 	inertias.append(kmeans.inertia_)
#
# 	mapping1[k] = sum(np.min(cdist(states_df, kmeans.cluster_centers_,
# 								   'euclidean'), axis=1)) / states_df.shape[0]
# 	mapping2[k] = kmeans.inertia_
#
#
#
# print('distortions = ',distortions)
#
# for key,val in mapping1.items():
# 	print(str(key)+' : '+str(val))
#
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method using Distortion')
# plt.show()


no_of_clusters = 10
kmeans = KMeans(n_clusters=no_of_clusters)
kmeans.fit(states_df)
y_kmeans = kmeans.predict(states_df)
print('y_kmeans = ',y_kmeans)

import pickle

with open('y_kmeans', 'wb') as fp:
    pickle.dump(y_kmeans, fp)

print('----computing mutual distances----')
import scipy.spatial
dist = scipy.spatial.distance.cdist(y_kmeans,y_kmeans)
avgDist = np.sum(dist)/dist.shape[0]
with open('avgClusterDis			t','wb') as fp:
	pickle.dump(avgDist,fp)


plt.scatter(states_df[:, 0], states_df[:, 1], c=y_kmeans , s=10, cmap='viridis')
# for i in range(len(states_df)):
# 	print('y means i = ',y_kmeans[i])
# 	plt.scatter(states_df[i, 0], states_df[i, 1], c=y_kmeans[i], s=10, cmap='viridis')
# zip joins x and y coordinates in pairs
count = 0
# for x,y in zip(states_df[:, 0],states_df[:, 1]):
# 	label = "{:.2f}".format(y)
#
# 	plt.annotate(count, # this is the text
#                  (x,y), # this is the point to label
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,10), # distance from text to points (x,y)
#                  ha='center') # horizontal alignment can be left, right or center
# 	count+=1
centers_pos = kmeans.cluster_centers_
ax.plot(centers_pos[:, 0], centers_pos[:, 1], 'kx', markersize=15, color='blue')
plt.show()

# sys.exit()
print('----generate series----')

series_dict = {}

def gen_series(pid,kmeans,vis=False):
	# if id_to_p[pid][3] == 1:
	if state_list_sm[pid][0] > 0.9:
		f_cluster = "P."
		return [f_cluster, f_cluster]
	if id_to_p[pid][3] == 1:
		f_cluster = "N."
		return [f_cluster, f_cluster]

	# return [f_cluster, f_cluster]
	max_id = pid
	points = []
	digits = []
	res = ""
	resX = ""
	f_cluster = ""
	while pid + 1 in id_to_p:
		max_id+=1
		if max_id not in id_to_p:
			break
		p_next= id_to_p[max_id][0]
		points.append(p_next)
		digits.append(digit_seq[max_id])

		# if id_to_p[max_id][3] == 1:
		if state_list_sm[max_id][0] > 0.9:
			f_cluster = "P"
		# if state_list_sm[max_id][0] < 0.1:
		# 	f_cluster = "N"

			break
		if id_to_p[max_id][3] == 1:
			f_cluster = "N"
			break
	clusters = kmeans.predict(points)
	clusters = list(clusters)
	clusters[-1] = f_cluster
	for i,c in enumerate(clusters):
		res += str(digits[i]) + "-" + str(c) + "."
		resX += "X-" + str(c) + "."
	# print('res = ',res)
	# print('resX = ',resX)
	if vis:
		fig, ax = plt.subplots()
		# ax.scatter(states_df[:, 0], states_df[:, 1], c=state_list_sm[:, 1], s=10)
		ax.scatter(states_df[:, 0], states_df[:, 1], c=y_kmeans, s=10, cmap='viridis')
		ax.plot(centers_pos[:, 0], centers_pos[:, 1], 'kx', markersize=15, color='blue')
		for m in range(pid,max_id):
			drawArrow(ax, id_to_p[m][0], id_to_p[m+1][0], digit_seq[m+1])
		plt.show()
	return [res, resX]

pid = 37
print('[db] res for pid = ',gen_series(pid,kmeans,vis=False))

seq_list = []
# for i in range(len(states_df)):
seq_dict = {i:gen_series(i,kmeans,vis=False)[0] for i in range(len(states_df) - 2)}
seq_dictX = {i:gen_series(i,kmeans,vis=False)[1] for i in range(len(states_df) - 2)}

import operator
sorted_seq = sorted(seq_dict.items(), key=operator.itemgetter(1))
sorted_seqX = sorted(seq_dictX.items(), key=operator.itemgetter(1))
print('----ppp----')
for s in sorted_seq:
	# print(s[0])
	if 'P' in s[1]:
		print(id_to_p[s[0]][0],s)

# for s in sorted_seqX:
	# print(s[0])
	# if 'P' in s[1]:
	# 	print(s[1])
# print('sorted_seq = ',sorted_seq[:50]

print('---visualizing P points----')

P_vis_points = []
N_vis_points = []
for k,v in seq_dict.items():
	if v == 'P.':
		P_vis_points.append(id_to_p[k][0])
	if v == 'N.':
		N_vis_points.append(id_to_p[k][0])

P_vis_points = np.array(P_vis_points)
N_vis_points = np.array(N_vis_points)

fig, ax = plt.subplots()
		# ax.scatter(states_df[:, 0], states_df[:, 1], c=state_list_sm[:, 1], s=10)
ax.scatter(P_vis_points[:, 0], P_vis_points[:, 1], c='blue', s=10, cmap='viridis')
ax.scatter(N_vis_points[:, 0], N_vis_points[:, 1], c='red', s=10, cmap='viridis')
# plt.show()
print('---state points----')

lbl_to_p_dict = {}
for k,v in seq_dict.items():
	if v not in lbl_to_p_dict:
		lbl_to_p_dict[v] = [k]
	else:
		lbl_to_p_dict[v].append(k)
print('lbl_to_p_dict = ',lbl_to_p_dict)

for k,v in lbl_to_p_dict.items():
	print('k,v = ',k,v)

def vis_points_f(lbl):
	vis_points = []
	for k,v in seq_dict.items():
		if v == lbl:
			vis_points.append(list(id_to_p[k][0]))
	return vis_points
vis_points = np.array(vis_points_f('1-P.'))
ax.scatter(vis_points[:, 0], vis_points[:, 1], c='green', s=10, cmap='viridis')
plt.show()

print('------implementing-----')
print('----finding all P points----')

P_vis_points = []
for k,v in seq_dict.items():
	if v[:-2] == 'P.':
		P_vis_points.append(id_to_p[k][0])

print('Done')

