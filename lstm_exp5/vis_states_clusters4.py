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
import pickle


# tf.compat.v1.disable_eager_execution()

df = pd.read_csv('long_states_df.csv',nrows=5000)

states_df = utils.process_df1(df['states'])
states_lg = utils.process_df1(df['logit_seq'])
digit_seq = df['words'].values
ind_seq = df['ind_seq'].values
reset_seq = df['reset_bool'].values

print("states_df = ",states_df.shape)

np.set_printoptions(precision=4)
state_list_sm = softmax(states_lg, axis=1)
print('state_list_sm = ',np.array(state_list_sm))
# print('df = ',df)

print('-----generate dictionaries----')
id_to_p = {}
p_to_id = {}
for i,state in enumerate(states_df):
	id_to_p[i] = (tuple(state),state_list_sm[i],reset_seq[i],ind_seq[i])
	p_to_id[tuple(state)] = i

state_points = {}

print(id_to_p)
print(p_to_id)

print('----get the clustering dict----')

with open ('./y_kmeans', 'rb') as fp:
	y_means = pickle.load(fp)

cluster_dicts = {}
print('y means = ',y_means)
for i,m in enumerate(y_means):
	if m not in cluster_dicts:
		cluster_dicts[m] = [i]
	else:
		cluster_dicts[m].append(i)

print('cluster_dicts = ',cluster_dicts)

print('----generate series----')

series_dict = {}

def gen_series(pid,y_means,vis=False):
	# if id_to_p[pid][3] == 1:
	if state_list_sm[pid][0] > 0.9:
		f_cluster = "P."
		return [f_cluster, f_cluster]
	if id_to_p[pid][3] == 1:
		f_cluster = "N."
		return [f_cluster, f_cluster]
	if id_to_p[pid][2] == 1:
		f_cluster = "S."
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
		p_next= max_id
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
	for i,c in enumerate(points):
		res += str(digits[i]) + "-" + str(y_means[c]) + "."
		resX += "X-" + str(c) + "."
	res = res[:-2] + f_cluster + "."
	resX = resX[:-2] + f_cluster + "."
	# print('res = ',res)
	# print('resX = ',resX)
	if vis:
		pass
	return [res, resX]

pid = 37
print('[db] res for pid = ',gen_series(pid,y_means,vis=False))

seq_list = []
# for i in range(len(states_df)):
seq_dict = {i:gen_series(i,y_means,vis=False)[0] for i in range(len(states_df) - 2)}
seq_dictX = {i:gen_series(i,y_means,vis=False)[1] for i in range(len(states_df) - 2)}

import operator
sorted_seq = sorted(seq_dict.items(), key=operator.itemgetter(1))
sorted_seqX = sorted(seq_dictX.items(), key=operator.itemgetter(1))
print('----ppp----')
for s in sorted_seq:
	# print(s[0])
	if 'P' in s[1]:
		print(id_to_p[s[0]][0],s)

print('Done')

print('---visualizing P points----')

P_vis_points = []
N_vis_points = []
S_vis_points = []
for k,v in seq_dict.items():
	if v == 'P.':
		P_vis_points.append(id_to_p[k][0])
	if v == 'N.':
		N_vis_points.append(id_to_p[k][0])
	zero_p = np.array([0, 0])
	eps = 0.01
	if id_to_p[k][2] == 1:
		S_vis_points.append(id_to_p[k][0])



P_vis_points = np.array(list(set(P_vis_points)))
N_vis_points = np.array(list(set(N_vis_points)))
S_vis_points = np.array(list(set(S_vis_points)))

fig, ax = plt.subplots()
ax.scatter(states_df[:, 0], states_df[:, 1], c=state_list_sm[:, 1], s=10)
# ax.scatter(P_vis_points[:, 0], P_vis_points[:, 1], c='blue', s=10, cmap='viridis')
# ax.scatter(N_vis_points[:, 0], N_vis_points[:, 1], c='red', s=10, cmap='viridis')
plt.show()
print('P_vis_points ',P_vis_points)
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

print('----implementing----')

print('sorted_seq = ',sorted_seq)
print('seq_dict = ',seq_dict)

def vis_cluster(c):
	fig, ax = plt.subplots()
	points = np.array([np.array(id_to_p[id][0]) for id in cluster_dicts[c]])
	if len(points) < 1:
		return
	ax.scatter(points[:, 0], points[:, 1], c='green', s=10)
	ax.scatter(P_vis_points[:, 0], P_vis_points[:, 1], c='blue', s=10, cmap='viridis')
	ax.scatter(N_vis_points[:, 0], N_vis_points[:, 1], c='red', s=10, cmap='viridis')
	plt.show()

def get_prev_states(state):
	res = set()
	for k,v in lbl_to_p_dict.items():
		if k.endswith(state):
			# print('k = ',k)
			tmp = k.split(state)[0]
			# print('tmp1 = ', tmp)
			if len(tmp) < 1:
				continue
			tmp = tmp.split(".")
			# print('tmp2 = ',tmp)
			if len(tmp) < 2:
				continue
			prev_state = tmp[-2].split("-")[-1]
			# print('prev_state = ', prev_state)
			res.add(prev_state)
	return list(res)
#
# for k,v in seq_dict.items():
# 	if v == 'P.':
# 		P_vis_points.append(id_to_p[k][0])

P_points = [k for k,v in seq_dict.items() if v == "P."]
N_points = [k for k,v in seq_dict.items() if v == "N."]
S_points = [k for k,v in seq_dict.items() if v == "S."]


print('----fixing cluster dicts---')
for k,v in cluster_dicts.items():
	for pid in P_points + N_points + S_points:
		if pid in cluster_dicts[k]:
			cluster_dicts[k].remove(pid)

for k in list(cluster_dicts):
	if len(cluster_dicts[k]) < 1:
		del cluster_dicts[k]

y_kmeans1 = list(y_means)
for i,k in enumerate(y_means):
	if state_list_sm[i][0] > 0.9:
		y_kmeans1[i] = 'P'
	if id_to_p[i][3] == 1:
		y_kmeans1[i] = 'N'
	if id_to_p[i][2] == 1:
		y_kmeans1[i] = 'S'

print('y_kmeans1 = ',y_kmeans1)
cluster_dicts['P'] = P_points
cluster_dicts['N'] = N_points
cluster_dicts['S'] = S_points

state_dicts = cluster_dicts.copy()

print('P_points = ',P_points)
print('N_points = ',N_points)

# for c in state_dicts:
# 	print('cluster_dicts',c,cluster_dicts[c])
# 	vis_cluster(c)

#ewp[pwep[]\
digit_lists = ['0','1']

def prev_points(point_ids):
	d = {str(y_kmeans1[pid - 1]) + "." + str(digit_seq[pid]):[]
		 for pid in point_ids if pid - 1 not in P_points}
	for pid in point_ids:
		if pid - 1 in P_points:
			continue
		d[str(y_kmeans1[pid - 1]) + "." + str(digit_seq[pid])].append(pid - 1)

	return d


automat_path = []
def process(path,lbl,point_set):
	if 'S' in lbl:
		s_name = 'O.' + str(digit_seq[point_set[0]])
		print('path = ',s_name + "-" + path)
		automat_path.append(s_name + "-" + path)
		return
		# return path
	d = prev_points(point_set)

	res = []
	for k in d:
		res.append(len(d[k]))
	if len(res) < 1:
		s_name = str(digit_seq[point_set[0]])
		print('path = ', s_name + "." + path)
		automat_path.append(s_name + "-" + path)
		return
	res = sorted(res,reverse=True)
	for state in d:
		if len(d[state]) not in res[:3]:
			continue
		process(state + "-" + path, state, d[state])

def process1(path,lbl,point_set,opf_name):

	if 'S' in lbl:
		s_name = 'O.' + str(digit_seq[point_set[0]])
		print('path = ',s_name + "-" + path)
		opf = open(opf_name, 'a')
		opf.write(s_name + "." + path + '\n')
		automat_path.append(s_name + "-" + path)
		return
		# return path
	d = prev_points(point_set)

	res = []

	for k in d:
		res.append(len(d[k]))
	if len(res) < 1:
		s_name = str(digit_seq[point_set[0]])
		print('path = ', s_name + "." + path)
		opf = open(opf_name, 'a')
		opf.write(s_name + "." + path + '\n')
		automat_path.append(s_name + "-" + path)
		return
	res = sorted(res,reverse=True)
	for state in d:
		if len(d[state]) not in res[:3]:
			continue
		process1(state.split(".")[1] + "-" + path, state, d[state],opf_name)

process('P.','P',P_points)
opf_name = 'path_file.txt'
opf = open(opf_name, 'w')
process1('P.','P',P_points,opf_name)
opf.close()
# print('automat_path = ',automat_path)

res = get_prev_states('P.')
# print('res = ',res)
vis_cluster('P')

#---generate automat----
# from automata import Automata
# automat = Automata('O','P')
# pre_dict = {}
# suf_dict = {}
# for path in automat_path:
# 	for i in range(len(path[:-2]),0,-1):
# 		# print('i = ',i)
# 		if path[i] == "-":
# 			pre, suf = path[:i], path[i+1:]
# 			pre_dict[pre] = suf
# 			suf_dict[suf] = pre
# 			digit = suf.split("-")[0].split(".")[1]
# 			# print('pre, suf, digit = ', pre, '  ', suf, '  ', digit)
# 			next_state = ""
# 			if "-" in suf:
# 				next_state = suf.split("-")[1]
# 			# print('next_state = ',next_state)
# 			automat.add_transition(suf,digit,next_state)

print('Done')
