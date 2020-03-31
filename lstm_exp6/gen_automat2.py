import networkx as nx
import matplotlib.pyplot as plt
from utils import draw_automat
from utils import get_active_states
from utils import automat_gen_string
import tensorflow as tf
import numpy as np
from utils import automat_gen_string_bfs
from scipy.special import softmax

class State(object):
	def __init__(self, name='X', prefix=set(), suffix=set(),transition = {},pre_states = {}):
		self.name = name
		self.prefix = prefix
		self.suffix = suffix
		self.transition = transition
		self.pre_states = pre_states

	def add_prefix(self, pre):
		self.prefix.add(pre)

	def add_suffix(self,suff):
		self.suffix.add(suff)

	def add_transition(self,char,state):
		self.transition[char] = state


def merge_state(state1,state2,automata):
	# state2.name += "-" + state1.name
	state2.prefix = state2.prefix.union(state1.prefix)
	state2.suffix = state2.suffix.union(state1.suffix)
	if state2.name == 'Start':
		state2.prefix = set()

	for t in state1.transition:
		state2.transition[t] = state1.transition[t]
	for p in state1.pre_states:
		state2.pre_states[p] = state1.pre_states[p]
		automata.states[p].transition[state1.pre_states[p]] = state2
	print('done merging')
	return state2

def merge_state_prefix(state1,state2,automata):
	# state2.name += "-" + state1.name
	state2.prefix = state2.prefix.union(state1.prefix)
	state2.suffix = state2.suffix.union(state1.suffix)
	if state2.name == 'Start':
		state2.prefix = set()

	# for t in state1.transition:
	# 	state2.transition[t] = state1.transition[t]
	for p in state1.pre_states:
		state2.pre_states[p] = state1.pre_states[p]
		automata.states[p].transition[state1.pre_states[p]] = state2
	print('done merging')
	return state2

class Automata(object):
	# Initializer / Instance Attributes
	def __init__(self, start_state=None, accepted_states=[]):
		self.start_state = start_state
		self.states = {self.start_state.name:self.start_state}
		self.accepted_states = accepted_states

	def add_states(self,state):
		self.states[state.name] = state

	def set_start_state(self,start_state_name):
		self.start_state = start_state_name

	def set_accepted_states(self,accepted_states_name):
		self.accepted_states = accepted_states_name

	def add_transition(self,from_state,char,to_state):
		if from_state.name not in self.states:
			self.states[from_state.name] = {}
		if to_state.name not in self.states:
			self.states[to_state.name] = {}
		self.states[from_state.name][char] = to_state


	def proceed(self,w):
		current_state = self.start_state
		for c in w:
			next_state = self.states[current_state][c]
			current_state = next_state
		return current_state

	def is_accepted(self,state_name):
		return state_name in self.accepted_states

	def draw(self):
		pass

def pick_min_state(stateList):
	minState = stateList[0]
	for state in stateList:
		if len(state.prefix) + len(state.suffix) < len(minState.prefix) + len(minState.suffix):
			minState = state
	return minState

def pick_min_state1(state1,state2):
	# len1 = len(state1.prefix) + len(state1.suffix)
	# len2 = len(state2.prefix) + len(state2.suffix)
	minpref1, minsuff1, minpref2, minsuff2 = 0,0,0,0
	if len(state1.prefix) >0 :
		minpref1 = min([len(pref) for pref in state1.prefix])
	if len(state1.suffix) >0 :
		minsuff1 = min([len(suff) for suff in state1.suffix])
	if len(state2.prefix) >0 :
		minpref2 = min([len(pref) for pref in state2.prefix])
	if len(state2.suffix) >0 :
		minsuff2 = min([len(suff) for suff in state2.suffix])

	if minpref1 +  minsuff1 < minpref2 + minsuff2:
		return state2, state1
	else:
		return state1,state2


if __name__ == "__main__":

	print('----building the model----')

	state_size = 2
	num_classes = 2
	batch_size = 1
	alphabets = [1, 0]
	num_layers = 1

	print('-----')

	batchX_placeholder = tf.compat.v1.placeholder(tf.float32, [batch_size, None, len(alphabets)])
	# batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
	y_lbl_placeholder = tf.compat.v1.placeholder(tf.int64, [None, num_classes])

	W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
	b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

	# Unpack columns
	# inputs_series = tf.split(batchX_placeholder, tf.shape(batchX_placeholder)[1], axis=1)
	# labels_series = tf.unstack(batchY_placeholder, axis=1)

	# Forward passes
	lstm = tf.contrib.rnn.BasicLSTMCell(state_size)
	cell = tf.contrib.rnn.MultiRNNCell([lstm for _ in range(num_layers)])
	init_state = cell.zero_state(batch_size, tf.float32)

	states_series, current_state = tf.nn.dynamic_rnn(cell, batchX_placeholder, initial_state=init_state)

	logits_series = tf.matmul(tf.squeeze(states_series, axis=0), W2)
	prediction_series = tf.nn.softmax(logits_series, axis=1)

	# logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
	# predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

	output_logits = logits_series[-1]
	y_pred = prediction_series[-1]

	# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
	# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
	#             for logits, labels in zip(logits_series,labels_series)]

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_lbl_placeholder, logits=output_logits),
						  name='loss')

	# total_loss = tf.reduce_mean(losses)

	train_step = tf.compat.v1.train.AdagradOptimizer(0.3).minimize(loss)

	print('-----')


	sess = tf.compat.v1.Session()
	sess.run(tf.compat.v1.initialize_all_variables())
	saver = tf.compat.v1.train.Saver()
	saver.restore(sess, './my_test_model')


	def verify_automata(sess,startState):
		stringList = automat_gen_string_bfs(startState)
		for s in stringList:
			seq = s.split("-")[1:-1]
			_loss, _states_series, _current_state, _y_pred, _logits_series = sess.run(
				[loss, states_series, current_state, y_pred, logits_series],
				feed_dict={
					batchX_placeholder: [seq],
					y_lbl_placeholder: [1],
					# init_state: _current_state
					# cell_state: _current_cell_state,
					# hidden_state: _current_hidden_state

				})
			logit_series_sm = softmax(_logits_series, axis=1)
			for l in logit_series_sm:
				if l < 0.5:
					return False
		return True




	print('-----')



	inpf = open('path_file_sorted.txt', 'r')
	prefixDict = {}
	suffDict = {}
	suffDictSet = {}
	count = 1
	print('---start state---')
	startState = State(name='Start',prefix=set(), suffix=set(),transition = {},pre_states = {})

	myAutomat = Automata(start_state=startState)
	newLine = True
	for line in inpf:
		print('line = ',line)
		newLine = False
		# line1 = line.split(".")
		stateTrans = line[2:].strip().split("-")

		# pre = stateTrans[0]
		# suff = "".join(stateTrans[1:])
		startState.add_suffix("-".join(stateTrans[:]))
		suffDictSet = {k: v for k, v in suffDictSet.items() if v != startState.name}
		suffDictSet[frozenset(startState.suffix)] = startState.name
		# prevState = State(name=str(count), prefix=set(pre), suffix=set(suff),pre_states={startState.name: [pre]})
		# myAutomat.add_transition(startState,pre,prevState)
		myAutomat.add_states(startState)
		prevState = startState

		# draw_automat(myAutomat)

		def shortest_pre(state):
			min_pre = 99999
			res = ""
			for pre in state.prefix:
				if len(pre) < min_pre:
					min_pre = len(pre)
					res = pre
			return res
		for i in range(1,len(stateTrans)):
			# pre = "-".join(stateTrans[:i])
			if prevState.name == 'Start':
				pre = stateTrans[i - 1]
			else:
				pre = shortest_pre(prevState) + "-" + stateTrans[i - 1]
			suff = "-".join(stateTrans[i:])

			stateName = str(count)
			if i == len(stateTrans) - 1:
				stateName = 'P'
			currState = State(name=stateName,prefix=set(), suffix=set(),transition = {},pre_states = {})

			currState.add_prefix(pre)
			currState.add_suffix(suff)
			prevState.add_transition(stateTrans[i - 1],currState)
			currState.pre_states = {prevState.name:stateTrans[i - 1]}
			# if suff == 'P':
			# 	currState.add_transition(stateTrans[i],currState)
			myAutomat.add_states(currState)


			mergeSuffix = False
			mergedState = currState

			if pre in prefixDict:
				# prefixDict[pre].append(count)
				#---merge---
				# mergeToState = pick_min_state(currState,pre)
				# prefixDict[pre] = str(mergeToState.name)
				state1,state2 = pick_min_state1(currState,myAutomat.states[prefixDict[pre]])
				mergedState = merge_state_prefix(state1,state2,myAutomat)
				currState = mergedState
				suffDictSet = {k: v for k, v in suffDictSet.items() if v != state1.name}
				prefixDict = {k: v for k, v in prefixDict.items() if v != state1.name}
	

			if frozenset(currState.suffix) in suffDictSet:
				# mergeToState = pick_min_state([currState] + suffDict[suff])
				# suffDict[suff] = str(mergeToState.name)
				state1,state2 = pick_min_state1(currState,myAutomat.states[suffDictSet[frozenset(currState.suffix)]])
				if state1.name == state2.name:
					prevState = state1
					count += 1
					continue
				mergeSuffix = True
				mergedState = merge_state(state1,state2, myAutomat)

				active_states = get_active_states(myAutomat)
				currState = mergedState
				suffDictSet = {k: v for k, v in suffDictSet.items() if v in active_states}
				prefixDict = {k: v for k, v in prefixDict.items() if v in active_states}

			# prefixDict[pre] = mergedState.name
			for pre in mergedState.prefix:
				prefixDict[pre] = mergedState.name
			suffDict[suff] = mergedState.name
			suffDictSet = {k:v for k, v in suffDictSet.items() if v != mergedState.name}
			suffDictSet[frozenset(currState.suffix)] = mergedState.name
			prevState = mergedState
			count+=1
			# draw_automat(myAutomat)

			if mergeSuffix:
				break
		# if newLine:
		# 	break

	draw_automat(myAutomat)
	print('------testing-----')


	s0 = State(name='Start',prefix=set(), suffix=set(),transition = {},pre_states = {})
	s1 = State(name='1',prefix=set(), suffix=set(),transition = {},pre_states = {})
	s2 = State(name='2',prefix=set(), suffix=set(),transition = {},pre_states = {})
	sP = State(name='P',prefix=set(), suffix=set(),transition = {},pre_states = {})
	s3 = State(name='3',prefix=set(), suffix=set(),transition = {},pre_states = {})

	s0.transition['1'] = s1
	s1.transition['1'] = s2
	# s1.transition['0'] = s1
	s2.transition['1'] = sP
	s0.transition['0'] = s3
	s3.transition['0'] = s1
	s3.transition['1'] = sP
	new_automat = Automata(start_state = s0)
	# for s in [s0, s1, s2, s3, sP]:
	# 	new_automat.states[s.name] = s

	print(automat_gen_string(new_automat))
	print('Done')