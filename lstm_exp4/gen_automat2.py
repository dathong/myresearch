import networkx as nx
import matplotlib.pyplot as plt
from utils import draw_automat
from utils import get_active_states
from utils import automat_gen_string

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