import networkx as nx
import matplotlib.pyplot as plt
from utils import draw_automat
from utils import get_active_states
from utils import automat_gen_string_dfs


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

class Automata:
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

	print('------testing-----')


	s0 = State(name='Start',prefix=set(), suffix=set(),transition = {},pre_states = {})
	s1 = State(name='1',prefix=set(), suffix=set(),transition = {},pre_states = {})
	s2 = State(name='2',prefix=set(), suffix=set(),transition = {},pre_states = {})
	sP = State(name='P',prefix=set(), suffix=set(),transition = {},pre_states = {})
	s3 = State(name='3',prefix=set(), suffix=set(),transition = {},pre_states = {})

	s0.transition['1'] = s1
	s1.transition['1'] = s2
	s2.transition['1'] = sP
	s1.transition['0'] = s3
	# s2.transition['1'] = sP
	s0.transition['0'] = s0
	s3.transition['0'] = s0
	# s3.transition['1'] = sP
	new_automat = Automata(start_state = s0)
	# for s in [s0, s1, s2, s3, sP]:
	# 	new_automat.states[s.name] = s

	print(automat_gen_string_bfs(s0))
	print('Done')