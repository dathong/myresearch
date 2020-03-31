import networkx as nx
import matplotlib.pyplot as plt

class Automata:
	# Initializer / Instance Attributes
	def __init__(self, start_state=None, accepted_states=[]):
		self.start_state = start_state
		self.states = {self.start_state:{}}
		self.accepted_states = accepted_states


	def set_start_state(self,start_state):
		self.start_state = start_state

	def set_accepted_states(self,accepted_states):
		self.accepted_states = accepted_states

	def add_state(self,state,accepted=False):
		if state not in self.states:
			self.states[state] = {}
		if accepted:
			self.accepted_states.add(state)

	def add_transition(self,from_state,char,to_state):
		if from_state not in self.states:
			self.states[from_state] = {}
		if to_state not in self.states:
			self.states[to_state] = {}
		self.states[from_state][char] = to_state


	def proceed(self,w):
		current_state = self.start_state
		for c in w:
			next_state = self.states[current_state][c]
			current_state = next_state
		return current_state

	def is_accepted(self,state):
		return state in self.accepted_states

	def draw(self):
		pass


if __name__ == "__main__":
	automat = Automata('100','100')


	automat.add_transition('A','1','B')
	automat.add_transition('B','1','C')
	automat.add_transition('C', '0', 'D')
	automat.add_transition('D', '1', 'B')
	# automat.add_transition('010','b','000')
	# automat.add_transition('000','b','010')
	# automat.add_transition('001','b','100')
	# automat.add_transition('100','b','001')
	# automat.add_transition('001','a','000')
	# automat.add_transition('000','a','001')

	# final_state = automat.proceed('ababb')
	# print(automat.proceed('ababb'))
	automat.draw()