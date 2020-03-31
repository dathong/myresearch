import networkx as nx
import matplotlib.pyplot as plt

class Automata:
	# Initializer / Instance Attributes
	def __init__(self, start_state=None, accepted_states=[]):
		self.start_state = start_state
		self.states = {self.start_state:{}}
		self.accepted_states = accepted_states

		self.DG = nx.DiGraph()
		# self.DG.add_node(start_state)

	def set_start_state(self,start_state):
		self.start_state = start_state
		self.DG.add_node(start_state)

	def set_accepted_states(self,accepted_states):
		self.accepted_states = accepted_states

	def add_state(self,state,accepted=True):
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
		self.DG.add_node(from_state)
		self.DG.add_node(to_state)
		self.DG.add_edge(from_state,to_state)
		self.DG[from_state][to_state]['label'] = char

	def proceed(self,w):
		current_state = self.start_state
		for c in w:
			next_state = self.states[current_state][c]
			current_state = next_state
		return current_state

	def is_accepted(self,state):
		return state in self.accepted_states

	def draw(self):
		pos = nx.circular_layout(self.DG)
		arc_weight = nx.get_edge_attributes(self.DG, 'label')
		nx.draw(self.DG, pos, with_labels=True, arrowstyle='->',edge_labels=arc_weight)
		# nx.draw_networkx(self.DG,with_labels=True)
		# nx.draw_networkx_edges(self.DG,pos)
		# Draw the edge labels
		nx.draw_networkx_edge_labels(self.DG, pos, edge_labels=arc_weight)
		plt.show()

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