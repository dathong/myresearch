import numpy as np
from graphviz import Digraph
# from gen_automat2 import Automata
# from gen_automat2 import State

def index_of(sub_arr,arr):
	arr = arr.tolist()
	jump = len(sub_arr)
    # print('sub_arr = ',sub_arr)
    # print('arr = ',arr)
	for i in range(0,len(arr[0])):
		if arr[0][i:i+jump] == sub_arr:
			return i + jump
	return -1

def process_df(field):
	res = []
	for s in field:
		s = s.replace('[','').replace(']','').strip()
		# print('s = ', s)
		first = float(s.split()[0])
		second = float(s.split()[1])
		res.append([first,second])
	return np.array(res)

def process_df1(field):
	res = []
	for s in field:
		s = s.replace('[','').replace(']','').replace(',','').strip()
		# print('s = ', s)
		first = float(s.split()[0])
		second = float(s.split()[1])
		res.append([first,second])
	return np.array(res)

def get_active_states(automat,startName = 'Start'):
	visited = set()
	def active_states(state, visited):
		if state.name == 'P':
			return

		visited.add(state.name)
		for t in state.transition:
			if state.transition[t].name in visited:
				continue
			active_states(state.transition[t],visited)

	startState = automat.states[startName]
	active_states(startState,visited)
	return visited

def automat_gen_string(automat,startName = 'Start', th = 3):
	visited = set()
	paths = []
	def active_states(path, prev_input,state, visited):

		if state.name == 'P':
			paths.append(path)
			return 
		if state.name not in visited:
			state.input_count = {prev_input:1}
			visited.add(state.name)
		else:
			if prev_input in state.input_count:
				if state.input_count[prev_input] > th:
					return
				state.input_count[prev_input] += 1
			else:
				state.input_count[prev_input] = 1

		for t in state.transition:
			active_states(path + '-' + t, t, state.transition[t],visited)

	startState = automat.states[startName]
	active_states("S","",startState,visited)
	return paths

def automat_gen_string_bfs(automat,startName = 'Start', th = 3):
	visited = set()
	paths = []
	def active_states(path, prefix,state, visited):

		if state.name == 'P':
			paths.append(path)
			return
		if state.name not in visited:
			# state.pre_states = prefix + "-" + state.name

			visited.add(state.name)
		else:
			if prev_input in state.input_count:
				if state.input_count[prev_input] > th:
					return
				state.input_count[prev_input] += 1
			else:
				state.input_count[prev_input] = 1

		for t in state.transition:
			active_states(path + '-' + t, t, state.transition[t],visited)

	startState = automat.states[startName]
	active_states("S","S",startState,visited)
	return paths

def draw_automat(automat,startName = 'Start'):
	dot = Digraph(comment='The Round Table', filename='fsm.gv')
	dot.attr(size='8,5')
	dot.attr('node', shape='circle')

	print('dot = ', dot)

	visited = set()
	startState = automat.states[startName]
	dot.node(startName)

	def dfs_drawing(state, visited):
		if state.name == 'P' or state.name in visited:
			return
		dot.node(state.name)
		visited.add(state.name)
		for t in state.transition:
			# if state.transition[t].name in visited:
			# 	continue
			dot.node(state.transition[t].name)
			dot.edge(state.name, state.transition[t].name, label=str(t))
			dfs_drawing(state.transition[t],visited)

	dfs_drawing(startState,visited)
	# dot.node('1')
	# dot.node('2')
	# dot.node('3')
	# dot.node('4')
	# dot.node('5')
	# dot.node('6')
	#
	# dot.attr('node', shape='circle')
	# dot.edge('1', '2', label='1')
	# dot.edge('2', '3', label='2')
	# dot.edge('3', '4', label='2')
	# dot.edge('4', '5', label='2')
	# dot.edge('5', '6', label='2')
	# dot.edge('1', '6', label='2')

	print(dot.source)

	dot.render('round-table.gv', view=True, format='pdf')

# if __name__ == "__main__":
#     a = [[0,1,1,1,0,0,1,0,1,0]]
#     print(index_of([1,1,1],a))
#     print(index_of([1, 0 ,0,0], a))
#     print('----test automat----')
#     automat = Automata()
#     s0 = State(name='Start')
#     s1 = State(name='1')
#     s2 = State(name='2')
#     sP = State(name='P')
#     s3 = State(name='3')

#     s0.transition['1'] = s1
#     s1.transition['1'] = s2
#     s2.transition['1'] = sP
#     s0.transition['0'] = s3
#     s3.transition['0'] = sP

#     for s in [s0,s1,s2,s3,sP]:
#         automat.states[s.name] = s

#     print(automat_gen_string(automat))

