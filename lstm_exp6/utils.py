import numpy as np
from graphviz import Digraph

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

def automat_gen_string_dfs(automat,startName = 'Start', th = 1):
	visited = set()
	paths = []
	def process(path, prev_states,state, visited):

		if state.name == 'P':
			paths.append(path + "-P")
			return
		if state.name not in visited:
			state.cycles = {}
			visited.add(state.name)
		
		if state.name in prev_states:
			cycle_path = prev_states[prev_states.rindex(state.name):]

			if cycle_path in state.cycles:
				state.cycles[cycle_path] += 1
			else:
				state.cycles[cycle_path] = 1

			cycle_counts = {}

			prev_states_str = prev_states
			for cycle_path in state.cycles:
				cycle_counts[cycle_path] = prev_states.count(cycle_path)


			if state.name == 'Start':
				print('-------')
				print('state_cycles = ',state.name,state.cycles)
				print('prev_states: ',prev_states)
				print('cycle_counts = ',cycle_counts)
			if any([True if cycle_counts[x] > th else False for x in cycle_counts]):
				return


		for t in state.transition:
			process(path + '-' + t, prev_states + "-" + state.name, state.transition[t],visited)

	startState = automat.states[startName]
	process("S","",startState,visited)
	print('path len = ',len(paths))
	return paths


def automat_gen_string_bfs(startState, th = 10):

	def process(state, th):
		res_path = []
		queue = [state]
		paths = [state.name]
		prev_states = [state.name]
		# curr_prevs = [startName]
		while len(queue) > 0:
			pop_state = queue.pop(0)
			curr_path = paths.pop(0)
			curr_prevs = prev_states.pop(0)
			if pop_state.name == 'P':
				res_path.append(curr_path + '-P')
			if len(curr_path.split("-")) > th:
				break
			for t in pop_state.transition:
				queue.append(pop_state.transition[t])
				paths.append(curr_path + "-" + str(t))
				prev_states.append(curr_prevs + "-" + pop_state.transition[t].name)
		return res_path

	# startState = automat.states[startName]
	paths = process(startState,10)
	print('path len = ',len(paths))
	return paths

def convert_data_x(x,alphabets):
    d = {a:i for i,a in enumerate(alphabets)}
    res = []
    for ip in x:
        res1 = []
        for e in ip:
            v = [0] * len(d)
            v[d[e]] = 1
            res1.append(v)
        res.append(res1)
    return res

def convert_data_x_todigit(x,alphabets):
	res = []
	for ip in x:
		res1 = []
		for e in ip:
			for i,d in enumerate(e):
				if str(d) == '1':
					res1.append(alphabets[i])
		res.append(res1)
	return res

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

if __name__ == "__main__":
    a = [[0,1,1,1,0,0,1,0,1,0]]
    print(index_of([1,1,1],a))
    print(index_of([1, 0 ,0,0], a))