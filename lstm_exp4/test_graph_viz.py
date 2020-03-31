from graphviz import Digraph

dot = Digraph(comment='The Round Table', filename='fsm.gv')
dot.attr(size='8,5')

print('dot = ',dot)

dot.node('1')
dot.node('2')
dot.node('3')
dot.node('4')
dot.node('5')
dot.node('6')

dot.attr('node', shape='circle')
dot.edge('1', '2',label='1')
dot.edge('2', '3',label='2')
dot.edge('3', '4',label='2')
dot.edge('4', '5',label='2')
dot.edge('5', '6',label='2')
dot.edge('1', '6',label='2')

print(dot.source)

dot.render('round-table.gv', view=True)