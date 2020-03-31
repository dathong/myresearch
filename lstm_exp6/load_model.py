from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sys
from sklearn.utils import shuffle
from utils import index_of
from utils import convert_data_x
from utils import convert_data_x_todigit

tf.compat.v1.disable_eager_execution()

num_epochs = 1
total_series_no = 10000
truncated_backprop_length = 15
state_size = 2
num_classes = 2
echo_step = 3
batch_size = 1
num_layers = 1
num_batches = total_series_no//batch_size//truncated_backprop_length
alphabets = [1,0]


def sub_list(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return False

def get_batches(A, k):
    # For item i in a range that is a length of l,
    for i in range(0, len(A), k):
        # Create an index range for l of n items:
        yield A[i:i+k]


def generateData():
    x = np.array(np.random.choice(2, total_series_no, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

def generate_data1():
    seq_length = 15
    inp_arr = []
    lbl_arr = []
    sub_arr = [1,1,1,1,1]
    for i in range(0,total_series_no//2):
        inp_a = np.random.choice(2,seq_length)
        rand_indx = random.randint(0,10)
        inp_a[rand_indx:rand_indx + 5] = sub_arr
        inp_arr.append(inp_a)
        # lbl_arr.append([1,0])
    for i in range(0,total_series_no//2):
        inp_a = np.random.choice(2,seq_length)
        inp_arr.append(inp_a)
        # lbl_arr.append([1,0])
    for a in inp_arr:
        if sub_list(sub_arr,a):
            lbl_arr.append([1,0])
        else:
            lbl_arr.append([0,1])
    # inp_arr, lbl_arr = shuffle(inp_arr, lbl_arr)
    return np.array(inp_arr),np.array(lbl_arr)

x,y = generate_data1()
# x,y = np.array(x), np.array(y)
print(x,x.shape)
print(y,y.shape)
# sys.exit()

batchX_placeholder = tf.compat.v1.placeholder(tf.float32, [batch_size,None,len(alphabets)])
# batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
y_lbl_placeholder = tf.compat.v1.placeholder(tf.int64, [None, num_classes])

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
# inputs_series = tf.split(batchX_placeholder, tf.shape(batchX_placeholder)[1], axis=1)
# labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes
lstm = tf.contrib.rnn.BasicLSTMCell(state_size)
cell = tf.contrib.rnn.MultiRNNCell([lstm for _ in range(num_layers)])
init_state = cell.zero_state(batch_size, tf.float32)

states_series, current_state = tf.nn.dynamic_rnn(cell, batchX_placeholder, initial_state=init_state)


logits_series = tf.matmul(tf.squeeze(states_series,axis=0),W2)
prediction_series = tf.nn.softmax(logits_series,axis=1)

# logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
# predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

output_logits = logits_series[-1]
y_pred = prediction_series[-1]

# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
#             for logits, labels in zip(logits_series,labels_series)]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_lbl_placeholder, logits=output_logits), name='loss')

# total_loss = tf.reduce_mean(losses)

train_step = tf.compat.v1.train.AdagradOptimizer(0.3).minimize(loss)

def plot(loss_list, y_pred, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)


    print('batchX = ',batchX)
    # print('batchY = ', batchY)
    cut_idx = index_of([1,1,1,1,1],batchX)
    print('batchX cut = ',batchX[cut_idx:])
    print('batchY cut = ', batchY[cut_idx:])

    y_preds = np.array([(1 if out[0] < 0.5 else 0) for out in y_pred])
    print('y_preds = ', y_preds)

def count_acc(y_pred, batchY):


        y_preds = np.array([(1 if out < 0.5 else 0) for out in y_pred])
        # print('batchX = ', batchX)
        # print('batchY = ', batchY)
        # print('y_preds = ', y_preds)

        return sum([1 for i in range(batchY.shape[0]) if batchY[i][1] == y_preds[i]])/batchY.shape[0]


    # for batch_series_idx in range(5):
    #     one_hot_output_series = np.array(y_pred)[:, batch_series_idx, :]
    #     single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
    #     # print('single_output_series = ', single_output_series)
	#
    #     plt.subplot(2, 3, batch_series_idx + 2)
    #     plt.cla()
    #     plt.axis([0, truncated_backprop_length, 0, 2])
    #     left_offset = range(truncated_backprop_length)
    #     plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
    #     plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
    #     plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    # plt.draw()
    # plt.pause(0.0001)

saver = tf.compat.v1.train.Saver()



with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.initialize_all_variables())
    saver.restore(sess, './my_test_model')

    res = []
    state_seq = []
    word_seq = []
    logit_seq = []
    lbls_seq = []
    reset_bool = []
    ind_seq = []

    loss_list = []
    x, y = generate_data1()
    x0 = convert_data_x(x, [1, 0])
    print('x = ', x)
    print('x0 = ', x0)

    for epoch_idx in range(1):

        x1, y1 = shuffle(x0, y)
        all_acc = 0

        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        batchesX = np.array(list(get_batches(x1,batch_size)))
        batchesY = np.array(list(get_batches(y1,batch_size)))

        num_batches = batchesX.shape[0]

        for batch_idx in range(num_batches):
            # start_idx = batch_idx * truncated_backprop_length
            # end_idx = start_idx + truncated_backprop_length

            # batchX = x[:,start_idx:end_idx]
            # batchY = y[:,start_idx:end_idx]

            batchX = batchesX[batch_idx]
            batchY = batchesY[batch_idx]

            _loss, _states_series, _current_state, _y_pred, _logits_series = sess.run(
                [loss, states_series, current_state, y_pred, logits_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    y_lbl_placeholder: batchY,
                    # init_state: _current_state
                    # cell_state: _current_cell_state,
                    # hidden_state: _current_hidden_state

                })


            reset_bool.append(1)
            state_seq.extend(_states_series[0])
            ind_seq.extend([0 for i in range(len(_states_series[0]) - 1)])
            ind_seq.append(1)
            co_batchX = convert_data_x_todigit(batchX,alphabets)
            word_seq.extend(co_batchX[0])
            logit_seq.extend(_logits_series)
            lbls_seq.extend(_y_pred)
            reset_bool.extend([0 for i in range(len(batchX[0]) - 1)])

            acc = count_acc(_y_pred, batchY)
            if batch_idx % 100 == 0:
                print('batch_idx, acc = ',batch_idx,acc)
            all_acc += acc
        print('all_acc = ', all_acc, num_batches, all_acc / num_batches)

    import pandas as pd
    print('printing...')
    df = pd.DataFrame()
    df['states'] =  state_seq
    df['words'] = word_seq
    df['logit_seq'] = logit_seq
    df['reset_bool'] = reset_bool
    df['ind_seq'] = ind_seq
    # df['state_list_sm'] = list(state_list_sm)
    df.to_csv('long_states_df.csv', index=False, header=True)

# plt.ioff()
# plt.show()