#! /usr/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
from cnn_class import cnn
import time
import scipy.io as sio
from sklearn.metrics import classification_report, roc_auc_score, auc, roc_curve, f1_score
from RnnAttention.attention import attention
from scipy import interp


def multiclass_roc_auc_score(y_true, y_score):
    assert y_true.shape == y_score.shape
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_true.shape[1]
    # compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # compute macro-average ROC curve and ROC area
    # First aggregate all false probtive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc

###########################################################################
# prepare raw data
###########################################################################
subject_id = 1
data_folder = '/home/dalinzhang/scratch/datasets/BCICIV_2a_gdf'
data = sio.loadmat(data_folder+"/cross_sub/cross_subject_data_"+str(subject_id)+".mat")
print("subject id ", subject_id)

test_X	= data["test_x"] # [trials, channels, time length]
train_X	= data["train_x"]

test_y	= data["test_y"].ravel()
train_y = data["train_y"].ravel()


train_y = np.asarray(pd.get_dummies(train_y), dtype = np.int8)
test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)

###########################################################################
# crop data
###########################################################################

window_size = 400
step = 50
n_channel = 22


def windows(data, size, step):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += step


def segment_signal_without_transition(data, window_size, step):
	segments = []
	for (start, end) in windows(data, window_size, step):
		if(len(data[start:end]) == window_size):
			segments = segments + [data[start:end]]
	return np.array(segments)


def segment_dataset(X, window_size, step):
	win_x = []
	for i in range(X.shape[0]):
		win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
	win_x = np.array(win_x)
	return win_x


train_raw_x = np.transpose(train_X, [0, 2, 1])
test_raw_x = np.transpose(test_X, [0, 2, 1])


train_win_x = segment_dataset(train_raw_x, window_size, step)
print("train_win_x shape: ", train_win_x.shape)
test_win_x = segment_dataset(test_raw_x, window_size, step)
print("test_win_x shape: ", test_win_x.shape)

# [trial, window, channel, time_length]
train_win_x = np.transpose(train_win_x, [0, 1, 3, 2])
print("train_win_x shape: ", train_win_x.shape)

test_win_x = np.transpose(test_win_x, [0, 1, 3, 2])
print("test_win_x shape: ", test_win_x.shape)


# [trial, window, channel, time_length, 1]
train_x = np.expand_dims(train_win_x, axis = 4)
test_x = np.expand_dims(test_win_x, axis = 4)

num_timestep = train_x.shape[1]
###########################################################################
# set model parameters
###########################################################################
# kernel parameter
kernel_height_1st	= 22
kernel_width_1st 	= 45

kernel_stride		= 1

conv_channel_num	= 40

# pooling parameter
pooling_height_1st 	= 1
pooling_width_1st 	= 75

pooling_stride_1st = 10

# full connected parameter
attention_size = 512
n_hidden_state = 64

###########################################################################
# set dataset parameters
###########################################################################
# input channel
input_channel_num = 1

# input height 
input_height = train_x.shape[2]

# input width
input_width = train_x.shape[3]

# prediction class
num_labels = 4
###########################################################################
# set training parameters
###########################################################################
# set learning rate
learning_rate = 1e-4

# set maximum traing epochs
training_epochs = 200

# set batch size
batch_size = 10

# set dropout probability
dropout_prob = 0.5

# set train batch number per epoch
batch_num_per_epoch = train_x.shape[0]//batch_size

# instance cnn class
padding = 'VALID'

cnn_2d = cnn(padding=padding)

# input placeholder
X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name = 'X')
Y = tf.placeholder(tf.float32, shape=[None, num_labels], name = 'Y')
train_phase = tf.placeholder(tf.bool, name = 'train_phase')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# first CNN layer
conv_1 = cnn_2d.apply_conv2d(X, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride, train_phase)
print("conv 1 shape: ", conv_1.get_shape().as_list())
pool_1 = cnn_2d.apply_max_pooling(conv_1, pooling_height_1st, pooling_width_1st, pooling_stride_1st)
print("pool 1 shape: ", pool_1.get_shape().as_list())

pool1_shape = pool_1.get_shape().as_list()
pool1_flat = tf.reshape(pool_1, [-1, pool1_shape[1]*pool1_shape[2]*pool1_shape[3]])

fc_drop = tf.nn.dropout(pool1_flat, keep_prob)	

lstm_in = tf.reshape(fc_drop, [-1, num_timestep, pool1_shape[1]*pool1_shape[2]*pool1_shape[3]])

########################## RNN ########################
cells = []
for _ in range(2):
	cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_state, forget_bias=1.0, state_is_tuple=True)
	cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
	cells.append(cell)
lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

# output ==> [batch, step, n_hidden_state]
rnn_op, states = tf.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state=init_state, time_major=False)

########################## attention ########################
with tf.name_scope('Attention_layer'):
    attention_op, alphas = attention(rnn_op, attention_size, time_major = False, return_alphas=True)

attention_drop = tf.nn.dropout(attention_op, keep_prob)	

########################## readout ########################
y_ = cnn_2d.apply_readout(attention_drop, rnn_op.shape[2].value, num_labels)

# probability prediction 
y_prob = tf.nn.softmax(y_, name = "y_prob")

# class prediction 
y_pred = tf.argmax(y_prob, 1, name = "y_pred")

########################## loss and optimizer ########################
# cross entropy cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name = 'loss')


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	# set training SGD optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))

########################## define accuracy ########################
# calculate prediction accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')


###########################################################################
# train test and save result
###########################################################################

# run with gpu memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

train_acc = []
test_acc = []
best_test_acc = []
train_loss = []
with tf.Session(config=config) as session:
	session.run(tf.global_variables_initializer())
	best_acc = 0
	for epoch in range(training_epochs):
		pred_test = np.array([])
		true_test = []
		prob_test = []
		########################## training process ########################
		for b in range(batch_num_per_epoch):
			offset = (b * batch_size) % (train_y.shape[0] - batch_size) 
			batch_x = train_x[offset:(offset + batch_size), :, :, :, :]
			batch_x = batch_x.reshape([len(batch_x)*num_timestep, n_channel, window_size, 1])
			batch_y = train_y[offset:(offset + batch_size), :]
			_, c = session.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1-dropout_prob, train_phase: True})
		# calculate train and test accuracy after each training epoch
		if(epoch%1 == 0):
			train_accuracy 	= np.zeros(shape=[0], dtype=float)
			test_accuracy	= np.zeros(shape=[0], dtype=float)
			train_l 		= np.zeros(shape=[0], dtype=float)
			test_l			= np.zeros(shape=[0], dtype=float)
			# calculate train accuracy after each training epoch
			for i in range(batch_num_per_epoch):
				########################## prepare training data ########################
				offset = (i * batch_size) % (train_y.shape[0] - batch_size) 
				train_batch_x = train_x[offset:(offset + batch_size), :, :, :]
				train_batch_x = train_batch_x.reshape([len(train_batch_x)*num_timestep, n_channel, window_size, 1])
				train_batch_y = train_y[offset:(offset + batch_size), :]

				########################## calculate training results ########################
				train_a, train_c = session.run([accuracy, cost], feed_dict={X: train_batch_x, Y: train_batch_y, keep_prob: 1.0, train_phase: False})
				
				train_l = np.append(train_l, train_c)
				train_accuracy = np.append(train_accuracy, train_a)
			print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Training Cost: ", np.mean(train_l), "Training Accuracy: ", np.mean(train_accuracy))
			train_acc = train_acc + [np.mean(train_accuracy)]
			train_loss = train_loss + [np.mean(train_l)]
			# calculate test accuracy after each training epoch
			for j in range(batch_num_per_epoch):
				########################## prepare test data ########################
				offset = (j * batch_size) % (test_y.shape[0] - batch_size) 
				test_batch_x = test_x[offset:(offset + batch_size), :, :, :]
				test_batch_x = test_batch_x.reshape([len(test_batch_x)*num_timestep, n_channel, window_size, 1])
				test_batch_y = test_y[offset:(offset + batch_size), :]
				
				########################## calculate test results ########################
				test_a, test_c, prob_v, pred_v = session.run([accuracy, cost, y_prob, y_pred], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, train_phase: False})
				
				test_accuracy = np.append(test_accuracy, test_a)
				test_l = np.append(test_l, test_c)
				pred_test = np.append(pred_test, pred_v)
				true_test.append(test_batch_y)
				prob_test.append(prob_v)
			if np.mean(test_accuracy) > best_acc :
				best_acc = np.mean(test_accuracy)
			true_test = np.array(true_test).reshape([-1, num_labels])
			prob_test = np.array(prob_test).reshape([-1, num_labels])
			auc_roc_test = multiclass_roc_auc_score(y_true=true_test, y_score=prob_test)
			f1 = f1_score(y_true=np.argmax(true_test, axis = 1), y_pred=pred_test, average = 'macro')
			print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, "Test Cost: ", np.mean(test_l), 
																					  "Test Accuracy: ", np.mean(test_accuracy), 
																					  "Test f1: ", f1, 
																					  "Test AUC: ", auc_roc_test['macro'], "\n")


















