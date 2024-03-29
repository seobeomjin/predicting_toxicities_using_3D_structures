import os, sys
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, recall_score
import matplotlib.pyplot as plt
import time
from rdkit.Chem import AllChem
from rdkit import Chem
#tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
# ============ multitask classifier =================

# load data =========================================
print('start loading data')
dataX = np.load('tox21_fp.npy')
dataY_concat  = np.load('tox21_Y.npy', allow_pickle=True)
index = np.load('tox21_index.npy', allow_pickle=True)
print(dataX.shape)

dataX_concat = []
for j in range(12):
    dataX_concat.append([dataX[i] for i in index[j]])

print('loading data is done!')
# ===================================================
start_time = time.time()

# hyperparameters
batch_size = 32
Max_len = 200 # for padding
embedding_size = 200
n_hid = 1024 # number of feature maps
win_size = 5 # window size of kernel
lr = 1e-4 # learning rate of optimzier

# lookup table
bit_size = 1024 # circular fingerprint
emb = tf.Variable(tf.random.uniform([bit_size, embedding_size], -1, 1), dtype=tf.float32)
pads = tf.constant([[1,0], [0,0]])
embeddings = tf.pad(emb, pads)

# input shape pre-processing ========================
data_x_concat = []
data_y_concat = []

for k in range(12): # k means each tasks
    data_x = [] # init it newly for each tasks 
    data_y = [] 
    for i in range(len(dataX_concat[k])):
        fp = [0] * Max_len # eg) [0,0,0,0, ..., 0] , length = 200 
        n_ones = 0
        for j in range(bit_size): # bit_size 1024 (circular fp)
            if dataX_concat[k][i][j] == 1: # in k_th task, i_th data, j_th index is ,, 
                fp[n_ones] = j+1 # eg) [0,23,45,786,1000,1001,0,0,...0] length = 200 
                n_ones += 1
        data_x.append(fp) # append i_th in k_th task 
        data_y.append([dataY_concat[k][i]]) 
    data_x = np.array(data_x, dtype=np.int32) 
    data_y = np.array(data_y, dtype=np.float32)
    data_x_concat.append(data_x) # append k_th tasks if it finished getting fp in k_th task
    data_y_concat.append(data_y)

# k(0-11) task inform would be maintained in training
train_x_concat, test_x_concat, train_y_concat, test_y_concat, valid_x_concat, valid_y_concat = [], [], [], [], [], []
for k in range(12): # there are each splitted dataset for each tasks 
    train_x, test_x, train_y, test_y = train_test_split(data_x_concat[k], data_y_concat[k], test_size=0.1) # whole data > (train / test)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111) # train data > (train / valid) 
    train_x_concat.append(train_x)
    test_x_concat.append(test_x)
    train_y_concat.append(train_y)
    test_y_concat.append(test_y)
    valid_x_concat.append(valid_x)
    valid_y_concat.append(valid_y)
    # "blah_concat" is that all tasks are remained 
    
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)
    print(valid_x.shape, valid_y.shape)

train_size = [len(train_x_concat[i]) for i in range(len(train_x_concat))] # getting each task size for train
test_size = [len(test_x_concat[i]) for i in range(len(test_x_concat))] # gettinf each test size for test 

##################################################################
# ================== CNN model construction ======================
##################################################################

def init_weights(shape):
    return tf.Variable(tf.compat.v1.random.normal(shape, stddev = 0.01))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))

p_keep_conv = tf.compat.v1.placeholder(dtype=tf.float32)

class model():
    def __init__(self, embedding_size, n_hid, win_size, p_keep_conv, Max_len):
         self.Max_len = Max_len
         self.nhid  = n_hid
         self.kernel_size = win_size
         self.w2 = init_weights([self.kernel_size, embedding_size, 1, self.nhid]) # 64
         self.w_o = init_weights([self.nhid, 1])
         
         self.b2 = bias_variable([1, self.nhid])
         self.b_o = bias_variable([1])
         self.p_keep_conv = p_keep_conv
             
    def conv_model(self, X): 
        self.l2_1 = tf.nn.relu(tf.nn.conv2d(X, self.w2, strides=[1, 1, 1, 1], padding='VALID', dilations=[1,1,1,1]) + self.b2)
        self.l2_2 = tf.squeeze(self.l2_1, [2])
        self.l2_3 = tf.nn.pool(self.l2_2, window_shape=[self.Max_len-self.kernel_size+1], pooling_type='MAX', padding='VALID')
        self.l2_4 = tf.nn.dropout(self.l2_3, 1-self.p_keep_conv)         
        self.lout = tf.reshape(self.l2_4, [-1, self.w_o.get_shape().as_list()[0]])
        return self.lout

X = tf.compat.v1.placeholder(tf.int32, [None, Max_len])
Y = tf.compat.v1.placeholder(tf.float32, [None, 1])
X_em = tf.nn.embedding_lookup(embeddings, X)
X_em = tf.reshape(X_em, [-1, Max_len, embedding_size, 1])

model = model(embedding_size, n_hid, win_size, p_keep_conv, Max_len)

py_x = model.conv_model(X_em)

# ==================== fully connecte layers ===================== #
temp_hid = n_hid
w1 = init_weights([temp_hid, 1])
w2 = init_weights([temp_hid, 1])
w3 = init_weights([temp_hid, 1])
w4 = init_weights([temp_hid, 1])
w5 = init_weights([temp_hid, 1])
w6 = init_weights([temp_hid, 1])
w7 = init_weights([temp_hid, 1])
w8 = init_weights([temp_hid, 1])
w9 = init_weights([temp_hid, 1])
w10 = init_weights([temp_hid, 1])
w11 = init_weights([temp_hid, 1])
w12 = init_weights([temp_hid, 1])

b1 = bias_variable([1])
b2 = bias_variable([1])
b3 = bias_variable([1])
b4 = bias_variable([1])
b5 = bias_variable([1])
b6 = bias_variable([1])
b7 = bias_variable([1])
b8 = bias_variable([1])
b9 = bias_variable([1])
b10 = bias_variable([1])
b11 = bias_variable([1])
b12 = bias_variable([1])

py_x1 = tf.sigmoid(tf.matmul(py_x, w1) + b1)
py_x2 = tf.sigmoid(tf.matmul(py_x, w2) + b2)
py_x3 = tf.sigmoid(tf.matmul(py_x, w3) + b3)
py_x4 = tf.sigmoid(tf.matmul(py_x, w4) + b4)
py_x5 = tf.sigmoid(tf.matmul(py_x, w5) + b5)
py_x6 = tf.sigmoid(tf.matmul(py_x, w6) + b6)
py_x7 = tf.sigmoid(tf.matmul(py_x, w7) + b7)
py_x8 = tf.sigmoid(tf.matmul(py_x, w8) + b8)
py_x9 = tf.sigmoid(tf.matmul(py_x, w9) + b9)
py_x10 = tf.sigmoid(tf.matmul(py_x, w10) + b10)
py_x11 = tf.sigmoid(tf.matmul(py_x, w11) + b11)
py_x12 = tf.sigmoid(tf.matmul(py_x, w12) + b12)

cost1 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x1)
cost2 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x2)
cost3 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x3)
cost4 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x4)
cost5 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x5)
cost6 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x6)
cost7 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x7)
cost8 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x8)
cost9 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x9)
cost10 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x10)
cost11 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x11)
cost12 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x12)

###### Are convolution weights also trained by these codes??? ######
train_op1 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost1) 
train_op2 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost2)
train_op3 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost3)
train_op4 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost4)
train_op5 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost5)
train_op6 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost6)
train_op7 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost7)
train_op8 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost8)
train_op9 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost9)
train_op10 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost10)
train_op11 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost11)
train_op12 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost12)

prediction_error1 = cost1
prediction_error2 = cost2
prediction_error3 = cost3
prediction_error4 = cost4
prediction_error5 = cost5
prediction_error6 = cost6
prediction_error7 = cost7
prediction_error8 = cost8
prediction_error9 = cost9
prediction_error10 = cost10
prediction_error11 = cost11
prediction_error12 = cost12


##################################################################
# ==================== training part =============================
##################################################################

SAVER_DIR = "model_tox21_multi"
saver = tf.compat.v1.train.Saver()
ckpt_path = os.path.join(SAVER_DIR, "model_tox21_multi")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    best_auc = 0
    best_idx = 0
    for i in range(80): # epoch
        training_batch = zip(range(0, len(train_x_concat[0]), batch_size),
                             range(batch_size, len(train_x_concat[0])+1, batch_size))
   #for start, end in tqdm.tqdm(training_batch):
        for start, end in training_batch:
            sess.run(train_op1, feed_dict={X: train_x_concat[0][start:end], Y: train_y_concat[0][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op2, feed_dict={X: train_x_concat[1][start:end], Y: train_y_concat[1][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op3, feed_dict={X: train_x_concat[2][start:end], Y: train_y_concat[2][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op4, feed_dict={X: train_x_concat[3][start:end], Y: train_y_concat[3][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op5, feed_dict={X: train_x_concat[4][start:end], Y: train_y_concat[4][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op6, feed_dict={X: train_x_concat[5][start:end], Y: train_y_concat[5][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op7, feed_dict={X: train_x_concat[6][start:end], Y: train_y_concat[6][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op8, feed_dict={X: train_x_concat[7][start:end], Y: train_y_concat[7][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op9, feed_dict={X: train_x_concat[8][start:end], Y: train_y_concat[8][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op10, feed_dict={X: train_x_concat[9][start:end], Y: train_y_concat[9][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op11, feed_dict={X: train_x_concat[10][start:end], Y: train_y_concat[10][start:end] ,p_keep_conv: 0.5})
            sess.run(train_op12, feed_dict={X: train_x_concat[11][start:end], Y: train_y_concat[11][start:end] ,p_keep_conv: 0.5})
            #sess.run(function what I want to run, feed_dict = {data for training and some params})

   # print validation loss
        merr = sess.run(prediction_error1, feed_dict={X: valid_x_concat[0], Y: valid_y_concat[0], p_keep_conv: 1.0}) #prediction_error1 means cost1
        print(i, merr,  end = ' ')
        merr = sess.run(prediction_error2, feed_dict={X: valid_x_concat[1], Y: valid_y_concat[1], p_keep_conv: 1.0})
        print(merr, end = ' ')
#        print(sess.run([model.l2_1, model.l2_2, model.l2_3, model.l2_4, model.lout], feed_dict={X: valid_x_concat[1], Y: valid_y_concat[1], p_keep_conv: 1.0}))
        merr = sess.run(prediction_error3, feed_dict={X: valid_x_concat[2], Y: valid_y_concat[2], p_keep_conv: 1.0})
        print(merr, end = ' ')
        merr = sess.run(prediction_error4, feed_dict={X: valid_x_concat[3], Y: valid_y_concat[3], p_keep_conv: 1.0})
        print(merr, end = ' ')
        merr = sess.run(prediction_error5, feed_dict={X: valid_x_concat[4], Y: valid_y_concat[4], p_keep_conv: 1.0})
        print(merr, end = ' ')
        merr = sess.run(prediction_error6, feed_dict={X: valid_x_concat[5], Y: valid_y_concat[5], p_keep_conv: 1.0})
        print(merr, end = ' ')
        merr = sess.run(prediction_error7, feed_dict={X: valid_x_concat[6], Y: valid_y_concat[6], p_keep_conv: 1.0})
        print(merr, end = ' ')
        merr = sess.run(prediction_error8, feed_dict={X: valid_x_concat[7], Y: valid_y_concat[7], p_keep_conv: 1.0})
        print(merr, end = ' ')
        merr = sess.run(prediction_error9, feed_dict={X: valid_x_concat[8], Y: valid_y_concat[8], p_keep_conv: 1.0})
        print(merr, end = ' ')
        merr = sess.run(prediction_error10, feed_dict={X: valid_x_concat[9], Y: valid_y_concat[9], p_keep_conv: 1.0})
        print(merr, end = ' ')
        merr = sess.run(prediction_error11, feed_dict={X: valid_x_concat[10], Y: valid_y_concat[10], p_keep_conv: 1.0})
        print(merr, end = ' ')
        merr = sess.run(prediction_error12, feed_dict={X: valid_x_concat[11], Y: valid_y_concat[11], p_keep_conv: 1.0})
        print(merr)


# calculate auc 
        val_preds1 = sess.run(py_x1, feed_dict={X: valid_x_concat[0], p_keep_conv: 1})
        val_preds2 = sess.run(py_x2, feed_dict={X: valid_x_concat[1], p_keep_conv: 1})
        val_preds3 = sess.run(py_x3, feed_dict={X: valid_x_concat[2], p_keep_conv: 1})
        val_preds4 = sess.run(py_x4, feed_dict={X: valid_x_concat[3], p_keep_conv: 1})
        val_preds5 = sess.run(py_x5, feed_dict={X: valid_x_concat[4], p_keep_conv: 1})
        val_preds6 = sess.run(py_x6, feed_dict={X: valid_x_concat[5], p_keep_conv: 1})
        val_preds7 = sess.run(py_x7, feed_dict={X: valid_x_concat[6], p_keep_conv: 1})
        val_preds8 = sess.run(py_x8, feed_dict={X: valid_x_concat[7], p_keep_conv: 1})
        val_preds9 = sess.run(py_x9, feed_dict={X: valid_x_concat[8], p_keep_conv: 1})
        val_preds10 = sess.run(py_x10, feed_dict={X: valid_x_concat[9], p_keep_conv: 1})
        val_preds11 = sess.run(py_x11, feed_dict={X: valid_x_concat[10], p_keep_conv: 1})
        val_preds12 = sess.run(py_x12, feed_dict={X: valid_x_concat[11], p_keep_conv: 1})

        val_aucs1 = roc_auc_score(valid_y_concat[0], val_preds1)
        val_aucs2 = roc_auc_score(valid_y_concat[1], val_preds2)
        val_aucs3 = roc_auc_score(valid_y_concat[2], val_preds3)
        val_aucs4 = roc_auc_score(valid_y_concat[3], val_preds4)
        val_aucs5 = roc_auc_score(valid_y_concat[4], val_preds5)
        val_aucs6 = roc_auc_score(valid_y_concat[5], val_preds6)
        val_aucs7 = roc_auc_score(valid_y_concat[6], val_preds7)
        val_aucs8 = roc_auc_score(valid_y_concat[7], val_preds8)
        val_aucs9 = roc_auc_score(valid_y_concat[8], val_preds9)
        val_aucs10 = roc_auc_score(valid_y_concat[9], val_preds10)
        val_aucs11 = roc_auc_score(valid_y_concat[10], val_preds11)
        val_aucs12 = roc_auc_score(valid_y_concat[11], val_preds12)

        val_aucs = [val_aucs1, val_aucs2, val_aucs3, val_aucs4, val_aucs5, val_aucs6, val_aucs7, val_aucs8, val_aucs9, val_aucs10, val_aucs11, val_aucs12]

        print('mean validation auc: ', end = ' ')
        print(np.mean(val_aucs))

        if best_auc < np.mean(val_aucs):
            auc1 = val_aucs1
            auc2 = val_aucs2
            auc3 = val_aucs3
            auc4 = val_aucs4
            auc5 = val_aucs5
            auc6 = val_aucs6
            auc7 = val_aucs7
            auc8 = val_aucs8
            auc9 = val_aucs9
            auc10 = val_aucs10
            auc11 = val_aucs11
            auc12 = val_aucs12
            best_auc = np.mean(val_aucs)
            best_idx = i
            save_path = saver.save(sess, ckpt_path, global_step = best_idx)
            print('model saved!')
            print()

print('best epoch index: '+str(best_idx))
print('best valid auc total: '+str(best_auc))
print('best valid auc nr-ar: '+str(auc1))
print('best valid auc nr-ar-lbd: '+str(auc2))
print('best valid auc nr-ahr: '+str(auc3))
print('best valid auc nr-aromatase: '+str(auc4))
print('best valid auc nr-er: '+str(auc5))
print('best valid auc nr-er-lbd: '+str(auc6))
print('best valid auc nr-ppar-gamma: '+str(auc7))
print('best valid auc sr-are: '+str(auc8))
print('best valid auc sr-atad5: '+str(auc9))
print('best valid auc sr-hse: '+str(auc10))
print('best valid auc sr-mmp: '+str(auc11))
print('best valid auc sr-p53: '+str(auc12))
print("=== %s seconds ===" % (time.time() - start_time))


####################################################################
#=========================== test part ============================#
####################################################################
saver = tf.compat.v1.train.Saver()
ckpt_path = os.path.join(SAVER_DIR, "model_tox21_multi")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("model loaded successfully!")

# test set
    preds1 = sess.run(py_x1, feed_dict={X: test_x_concat[0], p_keep_conv: 1})
    preds2 = sess.run(py_x2, feed_dict={X: test_x_concat[1], p_keep_conv: 1})
    preds3 = sess.run(py_x3, feed_dict={X: test_x_concat[2], p_keep_conv: 1})
    preds4 = sess.run(py_x4, feed_dict={X: test_x_concat[3], p_keep_conv: 1})
    preds5 = sess.run(py_x5, feed_dict={X: test_x_concat[4], p_keep_conv: 1})
    preds6 = sess.run(py_x6, feed_dict={X: test_x_concat[5], p_keep_conv: 1})
    preds7 = sess.run(py_x7, feed_dict={X: test_x_concat[6], p_keep_conv: 1})
    preds8 = sess.run(py_x8, feed_dict={X: test_x_concat[7], p_keep_conv: 1})
    preds9 = sess.run(py_x9, feed_dict={X: test_x_concat[8], p_keep_conv: 1})
    preds10 = sess.run(py_x10, feed_dict={X: test_x_concat[9], p_keep_conv: 1})
    preds11 = sess.run(py_x11, feed_dict={X: test_x_concat[10], p_keep_conv: 1})
    preds12 = sess.run(py_x12, feed_dict={X: test_x_concat[11], p_keep_conv: 1})

    aucs1 = roc_auc_score(test_y_concat[0], preds1)
    aucs2 = roc_auc_score(test_y_concat[1], preds2)
    aucs3 = roc_auc_score(test_y_concat[2], preds3)
    aucs4 = roc_auc_score(test_y_concat[3], preds4)
    aucs5 = roc_auc_score(test_y_concat[4], preds5)
    aucs6 = roc_auc_score(test_y_concat[5], preds6)
    aucs7 = roc_auc_score(test_y_concat[6], preds7)
    aucs8 = roc_auc_score(test_y_concat[7], preds8)
    aucs9 = roc_auc_score(test_y_concat[8], preds9)
    aucs10 = roc_auc_score(test_y_concat[9], preds10)
    aucs11 = roc_auc_score(test_y_concat[10], preds11)
    aucs12 = roc_auc_score(test_y_concat[11], preds12)

    aucs = [aucs1, aucs2, aucs3, aucs4, aucs5, aucs6, aucs7, aucs8, aucs9, aucs10, aucs11, aucs12]
    test_auc = np.mean(aucs)

    print('test auc total: '+str(test_auc))
    print('test auc nr-ar: '+str(aucs1))
    print('test auc nr-ar-lbd: '+str(aucs2))
    print('test auc nr-ahr: '+str(aucs3))
    print('test auc nr-aromatase: '+str(aucs4))
    print('test auc nr-er: '+str(aucs5))
    print('test auc nr-er-lbd: '+str(aucs6))
    print('test auc nr-ppar-gamma: '+str(aucs7))
    print('test auc sr-are: '+str(aucs8))
    print('test auc sr-atad5: '+str(aucs9))
    print('test auc sr-hse: '+str(aucs10))
    print('test auc sr-mmp: '+str(aucs11))
    print('test auc sr-p53: '+str(aucs12))
