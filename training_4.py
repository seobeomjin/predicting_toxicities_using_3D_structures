import numpy as np
np.random.seed(123)

import pandas as pd
from math import sqrt, ceil

import h5py

from sklearn.utils import shuffle
import tensorflow as tf

from tfbio.data import Featurizer, make_grid, rotate
import net_3 as net ## custom network for predicting ic50

import os.path

import matplotlib as mpl
mpl.use('agg')

import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
sns.set_color_codes()
color = {'training': 'b', 'validation': 'g', 'test': 'r'}

import time
timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')


datasets = ['training', 'validation', 'test']


def input_dir(path):
    """Check if input directory exists and contains all needed files"""
    global datasets

    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise IOError('Incorrect input_dir specified: no such directory')
    for dataset_name in datasets:
        dataset_path = os.path.join(path, '%s_set.hdf' % dataset_name)
        if not os.path.exists(dataset_path):
            raise IOError('Incorrect input_dir specified:'
                          ' %s set file not found' % dataset_path)
    return path

import argparse
parser = argparse.ArgumentParser(
    description='Train 3D colnvolutional neural network on toxicity data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

io_group = parser.add_argument_group('I/O')
io_group.add_argument('--input_dir', '-i', required=True, type=input_dir,
                      help='directory with training, validation and test sets')
io_group.add_argument('--log_dir', '-l', default='./logdir/',
                      help='directory to store tensorboard summaries')
io_group.add_argument('--output_prefix', '-o', default='./output',
                      help='prefix for checkpoints, predictions and plots')
io_group.add_argument('--grid_spacing', '-g', default=1.0, type=float,
                      help='distance between grid points')
io_group.add_argument('--max_dist', '-d', default=10.0, type=float,
                      help='max distance from complex center')

arc_group = parser.add_argument_group('Netwrok architecture')
arc_group.add_argument('--conv_patch', default=5, type=int,
                       help='patch size for convolutional layers')
arc_group.add_argument('--pool_patch', default=2, type=int,
                       help='patch size for pooling layers')
arc_group.add_argument('--conv_channels', metavar='C', default=[64, 128, 256],
                       type=int, nargs='+',
                       help='number of fileters in convolutional layers')
arc_group.add_argument('--dense_sizes', metavar='D', default=[1000, 500, 200],
                       type=int, nargs='+',
                       help='number of neurons in dense layers')

reg_group = parser.add_argument_group('Regularization')
reg_group.add_argument('--keep_prob', dest='kp', default=0.5, type=float,
                       help='keep probability for dropout')
reg_group.add_argument('--l2', dest='lmbda', default=0.001, type=float,
                       help='lambda for weight decay')
reg_group.add_argument('--rotations', metavar='R', default=list(range(24)),
                       type=int, nargs='+',
                       help='rotations to perform')

tr_group = parser.add_argument_group('Training')
tr_group.add_argument('--learning_rate', default=1e-5, type=float,
                      help='learning rate')
tr_group.add_argument('--batch_size', default=20, type=int,
                      help='batch size')
tr_group.add_argument('--num_epochs', default=20, type=int,
                      help='number of epochs')
tr_group.add_argument('--num_checkpoints', dest='to_keep', default=10, type=int,
                      help='number of checkpoints to keep')

args = parser.parse_args()

prefix = os.path.abspath(args.output_prefix) + '-' + timestamp
logdir = os.path.join(os.path.abspath(args.log_dir), os.path.split(prefix)[1])

featurizer = Featurizer()

print('\n---- FEATURES ----\n')
print('atomic properties:', featurizer.FEATURE_NAMES)

columns = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}

ids = {}
toxicity = {}
coords = {}
features = {}

splitted_datasets = ['training1', 'training2','validation1','validation2', 'test1','test2']
protein_list = ['andro','estro']

splitted_ids = {}
splitted_toxicity = {}
splitted_coords = {}
splitted_features = {}

for dictionary in [ids, toxicity, coords, features]:
    for dataset_name in datasets:
        dictionary[dataset_name] = []

for dictionary in [splitted_ids, splitted_toxicity, splitted_coords, splitted_features]:
    for splitted_dataset_name in splitted_datasets:
        dictionary[splitted_dataset_name] = []

for dataset_name in datasets:
    dataset_path = os.path.join(input_dir, '%s_set.hdf' % dataset_name)
    with h5py.File(dataset_path, 'r') as f:
        for pdb_id in f: #pdb_id  >>> androgenSDF0
            dataset = f[pdb_id]
            for i in range(len(protein_list)):
                if protein_list[i] in pdb_id : 
                    splitted_coords[dataset_name + str(i+1)].append(dataset[:, :3])
                    splitted_features[dataset_name + str(i+1)].append(dataset[:, 3:])
                    splitted_toxicity[dataset_name + str(i+1)].append(dataset.attrs['toxicity'])
                    splitted_ids[dataset_name + str(i+1)].append(pdb_id) 
    
for k in splitted_ids.keys():
    splitted_ids[k] = np.array(splitted_ids[k])
    splitted_toxicity[k] = np.reshape(splitted_toxicity[k], (-1, 1))


# normalize charges
# for task 1
charges1 = []
for feature_data in splitted_features['training1']:
    charges1.append(feature_data[..., columns['partialcharge']])

charges1 = np.concatenate([c.flatten() for c in charges1])

m1 = charges1.mean()
std1 = charges1.std()
print('charges1: mean=%s, sd=%s' % (m1, std1))
print('use sd as scaling factor')

# for task2
charges2 = []
for feature_data in features['training2']:
    charges2.append(feature_data[..., columns['partialcharge']])

charges2 = np.concatenate([c.flatten() for c in charges2])

m2 = charges2.mean()
std2 = charges2.std()
print('charges2: mean=%s, sd=%s' % (m2, std2))
print('use sd as scaling factor')


def get_batch_1(dataset_name, indices, rotation=0):
    global splitted_coords, splitted_features, std1
    x = []
    for i, idx in enumerate(indices):
        coords_idx = rotate(splitted_coords[dataset_name][idx], rotation)
        features_idx = splitted_features[dataset_name][idx]
        x.append(make_grid(coords_idx, features_idx,
                 grid_resolution=args.grid_spacing,
                 max_dist=args.max_dist))
    x = np.vstack(x)
    x[..., columns['partialcharge']] /= std1
    return x

def get_batch_2(dataset_name, indices, rotation=0):
    global splitted_coords, splitted_features, std1
    x = []
    for i, idx in enumerate(indices):
        coords_idx = rotate(splitted_coords[dataset_name][idx], rotation)
        features_idx = splitted_features[dataset_name][idx]
        x.append(make_grid(coords_idx, features_idx,
                 grid_resolution=args.grid_spacing,
                 max_dist=args.max_dist))
    x = np.vstack(x)
    x[..., columns['partialcharge']] /= std2
    return x


print('\n---- DATA ----\n')

tmp = get_batch_1('training', range(min(50, len(features['training']))))

assert ((tmp[:, :, :, :, columns['molcode']] == 0.0).any()
        and (tmp[:, :, :, :, columns['molcode']] == 1.0).any()
        and (tmp[:, :, :, :, columns['molcode']] == -1.0).any()).all()

idx1 = [[i[0]] for i in np.where(tmp[:, :, :, :, columns['molcode']] == 1.0)]
idx2 = [[i[0]] for i in np.where(tmp[:, :, :, :, columns['molcode']] == -1.0)]

print('\nexamples:')
for mtype, mol in [['ligand', tmp[idx1]], ['protein', tmp[idx2]]]:
    print(' ', mtype)
    for name, num in columns.items():
        print('  ', name, mol[0, num])
    print('')


# Best error we can get without any training (MSE from training set mean):
# task 1
t1_baseline = ((splitted_toxicity['training1'] - splitted_toxicity['training1'].mean()) ** 2.0).mean()
v1_baseline = ((splitted_toxicity['validation1'] - splitted_toxicity['training1'].mean()) ** 2.0).mean()
print('baseline mse1: training1 =%s, validation1=%s' % (t1_baseline, v1_baseline))

# task 1
t2_baseline = ((splitted_toxicity['training2'] - splitted_toxicity['training2'].mean()) ** 2.0).mean()
v2_baseline = ((splitted_toxicity['validation2'] - splitted_toxicity['training2'].mean()) ** 2.0).mean()
print('baseline mse1: training2 =%s, validation2=%s' % (t2_baseline, v2_baseline))


# NET PARAMS

ds_sizes = {dataset: len(splitted_toxicity[dataset]) for dataset in splitted_datasets}
_, isize, *_, in_chnls = get_batch_1('training1', [0]).shape
osize = 1
# ds_sizes > {'training': 1555, 'validation': 20, 'test': 20}
# _ > [21,21]
# isize > 21 
# *_ > 21 21
# in_chnls > 19 

for set_name, set_size in ds_sizes.items():
    print('%s %s samples' % (set_size, set_name))

num_batches = {dataset: ceil(size / args.batch_size) 
               for dataset, size in ds_sizes.items()}
# training : ceil[1555 / 20]
# test : ceil [20 / 20] 
# valid : ceil [20/20] 이런 식으로 

# num_batches > {'training': 78, 'validation': 1, 'test': 1}

print('\n---- MODEL ----\n')
print((isize - 1) * args.grid_spacing, 'A box')
print(in_chnls, 'features')
print('')
print('convolutional layers: %s channels, %sA patch + max pooling with %sA patch'
      % (', '.join((str(i) for i in args.conv_channels)), args.conv_patch,
         args.pool_patch))
print('fully connected layers:', ', '.join((str(i) for i in args.dense_sizes)),
      'neurons')
print('regularization: dropout (keep %s) and L2 (lambda %s)'
      % (args.kp, args.lmbda))
print('')
print('learning rate', args.learning_rate)
print(num_batches['training'], 'batches,', args.batch_size, 'examples each')
print(num_batches['validation'], 'validation batches')
print(num_batches['test'], 'test batches')
print('')
print(args.num_epochs, 'epochs, best', args.to_keep, 'saved')

graph = net.make_SB_network(isize=isize, in_chnls=in_chnls, osize=osize,
                                  conv_patch=args.conv_patch,
                                  pool_patch=args.pool_patch,
                                  conv_channels=args.conv_channels,
                                  dense_sizes=args.dense_sizes,
                                  lmbda=args.lmbda,
                                  learning_rate=args.learning_rate)


train_writer = tf.summary.FileWriter(os.path.join(logdir, 'training_set'),
                                     graph, flush_secs=1)
val_writer = tf.summary.FileWriter(os.path.join(logdir, 'validation_set'),
                                   flush_secs=1)

net_summaries, training_summaries = net.make_summaries_SB(graph)

x = graph.get_tensor_by_name('input/structure:0') # <tf.Tensor 'input/structure:0' shape=(?, 21, 21, 21, 19) dtype=float32>
y1 = graph.get_tensor_by_name('output/prediction1:0')
y2 = graph.get_tensor_by_name('output/prediction2:0')
t1 = graph.get_tensor_by_name('input/toxicity1:0')
t2 = graph.get_tensor_by_name('input/toxicity2:0')
keep_prob = graph.get_tensor_by_name('fully_connected/keep_prob:0')

train1 = graph.get_tensor_by_name('training/train1:0')  
train2 = graph.get_tensor_by_name('training/train2:0')  
# graph.get_tensor_by_name => bring tensors from a certain variable scope by using name
# this code is in the vriable_scope('training') in net_3.py 
#    >>> train = optimizer.minimize(cost, global_step=global_step,name='train')
#
mse1 = graph.get_tensor_by_name('training/mse1:0')
mse2 = graph.get_tensor_by_name('training/mse2:0')
feature_importance = graph.get_tensor_by_name('net_properties/feature_importance:0')
global_step = graph.get_tensor_by_name('training/global_step:0')

convs = '_'.join((str(i) for i in args.conv_channels))
fcs = '_'.join((str(i) for i in args.dense_sizes))

with graph.as_default():
    saver = tf.train.Saver(max_to_keep=args.to_keep)


def batches(set_name):
    """Batch generator, yields slice indices"""
    global num_batches, args, ds_sizes 
    # num_batches = how many batches in each dataset(train, valid, test)
    # ds_sizes = dataset_sizes 
    for b in range(num_batches[set_name]):
        bi = b * args.batch_size # one batch mul batch_size 
        bj = (b + 1) * args.batch_size 
        if b == num_batches[set_name] - 1:
            bj = ds_sizes[set_name] # maybe only remainer set
        yield bi, bj

err1 = float('inf')
err2 = float('inf')

train_sample = min(args.batch_size, len(features['training']))
val_sample = min(args.batch_size, len(features['validation']))
#print(train_sample) >>>20
#print(val_sample) >>>20





print('\n---- TRAINING ----\n')
with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())

    summary_imp = tf.Summary()
    feature_imp = session.run(feature_importance)
    image = net.feature_importance_plot(feature_imp)
    summary_imp.value.add(tag='feature_importance_%s' % 0, image=image)
    train_writer.add_summary(summary_imp, 0)

    stats_net_1 = session.run(
        net_summaries,
        feed_dict={x: get_batch_1('training1', range(train_sample)),
                   t: splitted_toxicity['training1'][:train_sample],
                   keep_prob: 1.0}
    )

    stats_net_2 = session.run(
        net_summaries,
        feed_dict={x: get_batch_2('training2', range(train_sample)),
                   t: splitted_toxicity['training2'][:train_sample],
                   keep_prob: 1.0}
    )

    train_writer.add_summary(stats_net_1, 0)
    train_writer.add_summary(stats_net_2, 0)

    # in a epoch 
    # run(  train > [training_summaries, net_summaries] > training_summaries > [y,mse] > mse > feature_importance )
    for epoch in range(args.num_epochs): 
        for rotation in args.rotations:
            print('rotation', rotation)
            # TRAIN #

            # after I divide it , and then shuffling 
            x_t1, y_t1 = shuffle(range(ds_sizes['training1']), splitted_toxicity['training1'])
            x_t2, y_t2 = shuffle(range(ds_sizes['training2']), splitted_toxicity['training2'])

            for bi, bj in batches('training1'):
                session.run(train1, feed_dict={x: get_batch_1('training1',
                                                           x_t1[bi:bj],
                                                           rotation),
                                              t: y_t1[bi:bj], keep_prob: args.kp}) # shuffling 한 것의 인덱스를 순서대로 받아오는 것 

            for bi, bj in batches('training2'):
                session.run(train2, feed_dict={x: get_batch_2('training2',
                                                           x_t2[bi:bj],
                                                           rotation),
                                              t: y_t2[bi:bj], keep_prob: args.kp})                                 
            """
            for bi, bj in batches('training'):
            print(bi,bj)

            >>> 
            0 20 
            20 40 
            40 60 
            ... 
            1540 1555 
            """



            # SAVE STATS - per rotation #
            stats_t_1, stats_net_1 = session.run(
                [training_summaries, net_summaries],
                feed_dict={x: get_batch_1('training1', x_t1[:train_sample]),
                           t: y_t1[:train_sample],
                           keep_prob: 1.0}
            )

            stats_t_2, stats_net_2 = session.run(
                [training_summaries, net_summaries],
                feed_dict={x: get_batch_2('training2', x_t2[:train_sample]),
                           t: y_t2[:train_sample],
                           keep_prob: 1.0}
            )

            train_writer.add_summary(stats_t_1, global_step.eval())
            train_writer.add_summary(stats_net_1, global_step.eval())
            train_writer.add_summary(stats_t_2, global_step.eval())
            train_writer.add_summary(stats_net_2, global_step.eval())

            stats_v_1 = session.run(
                training_summaries,
                feed_dict={x: get_batch_1('validation1', range(val_sample)),
                           t: splitted_toxicity['validation1'][:val_sample],
                           keep_prob: 1.0}
            )

            stats_v_2= session.run(
                training_summaries,
                feed_dict={x: get_batch_2('validation2', range(val_sample)),
                           t: splitted_toxicity['validation2'][:val_sample],
                           keep_prob: 1.0}
            )

            val_writer.add_summary(stats_v_1, global_step.eval())
            val_writer.add_summary(stats_v_2, global_step.eval())

        # SAVE STATS - per epoch #
        # training set error
        pred_t1 = np.zeros((ds_sizes['training1'], 1))
        mse_t1 = np.zeros(num_batches['training1'])

        pred_t2 = np.zeros((ds_sizes['training2'], 1))
        mse_t2 = np.zeros(num_batches['training2'])


        for b, (bi, bj) in enumerate(batches('training1')):
            weight = (bj - bi) / ds_sizes['training1']

            pred_t1[bi:bj], mse_t1[b] = session.run(
                [y1, mse1],
                feed_dict={x: get_batch_1('training1', x_t1[bi:bj]),
                           t: y_t1[bi:bj],
                           keep_prob: 1.0}
            )

            mse_t1[b] *= weight

        for b, (bi, bj) in enumerate(batches('training2')):
            weight = (bj - bi) / ds_sizes['training2']

            pred_t2[bi:bj], mse_t2[b] = session.run(
                [y2, mse2],
                feed_dict={x: get_batch_2('training2', x_t2[bi:bj]),
                           t: y_t2[bi:bj],
                           keep_prob: 1.0}
            )

            mse_t2[b] *= weight

        mse_t1 = mse_t1.sum()
        mse_t2 = mse_t2.sum()

        summary_mse = tf.Summary()
        summary_mse.value.add(tag='mse_1', simple_value=mse_t1)
        summary_mse.value.add(tag='mse_2', simple_value=mse_t2)
        train_writer.add_summary(summary_mse, global_step.eval())

        # predictions distribution
        summary_pred = tf.Summary()
        summary_pred.value.add(tag='predictions_all',
                               histo=net.custom_summary_histogram(pred_t))
        train_writer.add_summary(summary_pred, global_step.eval())

        # validation set error
        mse_v1 = 0
        for bi, bj in batches('validation1'):
            weight = (bj - bi) / ds_sizes['validation1']
            mse_v1 += weight * session.run(
                mse1,
                feed_dict={x: get_batch_1('validation1', range(bi, bj)),
                           t: splitted_toxicity['validation1'][bi:bj],
                           keep_prob: 1.0}
            )

        mse_v2 = 0
        for bi, bj in batches('validation2'):
            weight = (bj - bi) / ds_sizes['validation2']
            mse_v2 += weight * session.run(
                mse2,
                feed_dict={x: get_batch_2('validation2', range(bi, bj)),
                           t: splitted_toxicity['validation2'][bi:bj],
                           keep_prob: 1.0}
            )

        summary_mse = tf.Summary()
        summary_mse.value.add(tag='mse_1', simple_value=mse_v1)
        summary_mse.value.add(tag='mse_2', simple_value=mse_v2)
        val_writer.add_summary(summary_mse, global_step.eval())

        # SAVE MODEL #
        print('epoch: %s train1 error: %s, validation1 error: %s, train2 error: %s, validation2 error: %s'
              % (epoch, mse_t1, mse_v1, mse_t2, mse_v2))

        if mse_v1 <= err1:
            err1 = mse_v1
            checkpoint = saver.save(session, prefix, global_step=global_step)

            # feature importance
            summary_imp = tf.Summary()
            feature_imp = session.run(feature_importance)
            image = net.feature_importance_plot(feature_imp)
            summary_imp.value.add(tag='feature_importance', image=image)
            train_writer.add_summary(summary_imp, global_step.eval())
        
        if mse_v2 <= err2:
            err2 = mse_v2
            checkpoint = saver.save(session, prefix, global_step=global_step)

            # feature importance
            summary_imp = tf.Summary()
            feature_imp = session.run(feature_importance)
            image = net.feature_importance_plot(feature_imp)
            summary_imp.value.add(tag='feature_importance', image=image)
            train_writer.add_summary(summary_imp, global_step.eval())


# FINAL PREDICTIONS


predictions = []
rmse = {}

with tf.Session(graph=graph) as session:
    tf.set_random_seed(123)

    saver.restore(session, os.path.abspath(checkpoint))
    saver.save(session, prefix + '-best')

    for dataset in splitted_datasets: 
        pred1 = np.zeros((ds_sizes[dataset+str(1)], 1))
        pred2 = np.zeros((ds_sizes[dataset+str(2)], 1))
        mse_dataset = 0.0

        for bi, bj in batches(dataset+str(1)):
            weight = (bj - bi) / ds_sizes[dataset+str(1)]
            pred[bi:bj], mse_batch = session.run(
                [y1, mse1],
                feed_dict={x: get_batch_1(dataset+str(1), range(bi, bj)),
                           t: splitted_toxicity[dataset+str(1)][bi:bj],
                           keep_prob: 1.0}
            )
            mse_dataset += weight * mse_batch

        predictions.append(pd.DataFrame(data={'pdbid': splitted_ids[dataset+str(1)],
                                              'real': splitted_toxicity[dataset+str(1)][:, 0],
                                              'predicted': pred1[:, 0],
                                              'set': dataset+str(1)}))
        rmse[dataset+str(1)] = sqrt(mse_dataset)

        for bi, bj in batches(dataset+str(2)):
            weight = (bj - bi) / ds_sizes[dataset+str(2)]
            pred[bi:bj], mse_batch = session.run(
                [y2, mse2],
                feed_dict={x: get_batch_2(dataset+str(2), range(bi, bj)),
                           t: splitted_toxicity[dataset+str(2)][bi:bj],
                           keep_prob: 1.0}
            )
            mse_dataset += weight * mse_batch

        predictions.append(pd.DataFrame(data={'pdbid': splitted_ids[dataset+str(2)],
                                              'real': splitted_toxicity[dataset+str(2)][:, 0],
                                              'predicted': pred2[:, 0],
                                              'set': dataset+str(2)}))
        rmse[dataset+str(2)] = sqrt(mse_dataset)


predictions = pd.concat(predictions, ignore_index=True)
predictions.to_csv(prefix + '-predictions.csv', index=False)

for set_name, tab in predictions.groupby('set'):
    grid = sns.jointplot('real', 'predicted', data=tab, color=color[set_name],
                         space=0.0, xlim=(0, 16), ylim=(0, 16),
                         annot_kws={'title': '%s set (rmse=%.3f)'
                                             % (set_name, rmse[set_name])})

    image = net.custom_summary_image(grid.fig)
    grid.fig.savefig(prefix + '-%s.pdf' % set_name)
    summary_pred = tf.Summary()
    summary_pred.value.add(tag='predictions_%s' % (set_name),
                           image=image)
    train_writer.add_summary(summary_pred)


train_writer.close()
val_writer.close()
