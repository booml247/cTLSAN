import os
import time
import json
import pickle
import random
import logging
from collections import OrderedDict
import boto3
import io
import sys

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from input import DataInput, DataInputTest
from model import Model

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

time_line = []
auc_value = []

# pylint: disable=line-too-long
# Network parameters
tf.app.flags.DEFINE_integer('hidden_units', 64, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_blocks', 1, 'Number of blocks in each attention')
tf.app.flags.DEFINE_integer('num_heads', 8, 'Number of heads in each attention')
tf.app.flags.DEFINE_integer('Ls', 10, 'Length of long-term sessions')
tf.app.flags.DEFINE_float('dropout', 0.0, 'Dropout probability(0.0: no dropout)')
tf.app.flags.DEFINE_float('regulation_rate', 0.00005, 'L2 regulation rate')

tf.app.flags.DEFINE_integer('itemid_embedding_size', 32, 'Item id feature size')
tf.app.flags.DEFINE_integer('userid_embedding_size', 32, 'User id feature size')
tf.app.flags.DEFINE_integer('cateid_embedding_size', 32, 'Category id feature size')

# Training parameters
tf.app.flags.DEFINE_boolean('from_scratch', True, 'Romove model_dir, and train from scratch, default: False')
tf.app.flags.DEFINE_string('model_dir', 'save_path', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer for training: (adadelta, adam, rmsprop, sgd*)')
tf.app.flags.DEFINE_float('learning_rate', 1.0, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm')

tf.app.flags.DEFINE_integer('train_batch_size', 32, 'Training Batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 128, 'Testing Batch size')
tf.app.flags.DEFINE_integer('max_epochs', 20, 'Maximum number of training epochs')

tf.app.flags.DEFINE_integer('display_freq', 100, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('eval_freq', 1000, 'Display training status every this iteration')


# Runtime parameters
tf.app.flags.DEFINE_string('cuda_visible_devices', '3', 'Choice which GPU to use')
tf.app.flags.DEFINE_float('per_process_gpu_memory_fraction', 0.0, 'Gpu memory use fraction, 0.0 for allow_growth=True')
tf.app.flags.DEFINE_string('dataset', 'CDs_and_Vinyl', 'dataset name')
# pylint: enable=line-too-long

FLAGS = tf.app.flags.FLAGS

def create_model(sess, config, item_cate_list):

  print(json.dumps(config, indent=4), flush=True)
  model = Model(config, item_cate_list)

  print('All global variables:')
  for v in tf.global_variables():
    if v not in tf.trainable_variables():
      print('\t', v)
    else:
      print('\t', v, 'trainable')

  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print('Reloading model parameters..', flush=True)
    model.restore(sess, ckpt.model_checkpoint_path)
    metric_ops = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metric")
    sess.run(tf.initialize_variables(metric_ops))
  else:
    if not os.path.exists(FLAGS.model_dir):
      os.makedirs(FLAGS.model_dir)
    print('Created new model parameters..', flush=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

  return model

def eval_auc(sess, test_set, model, config):
  auc_sum = 0.0
  for _, batch in DataInputTest(test_set, FLAGS.test_batch_size, config['Ls']):
    auc_sum += model.eval_auc(sess, batch) * len(batch[0])
  res = auc_sum / len(test_set)
  model.eval_writer.add_summary(
    summary=tf.Summary(
        value=[tf.Summary.Value(tag='AUC', simple_value=res)]),
    global_step=model.global_step.eval())

  return res

def eval_prec(sess, test_set, model, config):
  for _, batch in DataInputTest(test_set, FLAGS.test_batch_size, config['Ls']):
    model.eval_prec(sess, batch)
  prec = sess.run([model.prec_1, model.prec_10, model.prec_20, model.prec_30, model.prec_40, model.prec_50])
  for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
    model.eval_writer.add_summary(
      summary=tf.Summary(
          value=[tf.Summary.Value(tag='P@'+str(k), simple_value=prec[i])]),
      global_step=model.global_step.eval())
  return prec

def eval_recall(sess, test_set, model, config):
  for _, batch in DataInputTest(test_set, FLAGS.test_batch_size, config['Ls']):
    model.eval_recall(sess, batch)
  recall = sess.run([model.recall_1, model.recall_10, model.recall_20, model.recall_30, model.recall_40, model.recall_50])
  for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
    model.eval_writer.add_summary(
      summary=tf.Summary(
          value=[tf.Summary.Value(tag='R@'+str(k), simple_value=recall[i])]),
      global_step=model.global_step.eval())
  return recall


def train():
  start_time = time.time()

  if FLAGS.from_scratch:
    if tf.gfile.Exists(FLAGS.model_dir):
      tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

  # Loading data
  print('Loading data..', flush=True)
  s3 = boto3.client('s3')
  bucket = 'your-s3-bucket'
  s3_key = f'TLSAN_input_data/{FLAGS.dataset}_dataset.pkl'
  buffer = io.BytesIO()

  # Download the pickle file into memory
  s3.download_fileobj(bucket, s3_key, buffer)
  buffer.seek(0)
  train_set = pickle.load(buffer)
  test_set = pickle.load(buffer)
  user_count, item_count, cate_count = pickle.load(buffer)
  item_cate_list = pickle.load(buffer)

  # Config GPU options
  if FLAGS.per_process_gpu_memory_fraction == 0.0:
    gpu_options = tf.GPUOptions(allow_growth=True)
  elif FLAGS.per_process_gpu_memory_fraction == 1.0:
    gpu_options = tf.GPUOptions()
  else:
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices

  # Build Config
  config = OrderedDict(sorted(FLAGS.__flags.items()))
  for k, v in config.items():
    config[k] = v.value
  config['user_count'] = user_count
  config['item_count'] = item_count
  config['cate_count'] = cate_count


  # Initiate TF session
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Create a new model or reload existing checkpoint
    model = create_model(sess, config, item_cate_list)
    print('Init finish.\tCost time: %.2fs' % (time.time()-start_time),
          flush=True)

    # Eval init AUC
    print('Init AUC: %.4f' % eval_auc(sess, test_set, model, config))
    # Eval init precision
    print('Init precision:')
    prec = eval_prec(sess, test_set, model, config)
    for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
      print('@' + str(k) + ' = %.4f' % prec[i], end=' ')
    print()
    # Eval init recall
    print('Init recall:')
    recall = eval_recall(sess, test_set, model, config)
    for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
      print('@' + str(k) + ' = %.4f' % recall[i], end=' ')
    print()

    # Start training
    lr = FLAGS.learning_rate
    epoch_size = round(len(train_set) / FLAGS.train_batch_size)
    print('Training..\tmax_epochs: %d\tepoch_size: %d' %
          (FLAGS.max_epochs, epoch_size), flush=True)

    start_time, avg_loss, best_auc = time.time(), 0.0, 0.0
    best_prec = [0, 0, 0, 0, 0, 0]
    best_recall = [0, 0, 0, 0, 0, 0]

    for _ in range(FLAGS.max_epochs):
      random.shuffle(train_set)
      for _, batch in DataInput(train_set, FLAGS.train_batch_size, config['Ls']):

        add_summary = bool(model.global_step.eval() % FLAGS.display_freq == 0)
        step_loss = model.train(sess, batch, lr, add_summary)
        avg_loss += step_loss

        if model.global_step.eval() % FLAGS.eval_freq == 0:
          test_auc = eval_auc(sess, test_set, model, config)

          # Training curve
          time_line.append(time.time()-start_time)
          auc_value.append(test_auc)

          print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_auc: %.4f\t' %
                (model.global_epoch_step.eval(), model.global_step.eval(),
                 avg_loss / FLAGS.eval_freq, test_auc),
                flush=True)
          print('Precision:')
          prec = eval_prec(sess, test_set, model, config)
          for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
            print('@' + str(k) + ' = %.4f' % prec[i], end=' ')
          print()
          print('Recall:')
          recall = eval_recall(sess, test_set, model, config)
          for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
            print('@' + str(k) + ' = %.4f' % recall[i], end=' ')
          print()

          avg_loss = 0.0

          if model.global_step.eval() > 20000:
            for i in range(6):
              if prec[i] > best_prec[i]:
                best_prec[i] = prec[i]
              if recall[i] > best_recall[i]:
                best_recall[i] = recall[i]
          if test_auc > 0.8 and test_auc > best_auc:
            best_auc = test_auc
            model.save(sess)

        if model.global_step.eval() == 150000:
          lr = 0.1

      print('Epoch %d DONE\tCost time: %.2f' %
            (model.global_epoch_step.eval(), time.time()-start_time),
            flush=True)
      model.global_epoch_step_op.eval()
    model.save(sess)
    print('Best test_auc:', best_auc)
    print('Best precision:')
    for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
      print('@' + str(k) + ' = %.4f' % best_prec[i], end=' ')
    print()
    print('Best recall:')
    for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
      print('@' + str(k) + ' = %.4f' % best_recall[i], end=' ')
    print()
    print('Finished', flush=True)

    output_path = os.path.join(FLAGS.model_dir, 'result.txt')
    with open(output_path, 'w') as f:
      original_stdout = sys.stdout
      sys.stdout = f  # Redirect stdout to the file

      print('Best test_auc:', best_auc)
      print('Best precision:')
      for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
          print('@' + str(k) + ' = %.4f' % best_prec[i], end=' ')
      print()
      print('Best recall:')
      for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
          print('@' + str(k) + ' = %.4f' % best_recall[i], end=' ')
      print()
      print('Finished', flush=True)

      sys.stdout = original_stdout  # Restore stdout


def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
  with open('training_time.pkl') as f:
    pickle.dump((time_line, auc_value), f, pickle.HIGHEST_PROTOCOL)