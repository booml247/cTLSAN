import os
import time
import pickle
import random

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from input import DataInput, DataInputTest
from model import Model
import boto3
import io
import sys

max_epochs = 20
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

tf.app.flags.DEFINE_string('model_dir', 'save_path', 'Path to save model checkpoints')
tf.app.flags.DEFINE_integer('display_freq', 100, 'Display training status every this iteration')
tf.app.flags.DEFINE_string('dataset', 'CDs_and_Vinyl', 'dataset name')
FLAGS = tf.app.flags.FLAGS

train_batch_size = 32
test_batch_size = 128

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
model_dir = FLAGS.model_dir
from_scratch = True

time_line = []
auc_value = []

# Loading data
print('Loading data..', flush=True)
s3 = boto3.client('s3')
bucket = 'your-s3-bucket'
s3_key = f'Bi_LSTM_input_data/{FLAGS.dataset}_dataset.pkl'
buffer = io.BytesIO()

# Download the pickle file into memory
s3.download_fileobj(bucket, s3_key, buffer)
buffer.seek(0)
train_set = pickle.load(buffer)
test_set = pickle.load(buffer)
cate_list = pickle.load(buffer)
user_count, item_count, cate_count = pickle.load(buffer)

best_auc = 0.0
best_prec = [0, 0, 0, 0, 0, 0]
best_recall = [0, 0, 0, 0, 0, 0]

def create_model(sess, user_count, item_count, cate_count, cate_list):

  model = Model(user_count, item_count, cate_count, cate_list, model_dir)

  print('All global variables:')
  for v in tf.global_variables():
    if v not in tf.trainable_variables():
      print('\t', v)
    else:
      print('\t', v, 'trainable')

  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print('Reloading model parameters..', flush=True)
    model.restore(sess, ckpt.model_checkpoint_path)
    metric_ops = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metric")
    sess.run(tf.initialize_variables(metric_ops))
  else:
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    print('Created new model parameters..', flush=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

  return model

def eval_auc(sess, model):
  auc_sum = 0.0
  for _, uij in DataInputTest(test_set, test_batch_size):
    auc_sum += model.eval_auc(sess, uij) * len(uij[0])
  test_auc = auc_sum / len(test_set)
  model.eval_writer.add_summary(
      summary=tf.Summary(
          value=[tf.Summary.Value(tag='Eval AUC', simple_value=test_auc)]),
      global_step=model.global_step.eval())
  global best_auc
  if best_auc < test_auc:
    best_auc = test_auc
    model.save(sess, model_dir + '/ckpt')
  return test_auc

def eval_prec(sess, model):
  for _, batch in DataInputTest(test_set, test_batch_size):
    model.eval_prec(sess, batch)
  prec = sess.run([model.prec_1, model.prec_10, model.prec_20, model.prec_30, model.prec_40, model.prec_50])
  return prec

def eval_recall(sess, model):
  for _, batch in DataInputTest(test_set, test_batch_size):
    model.eval_recall(sess, batch)
  recall = sess.run([model.recall_1, model.recall_10, model.recall_20, model.recall_30, model.recall_40, model.recall_50])
  return recall


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

  if from_scratch:
    if tf.gfile.Exists(model_dir):
      tf.gfile.DeleteRecursively(model_dir)
    tf.gfile.MakeDirs(model_dir)

  model = create_model(sess, user_count, item_count, cate_count, cate_list)

  # Eval init AUC
  print('Init AUC: %.4f' % eval_auc(sess, model))
  # Eval init precision
  print('Init precision:')
  prec = eval_prec(sess, model)
  for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
    print('@' + str(k) + ' = %.4f' % prec[i], end=' ')
  print()
  # Eval init recall
  print('Init recall:')
  recall = eval_recall(sess, model)
  for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
    print('@' + str(k) + ' = %.4f' % recall[i], end=' ')
  print()

  lr = 1.0
  start_time = time.time()
  for _ in range(max_epochs):

    random.shuffle(train_set)

    epoch_size = round(len(train_set) / train_batch_size)
    loss_sum = 0.0
    for _, uij in DataInput(train_set, train_batch_size):
      add_summary = bool(model.global_step.eval() % FLAGS.display_freq == 0)
      loss = model.train(sess, uij, lr, add_summary)
      loss_sum += loss

      if model.global_step.eval() % 1000 == 0:
        test_auc = eval_auc(sess, model)        
        if time.time()-start_time > 90000:
          with open('training_time.pkl', 'wb') as f:
            pickle.dump((time_line, auc_value), f, pickle.HIGHEST_PROTOCOL)
        time_line.append(time.time()-start_time)
        auc_value.append(test_auc)

        print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f' %
              (model.global_epoch_step.eval(), model.global_step.eval(),
               loss_sum / 1000, test_auc),
              flush=True)
        print('Precision:')
        prec = eval_prec(sess, model)
        for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
          print('@' + str(k) + ' = %.4f' % prec[i], end=' ')
        print()
        print('Recall:')
        recall = eval_recall(sess, model)
        for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
          print('@' + str(k) + ' = %.4f' % recall[i], end=' ')
        print()

        loss_sum = 0.0

        for i in range(6):
          if prec[i] > best_prec[i]:
            best_prec[i] = prec[i]
          if recall[i] > best_recall[i]:
            best_recall[i] = recall[i]

      if model.global_step.eval() == 270000:
        lr = 0.1

    print('Epoch %d DONE\tCost time: %.2f' %
          (model.global_epoch_step.eval(), time.time()-start_time),
          flush=True)
    model.global_epoch_step_op.eval()
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

with open('training_time.pkl', 'wb') as f:
  pickle.dump((time_line, auc_value), f, pickle.HIGHEST_PROTOCOL)