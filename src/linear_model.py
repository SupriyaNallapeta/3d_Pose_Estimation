from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import variable_scope as vs

import os
import numpy as np
from six.moves import xrange 
import tensorflow as tf
import data_utils

def kaiming(shape, dtype, partition_info=None):
  return(tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))

class LinearModel(object):
  def __init__(self,
               linear_size,
               num_layers,
               residual,
               batch_norm,
               max_norm,
               batch_size,
               learning_rate,
               summaries_dir,
               predict_14=False,
               dtype=tf.float32):
    self.HUMAN_2D_SIZE = 16 * 2
    self.HUMAN_3D_SIZE = 14 * 3 if predict_14 else 16 * 3

    self.input_size  = self.HUMAN_2D_SIZE
    self.output_size = self.HUMAN_3D_SIZE

    self.isTraining = tf.placeholder(tf.bool,name="isTrainingflag")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    self.train_writer = tf.summary.FileWriter( os.path.join(summaries_dir, 'train' ))
    self.test_writer  = tf.summary.FileWriter( os.path.join(summaries_dir, 'test' ))

    self.linear_size   = linear_size
    self.batch_size    = batch_size
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype, name="learning_rate")
    self.global_step   = tf.Variable(0, trainable=False, name="global_step")
    decay_steps = 100000  
    decay_rate = 0.96   
    self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate)

    with vs.variable_scope("inputs"):

      enc_in  = tf.placeholder(dtype, shape=[None, self.input_size], name="enc_in")
      dec_out = tf.placeholder(dtype, shape=[None, self.output_size], name="dec_out")

      self.encoder_inputs  = enc_in
      self.decoder_outputs = dec_out

    with vs.variable_scope( "linear_model" ):

      w1 = tf.get_variable( name="w1", initializer=kaiming, shape=[self.HUMAN_2D_SIZE, linear_size], dtype=dtype )
      b1 = tf.get_variable( name="b1", initializer=kaiming, shape=[linear_size], dtype=dtype )
      w1 = tf.clip_by_norm(w1,1) if max_norm else w1
      y3 = tf.matmul( enc_in, w1 ) + b1

      if batch_norm:
        y3 = tf.layers.batch_normalization(y3,training=self.isTraining, name="batch_normalization")
      y3 = tf.nn.relu( y3 )
      y3 = tf.nn.dropout( y3, self.dropout_keep_prob )

      for idx in range( num_layers ):
        y3 = self.two_linear( y3, linear_size, residual, self.dropout_keep_prob, max_norm, batch_norm, dtype, idx )

      w4 = tf.get_variable( name="w4", initializer=kaiming, shape=[linear_size, self.HUMAN_3D_SIZE], dtype=dtype )
      b4 = tf.get_variable( name="b4", initializer=kaiming, shape=[self.HUMAN_3D_SIZE], dtype=dtype )
      w4 = tf.clip_by_norm(w4,1) if max_norm else w4
      y = tf.matmul(y3, w4) + b4
    self.outputs = y
    self.loss = tf.reduce_mean(tf.square(y - dec_out))
    self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

    self.err_mm = tf.placeholder( tf.float32, name="error_mm" )
    self.err_mm_summary = tf.summary.scalar( "loss/error_mm", self.err_mm )

    opt = tf.train.AdamOptimizer( self.learning_rate )
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):

      gradients = opt.compute_gradients(self.loss)
      self.gradients = [[] if i==None else i for i in gradients]
      self.updates = opt.apply_gradients(gradients, global_step=self.global_step)

    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    self.saver = tf.train.Saver( tf.global_variables(), max_to_keep=10 )


  def two_linear( self, xin, linear_size, residual, dropout_keep_prob, max_norm, batch_norm, dtype, idx ):

    with vs.variable_scope( "two_linear_"+str(idx) ) as scope:

      input_size = int(xin.get_shape()[1])

      w2 = tf.get_variable( name="w2_"+str(idx), initializer=kaiming, shape=[input_size, linear_size], dtype=dtype)
      b2 = tf.get_variable( name="b2_"+str(idx), initializer=kaiming, shape=[linear_size], dtype=dtype)
      w2 = tf.clip_by_norm(w2,1) if max_norm else w2
      y = tf.matmul(xin, w2) + b2
      if  batch_norm:
        y = tf.layers.batch_normalization(y,training=self.isTraining,name="batch_normalization1"+str(idx))

      y = tf.nn.relu( y )
      y = tf.nn.dropout( y, dropout_keep_prob )

      w3 = tf.get_variable( name="w3_"+str(idx), initializer=kaiming, shape=[linear_size, linear_size], dtype=dtype)
      b3 = tf.get_variable( name="b3_"+str(idx), initializer=kaiming, shape=[linear_size], dtype=dtype)
      w3 = tf.clip_by_norm(w3,1) if max_norm else w3
      y = tf.matmul(y, w3) + b3

      if  batch_norm:
        y = tf.layers.batch_normalization(y,training=self.isTraining,name="batch_normalization2"+str(idx))

      y = tf.nn.relu( y )
      y = tf.nn.dropout( y, dropout_keep_prob )

      y = (xin + y) if residual else y

    return y

  def step(self, session, encoder_inputs, decoder_outputs, dropout_keep_prob, isTraining=True):
    input_feed = {self.encoder_inputs: encoder_inputs,
                  self.decoder_outputs: decoder_outputs,
                  self.isTraining: isTraining,
                  self.dropout_keep_prob: dropout_keep_prob}

    if isTraining:
      output_feed = [self.updates, 
                     self.loss,
                     self.loss_summary,
                     self.learning_rate_summary,
                     self.outputs]

      outputs = session.run( output_feed, input_feed )
      return outputs[1], outputs[2], outputs[3], outputs[4]

    else:
      output_feed = [self.loss,
                     self.loss_summary,
                     self.outputs]

      outputs = session.run(output_feed, input_feed)
      return outputs[0], outputs[1], outputs[2]

  def get_all_batches( self, data_x, data_y, camera_frame, training=True ):
    n = 0
    for key2d in data_x.keys():
      n2d, _ = data_x[ key2d ].shape
      n = n + n2d

    encoder_inputs  = np.zeros((n, self.input_size), dtype=float)
    decoder_outputs = np.zeros((n, self.output_size), dtype=float)

    idx = 0
    for key2d in data_x.keys():
      (subj, b, fname) = key2d
      key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
      key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and camera_frame else key3d

      n2d, _ = data_x[ key2d ].shape
      encoder_inputs[idx:idx+n2d, :]  = data_x[ key2d ]
      decoder_outputs[idx:idx+n2d, :] = data_y[ key3d ]
      idx = idx + n2d


    if training:
      idx = np.random.permutation( n )
      encoder_inputs  = encoder_inputs[idx, :]
      decoder_outputs = decoder_outputs[idx, :]

    n_extra  = n % self.batch_size
    if n_extra > 0:
      encoder_inputs  = encoder_inputs[:-n_extra, :]
      decoder_outputs = decoder_outputs[:-n_extra, :]

    n_batches = n // self.batch_size
    encoder_inputs  = np.split( encoder_inputs, n_batches )
    decoder_outputs = np.split( decoder_outputs, n_batches )

    return encoder_inputs, decoder_outputs
