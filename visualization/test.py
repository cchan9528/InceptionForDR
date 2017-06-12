import tensorflow as tf
from keras.models import load_model
import os, numpy
import keras.backend as K

# Credit to Shoyer
# https://github.com/tensorflow/tensorflow/issues/675
def jacobian(y, x):
  y_flat = tf.reshape(y, (-1,))
  jacobian_flat = tf.stack(
      [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y_flat)])
  return tf.reshape(jacobian_flat, y.shape.concatenate(x.shape))

if __name__=="__main__":
    model = load_model('test_model.h5')
    X_in = numpy.random.random((1,784))
    X = tf.placeholder(tf.float32, shape=(1,784))
    Y = model(X)
    sess = K.get_session()
    sess.run(Y, feed_dict={X: X_in})
    j = jacobian(Y,X)
    salience = sess.run(j, feed_dict={X:X_in})

    print(salience)
