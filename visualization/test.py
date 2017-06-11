from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import os, numpy

# Credit to Shoyer
# https://github.com/tensorflow/tensorflow/issues/675
def jacobian(y, x):
  y_flat = tf.reshape(y, (-1,))
  jacobian_flat = tf.stack(
      [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y_flat)])
  return tf.reshape(jacobian_flat, y.shape.concatenate(x.shape))

# >>> model = load_model('./test_model.h5')
# >>> X=tf.placeholder(tf.float32,shape=(1,784))
# >>> Y=model(X)
# >>> X_in = numpy.random.random((1,784))
# >>> sess = K.get_session()
# >>> sess.run(Y, feed_dict={X:X_in})
# array([[ 0.38661397,  0.09364946,  0.46532524,  0.05357145,  0.00083982]], dtype=float32)
# >>> j = test.jacobian(Y,X_in)
# >>> sess.run(j, feed_dict={X:X_in})

# >>> X = tf.placeholder(tf.float32, shape=(1,4,4,512))
# >>> Y = model(X)
# >>> Y = tf.reshape(Y, shape = (1,5))
# >>> sess.run(Y, feed_dict={X:X_in, K.learning_phase(): 1})
# >>> import test
# >>> j = test.jacobian(Y,X)
# >>> salience = sess.run(j, feed_dict={X:X_in, K.learning_phase():1})
# >>> salience.shape
# (1, 5, 1, 4, 4, 512)

if __name__=="__main__":
    model = Sequential([
        Dense(32, input_shape=(784,)),
        Activation('relu'),
        Dense(5),
        Activation('softmax'),
    ])

    model.compile( optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    train = numpy.random.random((1000,784))
    temp = numpy.array([0]*200+[1]*200+[2]*200+[3]*200+[4]*200)
    labels = numpy.zeros( (5*200, 5) )
    labels[ numpy.arange( 5*200 ), temp] = 1

    model.fit(train, labels, epochs = 50, batch_size=10)

    model.save('test_model.h5')
