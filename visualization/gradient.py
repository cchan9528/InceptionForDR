################################################################
#                                                              #
#                         doc                                  #
#                                                              #
################################################################

# overview    : find the jacobian of class scores w.r.t input
#               and overlay the result onto the original image
#               to produce a salience map

################################################################
#                                                              #
#                         config                               #
#                                                              #
################################################################

import sys
from keras import models
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as matplot
from PIL import Image, ImageFont, ImageDraw
import numpy as np

input_images = [
    "/data/DR/inceptionDR/modelPerfRecords/inceptionMini3-classAll.v8a/epoch8/imageLinks/0/0/3795_right.jpg",
    "/data/DR/inceptionDR/modelPerfRecords/inceptionMini3-classAll.v8a/epoch8/imageLinks/1/1/2495_right.jpg",
    "/data/DR/inceptionDR/modelPerfRecords/inceptionMini3-classAll.v8a/epoch8/imageLinks/2/2/42247_left.jpg",
    "/data/DR/inceptionDR/modelPerfRecords/inceptionMini3-classAll.v8a/epoch8/imageLinks/3/3/3868_left.jpg",
    "/data/DR/inceptionDR/modelPerfRecords/inceptionMini3-classAll.v8a/epoch8/imageLinks/4/4/3563_left.jpg",

    '/home/joseph/Desktop/td/0/10407_right.jpg',
    '/home/joseph/Desktop/td/4/13811_right.jpg'

]

channels= 3
rows = 800
cols = 800

blendalpha = .2

imaI = 3
################################################################
#                                                              #
#                         helper                               #
#                                                              #
################################################################
def getImgArray(path):
    src = Image.open(path).resize([rows, cols]);
    x = np.expand_dims(src, axis=0)
    return x

# Credit to Shoyer
# https://github.com/tensorflow/tensorflow/issues/675
def jacobian(y, x):
  y_flat = tf.reshape(y, (-1,))
  jacobian_flat = tf.stack(
      [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y_flat)])
  return tf.reshape(jacobian_flat, y.shape.concatenate(x.shape))

################################################################
#                                                               #
#                         runner                               #
#                                                              #
################################################################

if __name__ == "__main__":

    ''' Load a model, find it's gradient matrix at input, visualize salience '''


    # Load the model
    model = models.load_model('/data/DR/inceptionDR/matureModels/modelIncepMini3-0.74.h5')

    # Define the input (X), outputs(Y), gradient matrix (jacobian J)
    X_in = getImgArray(input_images[imaI])
    X = tf.placeholder(tf.float32, shape=(1,rows,cols,channels))
    Y = model(X)
    J = jacobian(Y,X)

    # Evaluate the Jacobian (rows are class gradients)
    sess = K.get_session()
    class_gradients = sess.run(J, feed_dict={X:X_in, K.learning_phase(): 1})
    class_gradients = class_gradients.reshape((5, rows, cols, channels))

    # Rearrange the gradient matrix at the input to match the input image shape
    class_gradient = []
    for row in class_gradients:
        class_gradient.append(row.reshape((rows,cols,channels)))

    # For each pixel, get max across all color chanmnels as the salience value
    oriImage = Image.open(input_images[imaI]).resize([rows, cols])
    oriImage.putalpha(100);
    for i, g in enumerate(class_gradient):
        salience = (np.amax(np.absolute(g), axis=2))
        salience /= 5
        colormap = matplot.get_cmap('jet')
        cp = colormap(salience)
        coloredSalience = Image.fromarray(np.uint8(cp*255))
        salienceMap = Image.blend(oriImage, coloredSalience, blendalpha)
        salienceMap.save('visImages/salGrad'+str(i)+'.jpg')
        salienceMap.show(title='salGrad'+str(i))
