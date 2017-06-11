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

import sys, numpy, keras.models, tensorflow
import matplotlib.pyplot as matplot

input_images = [
    "./../data/validation/0/1024_left.jpeg",
    "./../data/validation/1/1266_left.jpeg",
    "./../data/validation/2/1111_left.jpeg",
    "./../data/validation/3/16624_right.jpeg",
    "./../data/validation/4/18819_left.jpeg"
]

chanels= 3
rows = 150
cols = 150

blendalpha = .8

################################################################
#                                                              #
#                         runner                               #
#                                                              #
################################################################

if __name__ == "__main__":

    ''' Load a model, find it's gradient matrix at input, visualize salience '''

    if len(sys.argv) != 2:
        sys.exit("Error: \n Usage: python gradient.py [modelfile]")

    # Load the model
    model = keras.models.load_model(sys.argv[1])

    # Get gradient matrices

    # option 1:
    X = tensorflow.placeholder(shape=(None,4,4,512))
    Y = model(X)
    jacobian = tensorflow.gradients(model.output, X)
    jacobian /= K.sqrt(K.mean(K.square(jacobian)))*1e-2
    findJacobian = K.function([X, K.learning_phase()], [jacobian])
    findY = K.function([X], print(K.get_session().run(Y)))
    # if model.input is the original image
    j = findJacobian([input_image_as_numpy_array[0:1], 0])
    # if model.input is the flattened/transformed version of image like bottleneck
    # findJacobian([samples[0:1], 0])

    # option 2:
    # weights   = model.trainable_weights
    # gradients = model.optimizer.get_gradients(model.model.total_loss, weights)
    # gradient_at_input = gradients[0]

    # Rearrange the gradient matrix at the input to match the input image shape
        # NOTE: Assuming we have gradient NxM, n=number of pixels, M=number of weights in next layer?
    class_gradient = []
    for jacobianRow in j:
        class_gradient.append(jacobianRow.reshape((channels,rows,cols)))

    # For each pixel, select max across all color chanmnels as the salience value
        # NOTE: Assuming channels first! (numpy.shape returns channels first)
    # salienceMap = numpy.zeros(single_channel_input_shape)
    # if len(input_shape.shape) == 3:
    #     for pixel, garbage in numpy.ndenumerate(pixels[0,:,:]):
    #         row = pixel[0]; col = pixel[1];
    #         salienceMap[row,col] = max(numpy.absolute(pixels[:, row, col]))
    # else:
    #     for row_col, value in numpy.ndenumerate(pixels):
    #         salienceMap[row_col[0], row_col[1]] = abs(value)
    for i, g in enumerate(class_gradient):
        salience = (numpy.amax(numpy.absolute(g), axis=0)).flatten()
        colormap = matplot.get_cmap('jet')
        coloredSalience = Image.fromarray(numpy.uint8(colormap(salience)*255))
        salienceMap = Image.blend(input_images[i], coloredSalience, blendalpha)
        salienceMap.show()
