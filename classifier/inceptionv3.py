################################################################
#                                                              #
#                         doc                                  #
#                                                              #
################################################################

# overview    : further train a pre-trained InceptionV3
#               CNN architecture to classify
#               diabetic retinopathy (DR) fundus images

# method      : (1) using IncpetionV3 CNN, transform DR data
#               into "bottleneck" features; saved to
#               numpy files so they can be loaded later
#               (2) then, use the transformed data as the
#               input data for a fully-connected network that
#               will learn to classify fundus on DR severity

# note        : this is used for testing

# dependencies: keras
#               numpy

# heavily inspired by tutorial from Francois Chollet, Keras
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

################################################################
#                                                              #
#                         config                               #
#                                                              #
################################################################
import os, numpy
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

################################################################
#                                                              #
#                         helpers                              #
#                                                              #
################################################################
def bottleneckTransform(t_dir = '../data/training',
                        v_dir = '../data/validation',
                        batchSize = 10):

    '''Save convolutional layers output of InceptionV3 as input for end layer'''

    inputShape = (150, 150, 3) # keras.backend.image_data_format() for channels

    # Instantiate a VGG16 CNN w/o End Layer
    model = applications.InceptionV3(include_top=False,
                                     weights='imagenet',
                                     input_shape=inputShape)

    # Instantiate an image retriever
    datagetter = ImageDataGenerator(rescale = 1.0 / 255)

    # Transform training data
    numT = len([i for i in os.listdir( t_dir+'/'+os.listdir(t_dir)[0] ) ]) * 5
    trainGetter = datagetter.flow_from_directory( t_dir,
                                                  target_size = inputShape[:2],
                                                  batch_size  = batchSize,
                                                  class_mode  = None,
                                                  shuffle     = False )

    transformed_tData = model.predict_generator(trainGetter, numT//batchSize)

    numpy.save('inceptionv3_fc_training_data.npy', transformed_tData)

    # Transfrom validation data
    numV = len([i for i in os.listdir( v_dir+'/'+os.listdir(v_dir)[0] ) ]) * 5
    validGetter = datagetter.flow_from_directory( v_dir,
                                                  target_size = inputShape[:2],
                                                  batch_size  = batchSize,
                                                  class_mode  = None,
                                                  shuffle     = False )


    transformed_vData = model.predict_generator(validGetter, numV//batchSize)

    numpy.save('inceptionv3_fc_validation_data.npy', transformed_vData)

    print("Transformed data into bottleneck features using InceptionV3.")
    return (numT//5, numV//5)

def trainFCLayer( t_classAmt,
                  v_classAmt,
                  numpy_tData = 'inceptionv3_fc_training_data.npy',
                  numpy_vData = 'inceptionv3_fc_validation_data.npy',
                  numEpochs   = 30,
                  batchSize   = 10 ):

    '''Train a fully-connected (FC) network for new data classes'''

    # Load data
    tData = numpy.load(numpy_tData)
    vData = numpy.load(numpy_vData)

    # Load the 5 class labels as one-hot
    arr_tLabels = numpy.array([0]*t_classAmt+\
                              [1]*t_classAmt+\
                              [2]*t_classAmt+\
                              [3]*t_classAmt+\
                              [4]*t_classAmt)
    arr_vLabels = numpy.array([0]*v_classAmt+\
                              [1]*v_classAmt+\
                              [2]*v_classAmt+\
                              [3]*v_classAmt+\
                              [4]*v_classAmt)
    onehot_tLabels = numpy.zeros( (5 * t_classAmt , 5) )
    onehot_tLabels[numpy.arange( 5 * t_classAmt ), arr_tLabels] = 1
    onehot_vLabels = numpy.zeros( (5 * v_classAmt , 5) )
    onehot_vLabels[numpy.arange( 5 * v_classAmt ), arr_vLabels] = 1

    # Build FC model to place on top of the convolutional layers
    fcModel = Sequential()
    fcModel.add( Flatten( input_shape = tData.shape[1:]) )
    fcModel.add( Dense(256, activation='relu') )
    fcModel.add( Dropout(0.5) )
    fcModel.add( Dense(5, activation='softmax') )
    fcModel.compile( optimizer = 'rmsprop',
                     loss      = 'categorical_crossentropy',
                     metrics   = ['accuracy'])

    # Train FC model
    fcModel.fit( tData,
                 onehot_tLabels,
                 epochs          = numEpochs,
                 batch_size      = batchSize,
                 validation_data = (vData, onehot_vLabels) )

    # Save final weights to classify new data; we now have the new model
    fcModel.save_weights('inceptionv3_final_weights.h5')

    print("New parameters acquired from IncpetionV3 bottlenecks.")
    print("Classification ready.")

if __name__=="__main__":
    t_classAmt, v_classAmt = bottleneckTransform()
    trainFCLayer(t_classAmt, v_classAmt)
