################################################################
#                                                              #
#                         doc                                  #
#                                                              #
################################################################

# overview    : further train a pre-trained VGG16
#               CNN architecture to classify
#               diabetic retinopathy (DR) fundus images

# method      : (1) using VGG16 CNN, transform DR data
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
def bottleneckTransform(train_dir = '../data/training',
                        valid_dir = '../data/validation',
                        batchSize = 16):

    # Instantiate a VGG16 CNN
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagetter = ImageDataGenerator(rescale = 1.0 / 255)
    # Transform training data
    numTrainImgs = len([i for i in os.listdir(train_dir)])
    trainGetter = datagetter.flow_from_directory(train_dir,
                                   target_size = (150, 150),
                                   batch_size  = batchSize,
                                   class_mode  = None,
                                   shuffle     = False)

    transformed_trainData = model.predict_generator(trainGetter,
                                                    numTrainImgs//batchSize)

    # with open('fc_training_data.npy','w') as trainFile:
    #     numpy.save(trainFile, transformed_trainData)
    numpy.save('fc_training_data.npy', transformed_trainData)

    # Transfrom validation data
    numValidImgs = len([i for i in os.listdir(valid_dir)])
    validGetter = datagetter.flow_from_directory(valid_dir,
                                   target_size = (150, 150),
                                   batch_size  = batchSize,
                                   class_mode  = None,
                                   shuffle     = False)


    transformed_validData = model.predict_generator(validGetter,
                                                    numValidImgs//batchSize)

    # with open('fc_validation_data.npy', 'w') as validFile:
        # numpy.save(validFile, transformed_validData)
    numpy.save('fc_validation_data.npy', transformed_validData)

    print("Transformed data into bottleneck features.")

def trainFCLayer(numpy_tData   = 'fc_training_data.npy',
                 numpy_vData   = 'fc_validation_data.npy',
                 numpy_tLabels = 'fc_training_labels.npy',
                 numpy_vLabels = 'fc_validation_labels.npy',
                 numEpochs     = 50,
                 batchSize     = 16):

    # Load data
    tData = numpy.load(numpy_tData)
    vData = numpy.load(numpy_vData)

    # Load labels
    tLabels = numpy.load(numpy_tLabels)
    vLabels = numpy.load(numpy_vLabels)

    # Build FC model to place on top of the convolutional layers
    fcModel = Sequential()
    fcModel.add( Flatten( inputshape = tData.shape[1:]) )
    fcModel.add( Dense(256, activation='relu') )
    fcModel.add( Dropout(0.5) )
    fcModel.add( Dense(1, activation='sigmoid') )
    fcModel.compile( optimizer = 'rmsprop',
                     loss      = 'binary_crossentropy',
                     metrics   = ['accuracy'])

    # Train FC model
    fcModel.fit( tData,
                 tLabels,
                 epochs          = numEpochs,
                 batch_size      = batchSize,
                 validation_data = (vData, vLabels))

    # Save final weights to classify new data; we now have the new model
    fcModel.save_weights('final_weights.h5')

    print("New parameters acquired.")


if __name__=="__main__":
    bottleneckTransform()
    trainFCLayer()
