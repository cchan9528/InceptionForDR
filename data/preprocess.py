################################################################
#                                                              #
#                         doc                                  #
#                                                              #
################################################################

# overview    : preprocess data

# note        : this is used for testing

################################################################
#                                                              #
#                         config                               #
#                                                              #
################################################################
import sys, glob, os, numpy
from os import path as path
from PIL import Image

def resizeData(datadir, outputdir="resized/", size=(150,150)):
    # Create directory if needed
    try: os.makedirs(outputdir);
    except: raise;

    # Resize images and store into output directory
    for src in glob.glob(datadir+"/*"): # ok if user enters with end '/'
        if src==datadir:
            break
        srcname, srcext = path.splitext(path.basename(src))
        with Image.open(src) as res:
            res.thumbnail(size)
            resname = srcname+"_resized"+srcext
            if(srcext[0]=='.'):
                srcext=srcext[1:]
            res.save(outputdir+resname, srcext.upper())

    print("Your resized images are in " + outputdir)

def getLabels(labelfile, outputfile='labels_as_numpy_array.npy'):

    '''Expected format of a data,label : image (string), class (number)'''

    with open(labelfile) as lines:
        image_labels = [];
        for line in lines:
            image, label = line.strip().split(',')
            image_labels.append( (image, label) )
        numpy.save(outputfile, numpy.array(image_labels))

    print("Your labels are in " + outputfile + " as numpy arrays")

def partitionData(datadir="samples", t_classSamples=2000, v_classSamples=800):

    '''Move data from samples to training/validation directories'''

    # Determine if samples per class are possible
    img_classes = numpy.load('labels_as_numpy_array.npy')
    neededInClass = t_classSamples + v_classSamples
    for i in range(5):
        if( len( numpy.where( img_classes==str(i) )[0] ) < neededInClass):
            print("not enough data for class " + str(i))
            print("nothing was moved; please add more data for mentioned class")
            return -1

    # Move samples into class folders
    for classlabel in range(5):
        # Get all files within class
        filenames = [];
        rows = numpy.where( img_classes == str(classlabel) )[0]
        for row in rows:
            filenames.append( img_classes[ row, 0 ] )

        # Move the file into data type/class folder
        for i in range(neededInClass):
            if(i<t_classSamples):
                os.rename(datadir + "/" + filenames[i],
                          "training/" + str(classlabel) + "/" + filenames[i])
            else:
                os.rename(datadir + "/" + filenames[i],
                          "validation/" + str(classlabel) + "/" + filenames[i])

def addFiletypeExtension(labelfile, ext):

    ''' Add filetype extension. Expected format in file is image,score '''

    outfilename, outfileext = labelfile.split('.')
    with open(outfilename + "_with_ext." + outfileext, 'w') as outfile:
        with open(labelfile) as infile:
            for line in infile:
                filename, classlabel = line.strip().split(',')
                if '.' not in filename:
                    filename = filename + ext
                outfile.write(filename+","+classlabel+"\n");

if __name__ == "__main__":
    getLabels("labels.csv")
    partitionData(numTrain=1, numValid = 1)
    # CAREFUL IF YOUR DATASET IS HUGE!
    # resizeData("samples")
