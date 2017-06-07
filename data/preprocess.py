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
    img_class = numpy.load('labels_as_numpy_array.npy')
    needed = t_classSamples + v_classSamples
    all_samples = next(os.walk(datadir))[2]
    class_samples = []
    for i in range(5):
        class_masterlist = img_class[ numpy.where( img_class == str(i) )[0], 0 ]
        exist_samples = [img for img in class_masterlist if img in all_samples]
        class_samples.append(exist_samples);

    # Report
    flag = 0
    for i in range(5):
        print("class"+str(i)+":"+str(len(class_samples[i]))+"/"+str(needed))
        if( len( class_samples[i] ) < needed): flag = 1;
    if flag: print("nothing was moved; add more data classes"); return -1;

    # Move samples into class folders
    for classlabel in range(5):
        filenames = class_samples[classlabel];
        for i in range(needed):
            sampleDir = "validation/" if ( i>=t_classSamples )  else "training/"
            os.rename(datadir + "/" + filenames[i],
                      sampleDir + str(classlabel) + "/" + filenames[i])


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
