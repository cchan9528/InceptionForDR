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
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing.image import ImageDataGenerator

def directoryReport(datadir):
    label = [0, 0, 0, 0, 0]
    for i in range(5):
        label[i] = len(os.listdir(datadir + '/' + str(i)))
    print("Num files in \'" + datadir + '\' classes 0, 1, 2, 3, 4')
    print(str(label) + " => " + str(sum(label)) + " total files")
    return label

def generateClassSamples(total, datadir='training'):

    ''' Generate and insert 'total' new samples for each class'''

    # @Report
    old = [0, 0, 0, 0, 0]
    for i in range(5):
        old[i] = len(os.listdir(datadir + '/' + str(i)))

    # Configure the generator
    sampleGenerator = ImageDataGenerator( rotation_range = 40,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode='nearest' )
    batch = sampleGenerator.flow_from_directory( directory = datadir,
                                                 batch_size = 1, # KEEP @ 1
                                                 shuffle = False,
                                                 save_to_dir='extra/generated')

    # Create (total * 5) number of new samples
    fns = batch.filenames
    print(fns)
    numsources = len(fns)
    needed = [total] * 5
    remaining = sum(needed)
    while remaining != 0:
        next(batch)
        targetPath, ext = fns[batch.batch_index-1].strip().split('.')
        thisClass = int(targetPath.split('/')[0])
        if needed[thisClass] == 0:
            continue
        targetPath += '_' + str(remaining) + '.' + ext
        for fname in os.listdir('extra/generated'):
            os.rename('extra/generated/' + fname, datadir + '/' + targetPath)
        needed[thisClass] -= 1
        remaining -=1
        print("Total Remaining: " + str(remaining))

    # Report
    new = [0, 0, 0, 0, 0]
    for i in range(5):
        new[i] = len(os.listdir(datadir + '/' + str(i)))
    print("Num files in \'" + datadir + '\' classes 0, 1, 2, 3, 4')
    print("Before: " + str(old) + " => " + str(sum(old)) + " total files")
    print("After: " + str(new) + " => " + str(sum(new)) + " total files")
    print(str(sum(new)-sum(old)) + " new files in \'" + datadir + '\'')

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

def partitionClasses(datadir="extra"):

    '''Categorize all data by class'''

    img_class = numpy.load('labels_as_numpy_array.npy')
    all_samples = next(os.walk(datadir))[2]
    for sample in all_samples:
        label = img_class[ (numpy.where(img_class == sample))[0] ][0][1]
        print(datadir+'/'+sample+" --> "+datadir+'/'+label+'/'+sample)
        os.rename(datadir+'/'+sample, datadir+'/'+label+'/'+sample)
    directoryReport(datadir)

def partitionTV(datadir="extra", t_classSamples=2000, v_classSamples=800):

    '''Move data from samples to training/validation directories'''

    # Determine if samples per class are possible
    numExistPerClass = directoryReport(datadir)
    needed = t_classSamples + v_classSamples
    for i in numExistPerClass:
        if i < needed:
            print("nothing was moved; add more data classes"); return -1;

    # Move samples into class folders
    for classlabel in range(5):
        extra_class_samples = datadir + '/' + str(classlabel)
        filenames = os.listdir(extra_class_samples);
        for i in range(needed):
            targetDir = "validation/" if ( i>=t_classSamples )  else "training/"
            os.rename(extra_class_samples + "/" + filenames[i],
                      targetDir + str(classlabel) + "/" + filenames[i])

    print("Partitioning complete")

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
