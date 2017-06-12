################################################################
#                                                              #
#                         doc                                  #
#                                                              #
################################################################

# overview    : obtain heatmap by looking at
#               salience patches; this is done
#               by occluding the image many times
#               and creating a numeric overlay based
#               on scores obtained for occluded copies

# note        : many copies of the input image are made;
#               each is scored according to the
#               classifier/model originally used
#               (thus salience is what the model finds salient)

# dependencies: Pillow (or PIL)
#               numpy
#               matplotlib


# heavily inspired by Ryan Compton, Clarifai
# http://blog.clarifai.com/what-convolutional-neural-networks-see-at-when-they-see-nudity/#.WSZ_VBiZPMV

################################################################
#                                                              #
#                         config                               #
#                                                              #
################################################################
from PIL import Image, ImageDraw
import numpy
import matplotlib.pyplot as matplot
from keras import models
from modelArchitects import inceptionMini1
from modelConfigs.classAll import v9
import Var
import os

os.environ['CUDA_VISIBLE_DEVICES'] = Var.GPU[2]

model = models.load_model('/data/DR/inceptionDR/matureModels/modelIncepMini3-0.74.h5')
print('Model loaded.')

# parameters
srcalpha     = 1000;    # rgba a value for src image
blendalph    = .2;      # alpha parameter for blending
occlen       = 100;      # length of side of square occlusion
stride       = 20;      # small => many calls; big => less precision;
heatmapScale = 80;      # scales salience map matrix values before overlaying
filepath     = "/data/DR/inceptionDR/modelPerfRecords/inceptionMini3-classAll.v8a/epoch8/imageLinks/3/3/3868_left.jpg"; # relative to this script's dir
imaI = 2
################################################################
#                                                              #
#                        helpers                               #
#                                                              #
################################################################

#######################################
# obtain the score for occluded image
#######################################
def getScore(occludedImage):
    x = numpy.expand_dims(occludedImage, axis=0)
    score = model.predict(x)
    #print score
    score = score[0][imaI]
    print score
    return score # model output here

#######################################
# obtain occlusions for source image
#######################################
def getOcclusions(src, occlen, stride):
    # square occlusions; p1 = top-left corner, p2 = bottom-right corner
    w = occlen;
    h = occlen;
    occlusions = [];
    for p1_y in range(0, src.size[1], stride):
        for p1_x in range(0, src.size[0], stride):
            p2_x = p1_x + w; p2_y = p1_y + h;
            occlusions.append( (p1_x, p1_y, p2_x, p2_y) );
    print("generating occlusions for source...");
    return occlusions;

#######################################
# create occluded copy of source image
#######################################
def occlude(src, occlusion):
    occluded = src.copy();
    black = ImageDraw.Draw(occluded);
    black.rectangle(occlusion, fill=True);
    print("occluding source image with ", occlusion);
    #occluded.save('occludedImages/'+str(occlusion)+'.png')
    return occluded;

#######################################
# blend salience map with source image
#######################################
def overlayHeatmap(src, heatmap):
    colormap = matplot.get_cmap('jet');
    coloredSalience = Image.fromarray(numpy.uint8(colormap(heatmap)*255));
    print("overlaying salience heatmap...")
    return Image.blend(src, coloredSalience, blendalph);

#######################################
# produce occlusion-based salience map
#######################################
def getSalienceMap(src, occlen, stride):
    # create occlusions; specified by top-left, bottom-right corners
    occlusions = getOcclusions(src, occlen, stride);

    # create occluded copies of src image
    occ_scores = [];
    for occlusion in occlusions:
        occludedImage = occlude(src, occlusion);
        occ_score = (occlusion, getScore(occludedImage));
        occ_scores.append( occ_score);
        print(occ_score);

    # create salience heatmap
    src.putalpha(srcalpha);
    heatmap = numpy.zeros(src.size);
    #scores = [(os[1]-0.9946)*10000 for os in occ_scores]
    for i in range(len(occ_scores)):
        #scores = numpy.array(scores) / numpy.amax(scores)
        print("modifying heatmap with ",  (occ_scores[i][0], occ_scores[i][1]));

        (occlusion, score) = occ_scores[i];
        mask = numpy.zeros(src.size);
        mask[occlusion[0]:occlusion[2], occlusion[1]:occlusion[3]] = score;
        heatmap = heatmap + (mask - heatmap)/(i+1);
    heatmap = numpy.transpose(heatmap) * heatmapScale;
    heatmap /= numpy.sqrt(numpy.mean(numpy.square(heatmap)))
    # return blended result
    return overlayHeatmap(src, heatmap);

################################################################
#                                                              #
#                        runner                                #
#                                                              #
################################################################
if __name__ == "__main__":
    src = Image.open(filepath).resize([800, 800]);
    #src.putalpha(srcalpha);
    salienceMap = getSalienceMap(src, occlen, stride);
    salienceMap.save('sal-oclu3.png');
    salienceMap.show();
