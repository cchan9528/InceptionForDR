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
#               (thus salience is relative)

# heavily inspired by Ryan Compton, Clariai
# http://blog.clarifai.com/what-convolutional-neural-networks-see-at-when-they-see-nudity/#.WSZ_VBiZPMV

################################################################
#                                                              #
#                         config                               #
#                                                              #
################################################################
from Pillow import Image, ImageDraw
import numpy
import matplotplib.pyplot as matplot

# constants
srcalpha = 1000;
occlen   = 64;
stride   = 48;      # small => many calls; big => less precision

################################################################
#                                                              #
#                        helpers                               #
#                                                              #
################################################################

#######################################
# obtain the score for occluded image
#######################################
def score(occludedImage):
    return # model output here

#######################################
# obtain occlusions for source image
#######################################
def getOcclusions(src, occlen, stride):
    # square occlusions
    w = occlen;
    h = occlen;
    occlusions = [];
    for top in range(0, src.size[1], stride):
        for left in range(0, src.size[0], stride):
            occlusions.append( (left, top, left+width, top+height) );
    return occlusions;

#######################################
# create occluded copy of source image
#######################################
def occlude(src, occlusion):
    occluded = src.copy();
    black = ImageDraw.Draw(occluded);
    black.rectangle(occlusion, fill=True);
    return occluded;

#######################################
# blend salience map with source image
#######################################
def overlaySalienceMap(src, salienceMap):
    colormap = matplot.get_cmap('jet');
    coloredSalience = Image.fromarray(numpy.uint8(colormap(salienceMap)*255));
    return Image.blend(src, coloredSalience, .8);

#######################################
# produce occlusion-based salience map
#######################################
def getSalienceMap(src, occArea, stride):
    # create occlusions; specified by top-left, bottom-right pts
    occlusions = getOcclusions(src, occArea, stride);

    # create occluded copies of src image
    occ_scores = [];
    for occlusion in occlusions:
        occludedImage = occlude(src, occlusion);
        occ_scores.append( (occlusion, score(occludedImage)) );

    # create salience heatmap
    salienceMap = numpy.zeros(src.size);
    for i in range(len(occ_scores)):
        (occlusion, score) = occ_scores[i];
        mask = numpy.zeros(src.size);
        mask[occlusion[0]:occlusion[2], occlusion[1]:occlusion[3]] = score;
        salienceMap = salienceMap + (mask - salienceMap)/(i+1);
    salienceMap = numpy.transpose(salienceMap) * 80;

    # return blended result
    return overlaySalienceMap(src, salienceMap);

################################################################
#                                                              #
#                        runner                                #
#                                                              #
################################################################
if __name__ = "__main__":
    src = Image.open("data/sample/10_left.jpeg"); src.putalpha(srcalpha);
    salienceMap = getSalienceMap(src, occlen, stride);
    salienceMap.show();
