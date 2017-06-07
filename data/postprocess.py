################################################################
#                                                              #
#                         doc                                  #
#                                                              #
################################################################

# overview    : postprocess data

# note        : this is used for testing

################################################################
#                                                              #
#                         config                               #
#                                                              #
################################################################
import os

def replaceData():
    '''Move data from training/validation directories back to samples'''
    for i in range(5):
        tfiles = next(os.walk("training/" + str(i)))[2]
        vfiles = next(os.walk("validation/" + str(i)))[2]
        for tfile in tfiles:
            os.rename("training/"+str(i)+"/"+tfile, "samples/"+tfile)
        for vfile in vfiles:
            os.rename("validation/"+str(i)+"/"+vfile, "samples/"+vfile)

if __name__ == "__main__":
    replaceData();
