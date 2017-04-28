===============================================
Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
===============================================

Karen Simonyan
Andrea Vedaldi
Andrew Zisserman

===============================================
Abstract
===============================================

- Purpose: address CNN-based image classification model visualization

    - Highlighted vis techniques:
        - Note: these methods based on computing gradient of class core w.r.t. input image

        1: visualize notion of the class by generating an image

        2: compute class saliency map specific to a given image and class

            - these maps can be used for weakly supervised object segmentation w/ classification CNNs

- Additionally, establish connection b/w gradient-based CNN vis methods and deconv. nets

===============================================
[1] Introduction
===============================================

- Problem: understanding aspects of visual appearance captured inside a deep learning model
    - c.f. previous approaches
        - e.g. Deconvolutional Network architecture


- Purpose: address visualization of CNN classifications (subject categorization)

    1. [2] demonstrate understandable visualizations of CNN classification models can be obtained w/ numerical optimization of the input image

    2. [3] propose a method for computing spatial support of a given class in a given image (image-specific class saliency maps)

    3. [4] gradient-based visualization methods generalize deconv network reconstruction

===============================================
[2] Class Model Visualisation
===============================================
- Given a CNN that is trained for classifying images, ** numerically generate an image **

    - Consider **S<sub>c</sub>(I)**
        - S := score
        - c := class c for which score applies to
        - I := input image

    - Want to find an image I such that score S is high for specific class c

        - This can be thought of as **optimizing w.r.t I**
            - Thus, "back-propagate" w/ fixed CNN weights, changing I vector

        - In practice,
            1. Added mean image to result of zero-initialized optimization
            2. Used unnormalized class scores S<sub>c</sub> vs P<sub>c</sub>, the soft-max probabilities
                - P<sub>c</sub> could affect I optimization for specific class c

- **This can be understood as what the CNN understands/see for a given class**

===============================================
[3] Image-Specific Class Saliency Visualization
===============================================
- Spatial support of a class in an image can be extracted from CNN

    - i.e. Rank the pixels of image I based on influence for S<sub>c</sub>(I)
        - **This is Saliency**


- S<sub>c</sub>(I) is a non-linear function w.r.t. I
    - Take the 1<sup>st</sup> order taylor series approximation
        - S<sub>c</sub>(I) ~ W*I + b
            - W = weights = (dS/dI)(I<sub>0</sub>)
            - b = bias = S<sub>c</sub>(I<sub>0</sub>) - (dS/dI)(I<sub>0</sub>) * I<sub>0</sub>
                - c.f. paper

    - **The key here is W**
        - Derivative => score increases/decreases quickly @ associated pixels
            - Interpretation: These pixels correspond to object location
                - **Salient Pixels**

        - This is **because W is defined to be dS/dI**
            - thus we know which ones cause the greatest effect

- **This can be understood as what the CNN picks out of a given image when thinking about one particular class**

===============================================
[3.1] Class Saliency Extraction
===============================================
- To actually form saliency maps for a specific class c,

    - Grey-Scale:
        - Take magnitude of W and map back to original dimensions of I

    - Multi-Channel:
        - Take maximum of W<sub>i,j</sub> across all three channels
        - Repeat for all (i,j)
        - Map back to original dimensions of I

- Only image labels were used for training; No additional annotation was required/used
    - i.e. "weakly supervised"

===============================================
[3.2] Weakly Supervised Object Localisation
===============================================
- The saliency maps produced in previously mentioned method provide object location information
    - can thus be used to localise objects

- With an image and corresponding salience map,
    - Use GraphCut color segmentation algorithm
        - salience map may only capture most distinguishable features

    - In practice:

        - Gaussian Mixture Models (foreground and background)
            - foreground estimated from pixels w/ saliency higher than 95% quantile of saliency distribution
            - background estimated from pixels w/ saliency lower than 30% quantile of saliency distribution

        - Run GraphCut

        - Choose largest connected component of foreground pixels

===============================================
[4] Relation to Deconvolutional Networks
===============================================
- DNN effectively corresponds to gradient back-propagation through CNN

    - Computing approximate feature map reconstruction R<sub>n</sub> w/ DNN is equivalent to dF/dX<sub>N</sub> w/back-propagation
        - X<sub>N</sub> is N<sup>th</sup> convolutional layer in CNN

    - Note: gradient-based techniques can be applied to any layer, not just convolutional
        - Thus, gradient-based techniques can be use for visualization

        - **gradient-based techniques provide feature maps like DNN reconstructions; thus provide visualizations (of scores)**

===============================================
[5] Conclusion
===============================================
- 2 Visualization Techniques used for deep CNN classifiers

    1. Artificial Image Generation of a Given Class

    2. Image-specific Class Saliency Map

- Gradient-based visualisation techniques generalize DNN reconstruction procedure
