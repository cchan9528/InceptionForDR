c.f. http://wikicoursenote.com/wiki/GoingDeeperWithConvolutions#Inception_module

============
Introduction
============

- Inception is an experimental deep CNN architecture

- Google Lenet
    - Inception architecture
        - 22 layers deep
        - (1/12)x parameters, *significantly* more accurate than ILSVRC2012 winners

==========
Motivation
==========

- Large CNN networks prone to overfitting b/c of large # of parameters

- Uniform increased network size increases computational resources

- Sparse data structures for sparse networks are inefficient
    - indirect addressing for non-contiguous data => cache misses

==========
Background
==========

- Image Convolution
    - Convolving an image with a filter/kernel can reveal notable features
        - a kernel has a number of elements that compose a 2D matrix
            - these elements can be seen as weights for the pixels
        - e.g. Sobel edge detect kernel
    - In a neural network, these kernel elements are altered with training
        - optimization minimizes loss/error by changing these weights
            - this allows for more accurate feature detection


- Convolutional Neural Networks [CNN]
    - A graph that has layers (each 2D)
    - Each layer has units
        - at depth 0, this is the pixel
        - at depth n, this is a feature of that layer
    - Considering the i_th and (i-1)_th layers,
        - the j_th unit in the i_th layer is connected to only neighboring units of (i-1)
        - the same weights are applied to each unit in the i_th layer
            - this is where "convolution" comes from
                - shifting the weight window
                    - weights are for the connections b/w units b/w layers
    - Alternative: "Max-Pooling"
        - Instead of Convolving in the conventional multiply-add sense, pick max
            - Window/Pool a region
            - Pick max value w/in window/pool
                - i.e. don't weight the inputs
            - reduces computational power
            - introduces form of translational invariance
                - expect same values w/in small translations of window

=========
Inception
=========

- Two main ideas:
    - Approximation of a sparse structure w/ spatially repeated dense components
    - Dimension reduce when computational complexity should be kept in bounds

- Technique:
    - process the previous layer with 1x1(xD) kernels to dimension reduce when needed
        - needed to preserve sparse representations as much as possible
    - execute convolutions after dimension reduction and/or dimension reduction only

- Benefits:
    - allows for increasing number of units w/o uncontrollably blowing up complexity
    - dimension reduction allows for black-boxing large number of input filters of last stage
    - 2-3x faster than performing with non-inception
    - follows intuition:
        - visual info should be processed at various scales, aggregated, repeated at next stage
