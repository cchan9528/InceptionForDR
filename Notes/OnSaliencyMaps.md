c.f. http://www.scholarpedia.org/article/Saliency_map

==========
Definition
==========

- Salience : the quality of being particularly noticeable or important
    -i.e. pick out the important things in the scene

- Saliency maps integrate the normalized info from feature maps into one global conspicuity metric

    - They collect and emphasize bottom-up factors that draw a person's attention
        - bottom-up: instantaneous sensory input w/o info about internal state
            - e.g. firecracker going off
        - top-down: considers info about internal state
            - e.g. find difficult-to-find food items by a hungry animal

- Saliency depends on how different a stimulus is from its surroundings

- Use-flow:
    - Establish the topographical salience map
    - Find the position of the maximum salience
    - (Naturally) move to the 2nd highest, 3rd, etc.

- \*\*\* Bottom-up mechanisms (thus saliency maps) **DON'T completely determine attentional selection**

=======
Example
=======

![Example of Saliency Map](http://geometry.cs.ucl.ac.uk/projects/2014/globalContrast/paper_docs/teaser.JPG)

    (Top) Input
    (Middle) Hi-Res Salience
    (Bottom) ROI Mask
    SRC: http://bit.ly/2p3Uv7w
