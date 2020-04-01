Overview
========

Dependencies:
=============

    1. Ubuntu 16.04
    2. python3.5
    3. Python dependencies such as numpy
    4. Tensorflow >= 1.4
    5. OpenCV 3.4.9

Instructions to run program:
============================

1)  Phase 1 (Conventional Pipeline):
```

1.  Create a folder in which you can put the requisite videos and images.
2.  Run command 'python3 dataUtils/dataGeneration.py --video <Target-Video-File-Path> --sourceImg <source-Image-File-Path>
--method <tps-Tri-TriD> --shape_predictor <landmark-dat-file> --output_name <Path-and-Name-of-output-file-with-.mp4-in-the-name>' to start generating video with faces swapped. 

Default Values have been given as inputs. The test results for the project are stored in a separate 'Test' folder inside the 'Data' folder.
```
Different methods specified under --method flag determines the algorithm that needs to be used for face swapping. tps is for This Plate Splines, Tri fror triangulation, and TriD for using triangulation on videos with two faces.

2) Phase 2 (Deep Learning):

This includes the modified version of [code](https://github.com/YadiraF/PRNet). In order to run it for our use case follow the instructions below:
```
1. Navigate to the PRNet folder.
2. Download the PRN trained model at [Google Drive](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view), and put it into Data/net-data.
3. CUDA_VISIBLE_DEVICES=0 python custom_input.py -i <path of the target video> -r <path of the source image> --video <1 if the target is video and 0 if it is image> --output_name <path of the utput video/image> --face <number of faces>
Example: CUDA_VISIBLE_DEVICES=0 python custom_input.py -i ../TestSet_P2/Test2.mp4 -r ../TestSet_P2/Scarlett.jpg --video 1 --output_name ../TestSet_P2/Data2OutputPRNet.mp4 --face 2
```
If no GOU is present on the system then remove CUDA_VISIBLE_DEVICES=0 from the above command.

