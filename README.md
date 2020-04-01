Overview
========

Dependencies:
=============

    1. Ubuntu 16.04
    2. python3.5
    3. Python dependencies such as numpy
    4. Tensorflow 1.12.0
    5. OpenCV 3.4.9

Instructions to run program:
============================

1)  Phase 1 (Conventional Pipeline): \`\`\`

1.  Create a folder in which you can put the requisite videos and images.
2.  Run command 'python3 dataUtils/dataGeneration.py --video <Target-Video-File-Path> --sourceImg <source-Image-File-Path>
--method <tps-Tri-TriD> --shape_predictor <landmark-dat-file> --output_name <Path-and-Name-of-output-file-with-.mp4-in-the-name>' to start generating video with faces swapped. 

Default Values have been given as inputs. The test results for the project are stored in a separate 'Test' folder inside the 'Data' folder.

\`\`\`


