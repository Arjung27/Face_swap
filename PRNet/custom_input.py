import argparse
import os
from api import PRN
from demo_texture import texture_editing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Texture Editing by PRN')

    parser.add_argument('-i', '--image_path', default='../TestSet_P2/Test1.mp4', type=str,
                        help='path to input image')
    parser.add_argument('-r', '--ref_path', default='../TestSet_P2/Rambo.jpg', type=str, 
                        help='path to reference image(texture ref)')
    parser.add_argument('-o', '--output_path', default='TestImages/output.jpg', type=str, 
                        help='path to save output')
    parser.add_argument('--mode', default=1, type=int, 
                        help='ways to edit texture. 0 for modifying parts, 1 for changing whole')
    parser.add_argument('--video', default=0, help='1 for video input and 0 for image input')
    parser.add_argument('--output_name', default='../TestSet_P2/Data1OutputPRNet.mp4', 
    															help='Name of the output file')
    parser.add_argument('--shape_predictor', default="../shape_predictor_68_face_landmarks.dat", 
    																		help="Prdictor file")
    parser.add_argument('--face', type=int, default=1, help='Number of faces to be swapped 1 or 2')
    parser.add_argument('--gpu', default='0', type=str, 
                        help='set gpu id, -1 for CPU')

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = True) 

    texture_editing(prn, parser.parse_args())