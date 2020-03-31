import numpy as np 
import cv2
import dlib
import copy
import sys
import argparse
import math
from scipy import interpolate
from imutils import face_utils

def videoToImage(fname,tarname):
    cap = cv2.VideoCapture(fname)
    i=0
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(tarname+'/Img'+str(i)+'.jpg',frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()

def drawFaceLandmarks(img, rects):

    for (i,rects) in enumerate(rects):
        shape = predictor(img,rects)
        shape = face_utils.shape_to_np(shape)
    
        for (x,y) in shape:
            cv2.circle(img_drawn,(x,y),2,(0,255,0),-1)

def potentialEnergy(r):
    return (r**2)*(math.log(r**2))

def funcxy(index, points_tar, wt_x, wt_y):

    K = np.zeros((points_tar.shape[0], 1))
    value = np.zeros((index.shape[0],2))
    epsilon = 1e-11
    for j, pt1 in enumerate(index):
        for i, pt2 in enumerate(points_tar):
            K[i] = potentialEnergy(np.linalg.norm(pt2 - pt1, ord=2) + epsilon)
        
        # Implementing a1 + (a_x)x + (a_y)y + + np.matmul(K.T, wt[:-3])
        value[j,0] = wt_x[-1] + pt1[0]*wt_x[-3] + pt1[1]*wt_x[-2] + np.matmul(K.T, wt_x[:-3])
        value[j,1] = wt_y[-1] + pt1[0]*wt_y[-3] + pt1[1]*wt_y[-2] + np.matmul(K.T, wt_y[:-3])

    return value

def warp_images(img_tar, img_src, pt_tar, pt_src, wt_x, wt_y, K):

    # cv2.imshow("image", img_tar)
    # cv2.waitKey(0)
    mask = np.zeros_like(img_tar[:,:,0], np.uint8)
    img_gray = cv2.cvtColor(img_tar, cv2.COLOR_BGR2GRAY)
    convex_hull = cv2.convexHull(pt_tar)
    mask = cv2.fillConvexPoly(mask, convex_hull, 255)
    mask = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    pt1_min = np.asarray(([min(pt_tar[:,0]),min(pt_tar[:,1])])).astype(np.float32)
    pt2_min = np.asarray(([min(pt_src[:,0]),min(pt_src[:,1])])).astype(np.float32)
    pt1_max = np.asarray(([max(pt_tar[:,0]),max(pt_tar[:,1])])).astype(np.float32)
    pt2_max = np.asarray(([max(pt_src[:,0]),max(pt_src[:,1])])).astype(np.float32)

    x = np.arange(pt1_min[0],pt1_max[0]).astype(int)
    y = np.arange(pt1_min[1],pt1_max[1]).astype(int)
    # print(pt1_min[0],pt1_max[0], pt1_min[1],pt1_max[1], mask.shape)
    X,Y = np.mgrid[x[0]:x[-1],y[0]:y[-1]]
    X = np.reshape(X.flatten(), [X.shape[0]*X.shape[1],1])
    Y = np.reshape(Y.flatten(), [Y.shape[0]*Y.shape[1],1])
    index = np.hstack([X,Y])
    x_coord = np.zeros(((X.shape[0]),1))
    y_coord = np.zeros(((Y.shape[0]),1))

    value = funcxy(index, pt_tar, wt_x, wt_y)
    x_coord = value[:,0]
    x_coord[x_coord < pt2_min[0]] = pt2_min[0]
    x_coord[x_coord > pt2_max[0]] = pt2_max[0]
    y_coord = value[:,1]
    y_coord[y_coord < pt2_min[1]] = pt2_min[1]
    y_coord[y_coord > pt2_max[1]] = pt2_max[1]

    blue = interpolate.interp2d(range(img_src.shape[1]), range(img_src.shape[0]), img_src[:,:,0], kind='cubic')
    green = interpolate.interp2d(range(img_src.shape[1]), range(img_src.shape[0]), img_src[:,:,1], kind='cubic')
    red = interpolate.interp2d(range(img_src.shape[1]), range(img_src.shape[0]), img_src[:,:,2], kind='cubic')
    m = interpolate.interp2d(range(mask.shape[1]), range(mask.shape[0]), mask, kind='cubic')

    warped_img = img_tar.copy()
    mask_warped_img = np.zeros_like(warped_img[:,:,0])
    
    for a in range(x_coord.shape[0]):

        intesity = mask[index[a,1],index[a,0]]
        if intesity>0:
            warped_img[index[a,1],index[a,0],0] = blue(x_coord[a], y_coord[a])
            warped_img[index[a,1],index[a,0],1] = green(x_coord[a], y_coord[a])
            warped_img[index[a,1],index[a,0],2] = red(x_coord[a], y_coord[a])
            mask_warped_img[index[a,1],index[a,0]] = 255

    r = cv2.boundingRect(mask_warped_img)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_img, img_tar, mask_warped_img, center, cv2.NORMAL_CLONE)

    return output

def initializeDlib(p):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    return detector, predictor

def findFeatures(img, detector, predictor):

    rects = detector(img, 1)
    if len(rects) == 0:
        return False, 0
    else:
        for (i, rect) in enumerate(rects):

            shape = predictor(img, rect)
            shape = face_utils.shape_to_np(shape)

    return True, shape

def thinSplateSplineMat(points_tar, points_src):

    # Genrating the matrix [[K, P], [P.T, 0]] where P = (x,y,1)
    ones_mat = np.ones([points_tar.shape[0], 1])
    P = np.hstack([points_tar, ones_mat])
    P_trans = np.transpose(P)
    zero_mat = np.zeros((3,3))
    K = np.zeros([points_tar.shape[0], points_tar.shape[0]])
    epsilon = 1e-11
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i, j] = potentialEnergy(np.linalg.norm(points_tar[i] - points_tar[j], ord=2) + epsilon)

    row_one = np.hstack([K, P])
    row_two = np.hstack([P_trans, zero_mat])
    splate_mat = np.vstack([row_one, row_two])
    # Tune the labda for better results
    tune_lam = 1e-06
    identity = tune_lam*np.identity(splate_mat.shape[0])
    splate_mat_inv = np.linalg.inv(splate_mat + identity)
    V = np.concatenate([points_src, np.zeros([3,])])
    V = np.reshape(V, [V.shape[0],1])
    wt_coord = np.matmul(splate_mat_inv, V)

    return wt_coord, K

def main_tps(Flags):

    target_video = Flags.video
    source_image = Flags.sourceImg
    method = Flags.method
    detector, predictor = initializeDlib(Flags.shape_predictor)
    # print(target_video)
    cap = cv2.VideoCapture(target_video)
    image_source = cv2.imread(source_image)
    ret, trial = cap.read()
    h, w, _ = trial.shape
    print(h, w)
    vidWriter = cv2.VideoWriter(Flags.output_name,cv2.VideoWriter_fourcc(*'mp4v'), 24, (w, h))
    i = 0

    while (cap.isOpened()):

        print('Frame Number {}'.format(i))
        i += 1
        ret, img_target = cap.read()
        if ret == False:
            break
        # Creating copy of the target image
        img_tar = copy.deepcopy(img_target)
        img_src = image_source.copy()
        # Second parameter is the number of image pyramid layers to 
        # apply when upscaling the image prior to applying the detector 
        rects = detector(img_target, 1)
        index = np.max((0, len(rects)-2))

        if len(rects) == 1:

            img_tar = img_tar[int(rects[0].top()-50):int(rects[0].bottom()+50), \
                        int(rects[0].left()-50):int(rects[0].right()+50)]
        if len(rects) > 1:
            img_src = img_tar[int(rects[len(rects)-1].top()-50):int(rects[len(rects)-1].bottom()+50), \
                                int(rects[len(rects)-1].left()-50):int(rects[len(rects)-1].right()+50)]

            img_tar = img_tar[int(rects[len(rects)-2].top()-50):int(rects[len(rects)-2].bottom()+50), \
                                int(rects[len(rects)-2].left()-50):int(rects[len(rects)-2].right()+50)]

        if len(rects) > 0:

            flag_tar, points_tar = findFeatures(img_tar, detector, predictor)
            flag_src, points_src = findFeatures(img_src, detector, predictor)
            if (not flag_tar or not flag_src):
                continue

            wt_x, K = thinSplateSplineMat(points_tar, points_src[:,0])
            wt_y, K = thinSplateSplineMat(points_tar, points_src[:,1])
            warped = warp_images(img_tar, img_src, points_tar, points_src, wt_x, wt_y, K)
            img_target[int(rects[index].top()-50):int(rects[index].bottom()+50), \
                                    int(rects[index].left()-50):int(rects[index].right()+50)] = warped

            vidWriter.write(img_target)

            if len(rects) > 1:

                wt_x, K = thinSplateSplineMat(points_src, points_src[:,0])
                wt_y, K = thinSplateSplineMat(points_src, points_src[:,1])
                warped = warp_images(img_src, img_tar, points_src, points_tar, wt_x, wt_y, K)
                img_target[int(rects[len(rects)-1].top()-50):int(rects[len(rects)-1].bottom()+50), \
                                    int(rects[len(rects)-1].left()-50):int(rects[len(rects)-1].right()+50)] = warped

                vidWriter.write(img_target)

        else:
            vidWriter.write(img_target)
            continue

if __name__ == '__main__':
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--video', default='./TestSet_P2/Test1.mp4', help='Enter the path of target video')
    Parser.add_argument('--sourceImg', default='TestSet_P2/Rambo.jpg', help='Enter the path of source image')
    Parser.add_argument('--method', default='tps', help='Type the name of the method')
    Parser.add_argument('--shape_predictor', default="shape_predictor_68_face_landmarks.dat", help="Prdictor file")
    Parser.add_argument('--output_name', default='Data1OutputTPS.mp4', help='Name of the output file')
    Flags = Parser.parse_args()

    if Flags.method == 'tps':
        main_tps(Flags)