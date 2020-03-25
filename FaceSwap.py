import numpy as np 
import cv2
import dlib
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

def getFaceLandmarks(fname,p):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(p)

	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	bbox = detector(gray,0)

	for (i,bbox) in enumerate(bbox):
		shape = predictor(gray,bbox)
		shape = face_utils.shape_to_np(shape)
		
		for (x,y) in shape:
			cv2.circle(img,(x,y),2,(0,255,0),-1)

	return img,shape

def triangulation(land_points,img):
	points = np.array(land_points, np.int32)
	chull = cv2.convexHull(points)
	rect = cv2.boundingRect(chull)
	subdiv = cv2.Subdiv2D(rect)
	p_list = []
	for p in land_points:
		p_list.append((p[0],p[1]))
	for p in p_list:
		subdiv.insert(p)
	triangles = subdiv.getTriangleList()
	triangles = np.array(triangles, dtype=np.int32)

	for t in triangles:
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])

		cv2.line(img, pt1, pt2, (0, 0, 255), 2)
		cv2.line(img, pt2, pt3, (0, 0, 255), 2)
		cv2.line(img, pt1, pt3, (0, 0, 255), 2)

	return img


def main():
	'''
	#Code to convert video stream to a set of images
	fname = './TestSet_P2/Test1.mp4'
	tarname = './TestFolder'
	videoToImage(fname,tarname)
	'''
	fname = './TestFolder/Img22.jpg'
	p = "shape_predictor_68_face_landmarks.dat"

	IMAGE,shp = getFaceLandmarks(fname,p)
	IMG = triangulation(shp,IMAGE)
	cv2.imshow("Output", IMG)
	cv2.waitKey(0)


if __name__ == '__main__':
	main()																																				