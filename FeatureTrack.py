import numpy as np 
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.interpolate import interp2d

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

def videoDetector(fname,p):

	cap = cv2.VideoCapture(fname)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	vidWriter = cv2.VideoWriter("./video_output.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 24, (frame_width, frame_height))
	i=0
	while(cap.isOpened()):
		ret,frame = cap.read()
		if ret == False:
			break
		IMAGE,shp = getFaceLandmarks(frame,p)

		shp = np.asarray(shp,dtype='float64')
		state = shp
		print(state.shape)
		kalman = cv2.KalmanFilter(4, 2, 0)
		kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
		kalman.measurementMatrix = 1. * np.eye(2, 4)  # you can tweak these to make the tracker
		kalman.processNoiseCov = 1e-5 * np.eye(4, 4)  # respond faster to change and be less smooth
		kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
		kalman.errorCovPost = 1e-1 * np.eye(4, 4)
		kalman.statePost = state 
		while(1):
			ret, frame = cap.read()  # read another frame
			if ret == False:
				break
			prediction = kalman.predict()
			IMAGE,shp = getFaceLandmarks(frame,p)
			measurement = shp
			final = prediction
			if (shp==None):
				posterior = kalman.correct(measurement)
				final = posterior
			vidWriter.write(IMAGE)
			i+=1
			print(i)

	cap.release()
	vidWriter.release()


def getFaceLandmarks(img,p):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(p)
	#img = cv2.imread(fname)
	#img = imutils.resize(img,width = 320)
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
	rectangle = np.asarray(rect)
	subdiv = cv2.Subdiv2D(rect)
	p_list = []
	for p in land_points:
		p_list.append((p[0],p[1]))
	for p in p_list:
		subdiv.insert(p)
	triangles = subdiv.getTriangleList()
	triangles = np.array(triangles, dtype=np.int32)
	vert = []
	VPoint = []
	pt = []
	for t in triangles:
		pt.append((t[0], t[1]))
		pt.append((t[2], t[3]))
		pt.append((t[4], t[5]))
		temp = []
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])
		
		for i in range(3):
			for j in range(len(points)):
				if(abs(pt[i][0] - points[j][0]) < 1.0 and abs(pt[i][1] - points[j][1]) < 1.0):
					temp.append(j)
		if len(temp)==3:
			vert.append((points[temp[0]],points[temp[1]],points[temp[2]]))
			VPoint.append((temp[0],temp[1],temp[2]))
		pt=[]
		
		'''
		for i in range(len(points)):
			if (abs(pt1[0]-points[i][0])<1.0 and abs(pt1[1]-points[i][1])<1.0):
				temp.append(i)
			if (abs(pt2[0]-points[i][0])<1.0 and abs(pt2[1]-points[i][1])<1.0):
				temp.append(i)
			if (abs(pt3[0]-points[i][0])<1.0 and abs(pt3[1]-points[i][1])<1.0):
				temp.append(i)
		if len(temp)==3:
			vert.append((points[temp[0]],points[temp[1]],points[temp[2]]))
			VPoint.append((temp[0],temp[1],temp[2]))
		'''
		cv2.line(img, tuple(points[temp[0]]), tuple(points[temp[1]]), (0, 0, 255), 2)
		cv2.line(img, tuple(points[temp[1]]), tuple(points[temp[2]]), (0, 0, 255), 2)
		cv2.line(img, tuple(points[temp[0]]), tuple(points[temp[2]]), (0, 0, 255), 2)
	vert = np.asarray(vert)
	VPoint = np.asarray(VPoint)
	chull = np.reshape(chull,(chull.shape[0],chull.shape[2]))
	return img,vert,VPoint,chull


def main():
	'''
	#Code to convert video stream to a set of images
	fname = './TestSet_P2/Test3.mp4'
	tarname = './Batman'
	videoToImage(fname,tarname)
	'''
	
	tname = './TestFolder/Img25.jpg'
	sname = './TestSet_P2/Rambo.jpg'
	fname = './TestSet_P2/Test1.mp4'
	p = "shape_predictor_68_face_landmarks.dat"
	videoDetector(fname,p)
	'''
	img1 = cv2.imread(sname)
	img2 = cv2.imread(tname)

	IMAGE,shp = getFaceLandmarks(img2,p)
	#IMG,V,VP,rectangle = triangulation(shp,IMAGE)
	cv2.imshow('Output',IMAGE)
	cv2.waitKey(0)
	'''
	

if __name__ == '__main__':
	main()																																				