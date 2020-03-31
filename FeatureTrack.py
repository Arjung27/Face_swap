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
	sname = './TestSet_P2/Rambo.jpg'
	img1 = cv2.imread(sname)
	X = np.zeros((frame_height,frame_width,3))
	#while(cap.isOpened()):
	while(cap.isOpened()):
		ret,frame = cap.read()
		if ret == False:
			break
		cv2.imwrite('init.jpg',frame)
		lname = 'init.jpg'
		IMAGE,shp = getFaceLandmarks(lname,p)

		shp = np.asarray(shp,dtype='float64')
		st = shp.flatten().T
		st =np.reshape(st,(st.shape[0],1))
		#print(st.shape)
		vel = np.zeros((136,1),dtype='float64')
		state = np.concatenate((st,vel),axis=0)
		#print(state.shape)
		#print(state.shape)
		kalman = cv2.KalmanFilter(272, 136, 0)
		trMatrix = np.zeros((272,272),dtype='float64')
		for j in range(trMatrix.shape[0]):
			for k in range(trMatrix.shape[1]):
				if j==k or (k-136)==i:
					trMatrix[j][k]=1.0
				else:
					trMatrix[j][k]=0.0

		kalman.transitionMatrix = trMatrix
		#kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
        #                                [0., 1., 0., .1],
        #                                [0., 0., 1., 0.],
        #                                [0., 0., 0., 1.]])
		kalman.measurementMatrix = 1.*np.eye(136, 272)
		kalman.processNoiseCov = 1e-5 * np.eye(272)
		kalman.measurementNoiseCov = 1e-3 * np.eye(136, 136)
		kalman.errorCovPost = 1e-1 * np.eye(272, 272)
		kalman.statePost = np.zeros((272,1),dtype='float64')
		while(1):
			ret, frame = cap.read()  # read another frame
			if ret == False:
				break
			cv2.imwrite('temp.jpg',frame)
			tname = 'temp.jpg'
			img2 = cv2.imread(tname)
			prediction = kalman.predict()
			prediction = prediction.T
			prediction = np.reshape(prediction,(136,2))
			
			pred = prediction[:68,:]
			#for (x,y) in pred:
				#cv2.circle(frame,(int(x),int(y)),2,(0,255,0),-1)
			#------------------------------------)
			if (pred[:,0]>480).all():
				print((pred[:,0]>480).all())
				F = frame
			elif i==37 or i==38 or i==99 or i==106 or i==117 or i ==121 or i ==127 or i ==130 or i==136 or i==147 or i==201:
				F = X
			else:
				pred = np.array(pred,dtype='int32')
				_,V,VP,rectangle = triangulation(pred,frame)
				_,shps = getFaceLandmarks(sname,p)
				Vs,IMGs = doTriangulate(shps,VP,img1)
				a,b = swapFace(img2,img1,V,Vs)
				A = interpolate(img1,img2,a,b)
				F = blendFace(rectangle,img2,A)
			#-------------------------------------
			IMAGE,shp = getFaceLandmarks(tname,p)
			shp = np.asarray(shp,dtype='float64')
			st = shp.flatten().T
			st =np.reshape(st,(st.shape[0],1))
			measurement = st
			#print(type(measurement[0][0]))
			final = prediction
			if (shp[0,0]!=0.0):
				posterior = kalman.correct(measurement)
				final = posterior
			cv2.imshow('Out',F)
			cv2.waitKey(0)
			vidWriter.write(F)
			X=F
			i+=1
			print(i)

	cap.release()
	vidWriter.release()

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
		#cv2.line(img, tuple(points[temp[0]]), tuple(points[temp[1]]), (0, 0, 255), 2)
		#cv2.line(img, tuple(points[temp[1]]), tuple(points[temp[2]]), (0, 0, 255), 2)
		#cv2.line(img, tuple(points[temp[0]]), tuple(points[temp[2]]), (0, 0, 255), 2)
	vert = np.asarray(vert)
	VPoint = np.asarray(VPoint)
	chull = np.reshape(chull,(chull.shape[0],chull.shape[2]))
	return img,vert,VPoint,chull

def doTriangulate(PIndex,TIndex,img):
	T = []
	for ti in TIndex:
		T.append((PIndex[ti[0]],PIndex[ti[1]],PIndex[ti[2]]))
	T = np.asarray(T)
	#for t in T:
	#	cv2.line(img, tuple(t[0]), tuple(t[1]), (0, 0, 255), 2)
	#	cv2.line(img, tuple(t[1]), tuple(t[2]), (0, 0, 255), 2)
	#	cv2.line(img, tuple(t[0]), tuple(t[2]), (0, 0, 255), 2)
	return T, img

def affineBary(img,ini_tri,fin_tri,size):
	src = ini_tri
	dst = fin_tri
	x_min = min(src[0,0],src[1,0],src[2,0])
	x_max = max(src[0,0],src[1,0],src[2,0])
	y_min = min(src[0,1],src[1,1],src[2,1])
	y_max = max(src[0,1],src[1,1],src[2,1])
	x = np.linspace(x_min, x_max, x_max-x_min+1)
	y = np.linspace(y_min, y_max, y_max-y_min+1)
	mesh = np.meshgrid(x,y)
	mesh = np.asarray(mesh)
	mesh = mesh.reshape(*mesh.shape[:1], -1)
	grid = np.vstack((mesh, np.ones((1, mesh.shape[1]))))
	B = [[src[0][0],src[1][0],src[2][0]],[src[0][1],src[1][1],src[2][1]],[1,1,1]]
	B_inv = np.linalg.inv(B)

	bc = np.dot(B_inv,grid)

	Z = []
	D = []
	for i in range(bc.shape[1]):
		if bc[0,i]+bc[1,i]+bc[2,i]-1<0.0001 and 0<=bc[0,i] and 0<=bc[1,i] and 0<=bc[2,i] and bc[0,i]<=1 and bc[1,i]<=1 and bc[2,i]<=1:
			Z.append(bc[:,i])
			D.append((grid[0,i],grid[1,i]))

	Z = np.asarray(Z)
	Z = Z.T
	D = np.asarray(D,dtype='int32')
	D = D.T
	if len(Z)==0:
		Cs = np.zeros((1,3))
		Ds = np.zeros((1,3))
		return Cs,Ds
	A = [[dst[0][0],dst[1][0],dst[2][0]],[dst[0][1],dst[1][1],dst[2][1]],[1,1,1]]
	coord = np.dot(A,Z)
	xA = coord[0,:]/coord[2,:]
	yA = coord[1,:]/coord[2,:]

	C = [xA,yA]
	C = np.asarray(C)

	return C,D

def interpolate(img,size,pts,det):
	xi = np.linspace(0, img.shape[1], img.shape[1],endpoint=False)
	yi = np.linspace(0, img.shape[0], img.shape[0],endpoint=False)
	#dest = np.zeros((size[0],size[1],3), np.uint8)
	blue = img[:,:,0]
	b = interp2d(xi, yi, blue, kind='cubic')
	green = img[:,:,1]
	g = interp2d(xi, yi, green, kind='cubic')
	red = img[:,:,2]
	r = interp2d(xi, yi, red, kind='cubic')
	for i,(x,y) in enumerate(pts):
		bl = b(x,y)
		gr = g(x,y)
		re = r(x,y)
		size[det[i,1],det[i,0]] = (bl,gr,re)

	return size



def swapFace(d_img,s_img,d_tri,s_tri):
	L = np.zeros((2,1))
	D = np.zeros((2,1))

	for i in range(d_tri.shape[0]):
		z,m = affineBary(s_img,d_tri[i],s_tri[i],d_img.shape)
		if (z[0][0]==0) and (m[0][0]==0):
			continue
		L = np.concatenate((L,z),axis=1)
		D = np.concatenate((D,m),axis=1)
	L = np.asarray(L)
	L = L.T
	L = L[1:,:]
	D = np.asarray(D,dtype='int32')
	D = D.T
	D = D[1:,:]

	return L,D

def blendFace(hull,dimg,face):
	mask = np.zeros_like(dimg)
	cv2.fillPoly(mask, [hull], (255, 255, 255))
	r = cv2.boundingRect(np.float32([hull]))
	center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
	#cv2.circle(dimg,center,2,(0,255,0),-1)
	output = cv2.seamlessClone(np.uint8(face), dimg, mask, center, cv2.MIXED_CLONE)
	return output

def getFaceLandmarks(fname,p):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(p)
	img = cv2.imread(fname)
	#img = imutils.resize(img,width = 320)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	bbox = detector(gray,0)
	if (len(bbox)==0):
		s = np.zeros((68,2))
		return img,s
	for (i,bbox) in enumerate(bbox):
		shape = predictor(gray,bbox)
		shape = face_utils.shape_to_np(shape)
		
		#for (x,y) in shape:
		#	cv2.circle(img,(x,y),2,(0,255,0),-1)

	return img,shape


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