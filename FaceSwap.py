import numpy as np 
import cv2
import dlib
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

	vert = []
	for t in triangles:
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])
		vert.append((pt1,pt2,pt3))
		cv2.line(img, pt1, pt2, (0, 0, 255), 2)
		cv2.line(img, pt2, pt3, (0, 0, 255), 2)
		cv2.line(img, pt1, pt3, (0, 0, 255), 2)

	vert = np.asarray(vert)

	return img,vert

def affineBary(img,ini_tri,fin_tri):
	src = ini_tri
	dst = fin_tri
	min_y = np.where(src==src[:,1].min())
	y_min = src[min_y[0].item(),min_y[1].item()]
	max_y = np.where(src==src[:,1].max())
	y_max = src[max_y[0].item(),max_y[1].item()]
	min_x = np.where(src==src[:,0].min())
	x_min = src[min_x[0].item(),min_x[1].item()]
	max_x = np.where(src==src[:,0].max())
	x_max = src[max_x[0].item(),max_x[1].item()]

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
	for i in range(bc.shape[1]):
		if bc[0,i]+bc[1,i]+bc[2,i]-1<0.0001 and 0<=bc[0,i] and 0<=bc[1,i] and 0<=bc[2,i] and bc[0,i]<=1 and bc[1,i]<=1 and bc[2,i]<=1:
			Z.append(bc[:,i])

	Z = np.asarray(Z)
	Z = Z.T

	A = [[dst[0][0],dst[1][0],dst[2][0]],[dst[0][1],dst[1][1],dst[2][1]],[1,1,1]]
	coord = np.dot(A,Z)
	xA = coord[0,:]/coord[2,:]
	yA = coord[1,:]/coord[2,:]
	
	xi = np.linspace(0, img.shape[1], img.shape[1]+1)
	yi = np.linspace(0, img.shape[0], img.shape[0]+1)	
	for x,y in zip(xA,yA):
		blue = img[:,:,0]
		b = interp2d(xi, yi, blue, kind='cubic')
		bl = b(x,y)
		green = img[:,:,1]
		g = interp2d(xi, yi, green, kind='cubic')
		gr = g(x,y)
		red = img[:,:,2]
		r = interp2d(xi, yi, red, kind='cubic')
		re = r(x,y)

		print(bl.shape)


	return A
	



def main():
	'''
	#Code to convert video stream to a set of images
	fname = './TestSet_P2/Test1.mp4'
	tarname = './TestFolder'
	videoToImage(fname,tarname)
	'''
	tname = './TestFolder/Img22.jpg'
	sname = './TestSet_P2/Rambo.jpg'
	p = "shape_predictor_68_face_landmarks.dat"
	imge = cv2.imread(tname)

	IMAGE,shp = getFaceLandmarks(tname,p)
	IMG,V = triangulation(shp,IMAGE)

	IMAGEs,shps = getFaceLandmarks(sname,p)
	IMGs,Vs = triangulation(shps,IMAGEs)	
	a = affineBary(imge,V[0],Vs[0])
	#print(a[0][2]+a[1][2]+a[2][2])
	cv2.imshow("Output", IMG)
	cv2.waitKey(0)


if __name__ == '__main__':
	main()																																				