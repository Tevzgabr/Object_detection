import cv2
import imutils
import numpy as np
import math
import calibration as cal
import shape_detection as shape
import center_markerja as center


## Load the test and calibration images 
calib_img = cv2.imread('kalib_slika_koncna.png')
test_img= cv2.imread('test_koncni.png')
## Calibration of the test image
calibrated_img = cal.img_cal(calib_img,test_img)
## HSV converion for color definition
hsv = cv2.cvtColor(calibrated_img,cv2.COLOR_BGR2HSV)
## Masking of the image
mask = 255*np.ones(calibrated_img.shape, dtype = "uint8")
cv2.rectangle(mask, (300, 0), (1000, 380), (0, 0, 0), -1)
maskedImg = cv2.bitwise_or(calibrated_img, mask)
## Load the image, convert it to grayscale, blur it slightly,
## and threshold it
gray = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2GRAY)
gray = -gray + 255
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 190  , 255, cv2.THRESH_BINARY)[1]

## Find contours in the thresholded image
cnts= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cx = []
cy= []
shape1 =[]
angles = []
contures=[]
# loop over the contours
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	if(M['m00']!=0):
		#find centroid
		cX = int(M['m10'] / M['m00'])
		cY = int(M['m01'] / M['m00'])

	else:
		cX,cY=0,0 
	shape1.append(shape.detect(c))
	contures.append(c)
	cx.append(cX)
	cy.append(cY)
	## Draw the found contours
	cv2.drawContours(calibrated_img, [c], -1, (0, 255, 0), 2)
	cv2.circle(calibrated_img,(cX,cY),3,(255,255,255),-1)
	## Find and draw the minimal rectangle around every contour
	cent,width,angle = cv2.minAreaRect(c)
	box1 = np.int0(cv2.boxPoints((cent,width,angle)))
	cv2.drawContours(calibrated_img,[box1],0,(0,0,255),2)
	angles.append(angle)


## Define the center of the marker and the lenght of the sides (in pixels)
center_m,x_dist1,x_dist2,y_dist1,y_dist2 = center.center(calibrated_img)
x = math.floor(center_m[0]) ## float to int conversion because of cv2.circle
y = math.floor(center_m[1])
cv2.circle(calibrated_img, (x, y), 3, (255, 255, 255), -1)
true_angles = []
for i in range(len(angles)):
	angles1 = angles[i] + 90
	true_angles.append(angles1)

print('The angles are: ',true_angles)
distance = []
vectors = []
colors = []
object_shape = []
font = cv2.FONT_HERSHEY_SIMPLEX
i = 0
for i in range(len(cx)):
	# cv2.line(calibrated_img,(cols-1,right1[i]),(0,lefty1[i]),(0,255,0),2)
	# compute the centres, distances and vectors from the objects and the marker
	shapes = shape1[i]
	center = [cx[i],cy[i]]
	dist = (math.sqrt((center[0] - center_m[0])**2+(center[1] - center_m[1])**2))/y_dist1
	vector_x = (center[0] - center_m[0])/y_dist1
	vector_y = (abs(center[1] - center_m[1]))/y_dist1  ## y has to  be positive, because the y in the picture points downwards
	vector =[vector_x, vector_y]
	distance.append(dist)
	vectors.append(vector)
	h1 = hsv[cy[i],cx[i],0]
	s1 = hsv[cy[i],cx[i],1]
	v1 = hsv[cy[i],cx[i],2]
	# determine the shape of the recognized object
	if shapes == 'triangle':
		object_shape ='triangel'
		cv2.putText(calibrated_img,'triangle',(cx[i]+10,cy[i]+30), font, 0.5,(255,255,255),2,cv2.LINE_AA)
	elif shapes == 'square':
		cv2.putText(calibrated_img,'square',(cx[i]+10,cy[i]+30), font, 0.5,(255,255,255),2,cv2.LINE_AA)
		object_shape ='square'
	elif shapes == 'pentagon':
		cv2.putText(calibrated_img,'pentagon',(cx[i]+10,cy[i]+30), font, 0.5,(255,255,255),2,cv2.LINE_AA)
		object_shape == 'pentagon'
	elif shapes == 'hexagon':
		cv2.putText(calibrated_img,'hexagone',(cx[i]+10,cy[i]+30), font, 0.5,(255,255,255),2,cv2.LINE_AA)
		object_shape == 'hexagone'
	else: 
		object_shape == 'trapez'
		cv2.putText(calibrated_img,'trapez',(cx[i]+10,cy[i]+30), font, 0.5,(255,255,255),2,cv2.LINE_AA)

	# determine the color of the recognized object
	if ((h1<=80 and h1>= 40) and (s1<=255 and s1>= 180) and (v1<=100 and s1>= 45)):
		#print('barva jezelena')
		color = 'green'
		cv2.putText(calibrated_img,'green',(cx[i]+10,cy[i]+10), font, 0.5,(255,255,255),2,cv2.LINE_AA)
	elif((h1<=180 and h1>= 85) and (s1<=240 and s1>= 130) and (v1<=70 and s1>= 20)):
		#print('barva je modra')
		color = 'blue'
		cv2.putText(calibrated_img,'blue',(cx[i]+10,cy[i]+10), font, 0.5,(255,255,255),2,cv2.LINE_AA)
	elif((h1<=20 and h1>= 0) and (s1<=240 and s1>= 200) and (v1<=200 and s1>= 150)):
		#print('barva je oranžna')
		color = 'red'
		cv2.putText(calibrated_img,'red',(cx[i]+10,cy[i]+10), font, 0.5,(255,255,255),2,cv2.LINE_AA)
	else:
		#print('barva ni prepoznana')
		color = 'color not defined'
		cv2.putText(calibrated_img,'Not identified',(cx[i]+10,cy[i]+10), font, 0.5,(255,255,255),2,cv2.LINE_AA)
	colors.append(color)

# cv2.imwrite('slika.png', calibrated_img)
print('The distances are: ',distance)
print('The vectors are: ', vectors)
cv2.imshow("Image", calibrated_img)
cv2.waitKey()