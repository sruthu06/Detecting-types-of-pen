#importing modules
import cv2   
import numpy as np

#capturing video through webcam
cap=cv2.VideoCapture(0)

while(1):
	_, img = cap.read()
	    
	#converting frame to HSV
	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

	#Range of red color
	red_lower=np.array([136,87,111],np.uint8)
	red_upper=np.array([180,255,255],np.uint8)

	#Range of Blue color
	blue_lower=np.array([99,115,150],np.uint8)
	blue_upper=np.array([110,255,255],np.uint8)
	
	#Range of black color
	black_lower=np.array([0,0,0],np.uint8)
	black_upper=np.array([180,255,30],np.uint8)

	#finding the range of red,blue and black color in the image
	red=cv2.inRange(hsv, red_lower, red_upper)
	blue=cv2.inRange(hsv,blue_lower,blue_upper)
	black=cv2.inRange(hsv,black_lower,black_upper)
	
	#Morphological transformation, Dilation  	
	kernal = np.ones((5 ,5), "uint8")

        red=cv2.dilate(red, kernal)
	res=cv2.bitwise_and(img, img, mask = red)

	blue=cv2.dilate(blue,kernal)
	res1=cv2.bitwise_and(img, img, mask = blue)

	black=cv2.dilate(black,kernal)
	res2=cv2.bitwise_and(img, img, mask = black)    


	#Tracking the Red Color
	(_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>300):
			
			x,y,w,h = cv2.boundingRect(contour)	
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
			cv2.putText(img,"RED pen",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
			
	#Tracking the Blue Color
	(_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>300):
			x,y,w,h = cv2.boundingRect(contour)	
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.putText(img,"Blue pen",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))

	#Tracking the black Color
	(_,contours,hierarchy)=cv2.findContours(black,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>300):
			x,y,w,h = cv2.boundingRect(contour)	
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
			cv2.putText(img,"black  pen",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0))  
            
           
    	
    	cv2.imshow("Pen Tracking",img)
    	
    	if cv2.waitKey(10) & 0xFF == ord('q'):
    		cap.release()
    		cv2.destroyAllWindows()
    		break  
          

    
