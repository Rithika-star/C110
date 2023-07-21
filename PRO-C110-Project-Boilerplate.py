# To Capture Frame
import cv2


# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf

model=tf.keras.models.load_model('C:/Users/admin/Desktop/Python/PRO-C110-Project-Boilerplate-main/keras_model.h5')
# Attaching Cam indexed as 0, with the application software
cam= cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = cam.read()

	# if we were sucessfully able to read the frame


		# Flip the frame
	frame = np.flip(frame , 1)
		
		
		
		#resize the frame
	img=cv2.resize(frame,(224,224))
   
		# expand the dimensions
	testimg=np.array(img,dtype=np.float32)
    testimg=np.expand_dims(testimg,axis=1)
   
		# normalize it before feeding to the model
	nimage=testimg/255.0
    
		# get predictions from the model
	prediction=model.predict(nimage)
    #print(prediction)
		
		
		# displaying the frames captured
	cv2.imshow('feed' , frame)

		# waiting for 1ms
	code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
	if code == 32:
		break

# release the camera from the application software
cam.release()

# close the open window
cv2.destroyAllWindows()
