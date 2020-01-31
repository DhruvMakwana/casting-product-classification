#importing libraries
#To run this file insert following command in command propmt
#python predict.py --model output/VGG16.hdf5 --image input/def_front/test1.jpeg --output 
#output/def_front/test1.jpeg
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image
import numpy as np 
import argparse
import cv2
import os 

#command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, 
	help="path to input model")
ap.add_argument("-i", "--image", required=True, 
	help="path to test image")
ap.add_argument("-o", "--output", required=True,
	help="path to output image")
args = vars(ap.parse_args())

print("[Info] loading pre-trained network...")
model = load_model(args["model"])
imagepath = args["image"]

print("[Info] loading image...")
img = cv2.imread(imagepath)
img = cv2.resize(img, (224,224))
orig = img.copy()
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img/255

print("[Info] predicting output")
prediction = model.predict(img)
if (prediction<0.5):
     print("def_front")
     cv2.putText(orig, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
     cv2.imwrite(args["output"], orig)
else:
     print("ok_front")
     cv2.putText(orig, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
     cv2.imwrite(args["output"], orig)
cv2.imshow("Image", orig)
cv2.waitKey(0)