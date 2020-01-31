#Importing libraries
#To run this file, execute following command
#python train.py --train datasets/casting_data/train --test datasets/casting_data/test 
#--batch_size 32 --model output/VGG16.hdf5 --figpath output/VGG16.png
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.applications import VGG16
from keras.layers.core import Flatten, Dense, Dropout
import numpy as np
import argparse
import os


#command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", required=True,
	help="path training database")
ap.add_argument("-o", "--test", required=True,
	help="path to testing database")
ap.add_argument("-b", "--batch_size", type=int, default=32,
	help="path to testing database")
ap.add_argument("-m", "--model", required=True,
	help="path to save model")
ap.add_argument("-f", "--figpath", required=True,
	help="path to save training/testing loss/accuracy")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

print("[Info] loading imagenet weights...")
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(1024, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation='sigmoid')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
	layer.trainable = False

#image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = args["batch_size"]
print("[INFO] loading images...")
train_data_dir = args["train"]     #directory of training data
test_data_dir = args["test"]      #directory of test data

training_set = train_datagen.flow_from_directory(train_data_dir, 
	target_size=(224, 224),
	batch_size=batch_size, 
	class_mode='binary')
test_set = test_datagen.flow_from_directory(test_data_dir, 
	target_size=(224, 224),
	batch_size=batch_size,
	class_mode='binary')

print(training_set.class_indices)

print("[INFO] compiling model...")
opt = Adam(lr=0.001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit_generator(training_set,
	steps_per_epoch=training_set.samples//batch_size,
	validation_data=test_set,
	epochs=5,
	validation_steps=test_set.samples//batch_size)

print("[Info] serializing network...")
model.save(args["model"])

print("[Info] visualising model...")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 5), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 5), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 5), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 5), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
plt.savefig(args["figpath"])