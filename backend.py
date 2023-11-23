BACKEND PROGRAM


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator , img_to_array
, load_img
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.losses import categorical_crossentropy
Building our Model To train the data
# Working with pre trained model
base_model = MobileNet( input_shape=(224,224,3), include_top= False )
for layer in base_model.layers:
 layer.trainable = False
x = Flatten()(base_model.output)
x = Dense(units=7 , activation='softmax' )(x)
# creating our model.
model = Model(base_model.input, x)
model.compile(optimizer='adam', loss= categorical_crossentropy , metric
s=['accuracy'] )
visualizaing the data that is fed to train data gen
train_datagen = ImageDataGenerator(
 zoom_range = 0.2,
 shear_range = 0.2,
 horizontal_flip=True,
 rescale = 1./255
)
train_data = train_datagen.flow_from_directory(directory= "/content/tra
in",
 target_size=(224,224),
 batch_size=32,
 )
train_data.class_indices
val_datagen = ImageDataGenerator(rescale = 1./255 )
val_data = val_datagen.flow_from_directory(directory= "/content/test",
 target_size=(224,224),
 batch_size=32,
 )
# to visualize the images in the traing data denerator
t_img , label = train_data.next()
#----------------------------------------------------------------------
-------
# function when called will prot the images
def plotImages(img_arr, label):
 """
 input :- images array
 output :- plots the images
 """
 count = 0
 for im, l in zip(img_arr,label) :
 plt.imshow(im)
 plt.title(im.shape)
 plt.axis = False
 plt.show()
 
 count += 1
 if count == 10:
 break
#----------------------------------------------------------------------
-------
# function call to plot the images
plotImages(t_img, label)
having early stopping and model check point
# Loading the best fit model
from keras.models import load_model
model = load_model("/content/best_model.h5")
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'] , c = "red")
plt.title("acc vs v-acc")
plt.show()
plt.plot(h['loss'])
plt.plot(h['val_loss'] , c = "red")
plt.title("loss vs v-loss")
plt.show()
# path for the image to see if it predics correct class
path = "/content/test/angry/PrivateTest_1054527.jpg"
img = load_img(path, target_size=(224,224) )
i = img_to_array(img)/255
input_arr = np.array([i])
input_arr.shape
pred = np.argmax(model.predict(input_arr))
print(f" the image is of {op[pred]}")
# to display the image 
plt.imshow(input_arr[0])
plt.title("input image")
plt.show()
