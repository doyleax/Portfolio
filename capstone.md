
# coding: utf-8

# # Capstone Project
# ## Image Classification of Dog Breeds
# 
# #### Problem Statement: 
# Using images of Boston Bulls, Beagles, and Boxers from ImageNet, predict Boomer's mix of breeds.

# *Import Necessary Libraries*

# In[1]:


from os import listdir
from keras.utils import np_utils
from imutils import paths

import keras as ks
from keras import backend as K
import cv2
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage import restoration


# ## Part 1: Get Images from ImageNet

# I found an amazing [repo](https://github.com/tzutalin/ImageNet_Utils) with CLI commands to download images and crop using the provided bounding boxes from ImageNet.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'git clone --recursive https://github.com/tzutalin/ImageNet_Utils.git\ncd ImageNet_Utils\n\n# Download images of the three breeds using their synset ID\n# Boston: n02096585\n# Beagle: n02088364\n# Boxer:  n02108089\n\n./downloadutils.py --downloadOriginalImages --wnid n02096585\n./downloadutils.py --downloadOriginalImages --wnid n02088364\n./downloadutils.py --downloadOriginalImages --wnid n02108089\n\n\n# Download Bounding Boxes\n\n./downloadutils.py --downloadBoundingBox --wnid n02096585\n./downloadutils.py --downloadBoundingBox --wnid n02088364\n./downloadutils.py --downloadBoundingBox --wnid n02108089\n\n\n# Crop Images by Bounding Box XML\n\n./bbox_helper.py --save_boundingbox --bxmldir n02096585/\n./bbox_helper.py --save_boundingbox --bxmldir n02088364/\n./bbox_helper.py --save_boundingbox --bxmldir n02108089/\n\n\n# Create Train, Test subfolders for each breed\n\nmkdir data/{train/{n02096585,n02088364,n02108089},test/{n02096585,n02088364,n02108089}}')


# In[139]:


# Determine the number of images to create a test set 
# consisting of 30% of each breed

test_n02096585 = len(listdir('data/n02096585_bounding_box_imgs'))*.3
test_n02088364 = len(listdir('data/n02088364_bounding_box_imgs'))*.3
test_n02108089 = len(listdir('data/n02108089_bounding_box_imgs'))*.3

#310, 393, 953 for train
print(test_n02096585,test_n02088364,test_n02108089)
#93, 118, 286 for test


# In[ ]:


get_ipython().run_cell_magic('bash', '', "\n# Randomly select 30% of each breed's images and move to test folder\n\nshuf -zen93 data/n02088364_bounding_box_imgs/*  | xargs -0 mv -t data/test/n02088364\nshuf -zen118 data/n02096585_bounding_box_imgs/*  | xargs -0 mv -t data/test/n02096585\nshuf -zen286 data/n02108089_bounding_box_imgs/*  | xargs -0 mv -t data/test/n02108089\n\n\n# Use remaining 70% as train folder\n\nmv data/n02088364_bounding_box_imgs data/train/n02088364\nmv data/n02096585_bounding_box_imgs data/train/n02096585\nmv data/n02108089_bounding_box_imgs data/train/n02108089")


# ## Part 2: Image Augmentation

# Import images
# 
# *adapted from [source](http://machinelearningmastery.com/image-augmentation-deep-learning-keras/)*

# In[22]:


# grab the list of images that we'll be describing
print("[INFO] describing images...")

## train ##
imagePaths = list(paths.list_images("data/train"))

# initialize the data matrix and labels list
trainData = []
trainLabels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # construct a feature vector raw pixel intensities, then update
    # the data matrix and labels list
    image = cv2.resize(image, (28,28))
    trainData.append(image)
    trainLabels.append(label)

    # show an update every 200 images
    if i > 0 and i % 200 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))
        
print("[INFO] image processing complete")


## test ##
imagePaths = list(paths.list_images("data/test"))

# initialize the data matrix and labels list
testData = []
testLabels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # construct a feature vector raw pixel intensities, then update
    # the data matrix and labels list
    image = cv2.resize(image, (28,28))
    testData.append(image)
    testLabels.append(label)

    # show an update every 200 images
    if i > 0 and i % 200 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))
        
print("[INFO] image processing complete")


# In[23]:


#take out unique file identifers so that theyre all set to synset id
trainLabels = [i[0:9] for i in trainLabels]
testLabels = [i[0:9] for i in testLabels]

# encode the labels, converting them from strings to integers
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, 3] 
trainData = np.array(trainData) / 255.0
trainLabels = np_utils.to_categorical(trainLabels, 3)

testData = np.array(testData) / 255.0
testLabels = np_utils.to_categorical(testLabels, 3)


# Perform Image Augmentation & Save Images
# 
# *adapted from [[source]](http://machinelearningmastery.com/image-augmentation-deep-learning-keras/)*

# In[2]:


from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
from os import listdir
from keras import backend as K
K.set_image_dim_ordering('tf')


# In[15]:


# Save augmented images to file

# define data preparation
batch_size = 16


def image_augmentation(method,prefix,msg):

    # perform the specified augmentation
    datagen = method
    datagen.fit(trainData)
    i = 0

    generator = datagen.flow_from_directory('data/train/', target_size=(100,100),
        shuffle=False, batch_size=batch_size,
        save_to_dir='data/train',save_prefix=prefix)
   
    for batch in generator:
        i += 1
        if i > 20: # save 20 images
            break  # otherwise the generator would loop indefinitely
    
    images = generator.filenames
    classes = generator.classes
    print("Class Indices:",generator.class_indices)
    print(msg)
    print(len(listdir('data/train/'))-3," total augmented images")
    
    return images,classes


# The following was actually run individually for each class due to issues in where images were saved and in how they were not named according to class.

# In[17]:


# Flip Images
flip_imgs, flip_classes = image_augmentation(ImageDataGenerator(horizontal_flip=True, vertical_flip=True),'flip',"Saved flipped images")
# Shift Images
shift_imgs, shift_classes =image_augmentation(ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2),'shift',"Saved shifted images")
# Rotate Images
rotated_imgs, rotated_classes = image_augmentation(ImageDataGenerator(rotation_range=90),'rotate',"Saved rotated images")
# Center, Normalize Images
norm_imgs, norm_classes = image_augmentation(ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True),'norm',"Saved normalized images")


# ### Re-import all images, along with augmented ones

# In[74]:


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


print("[INFO] describing images...")
## train ##
imagePaths = list(paths.list_images("data/train"))
# initialize the data matrix and labels list
trainData = []
trainLabels = []
# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    
    image = cv2.resize(image, (28,28))
    trainData.append(image)
    
    trainLabels.append(label)
    if i > 0 and i % 200 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))
        
print("[INFO] image processing complete")


## test ##
imagePaths = list(paths.list_images("data/test"))
testData = []
testLabels = []
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    image = cv2.resize(image, (28,28))
    testData.append(image)
    
    testLabels.append(label)
    if i > 0 and i % 200 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))
        
print("[INFO] image processing complete")

#take out unique file identifers so that theyre all set to synset id
trainLabels = [i[0:9] for i in trainLabels]
testFiles = testLabels
testLabels = [i[0:9] for i in testLabels]
testFolders = testLabels

# encode the labels, converting them from strings to integers
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, 3]
trainData = np.array(trainData) / 255.0
trainLabels = np_utils.to_categorical(trainLabels, 3)

testData = np.array(testData) / 255.0
testLabels = np_utils.to_categorical(testLabels, 3)

print("[INFO] Cell Finished.")


# ## Part 3: Modeling

# In[75]:


from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import SGD

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D


# Create Convolutional Neural Network
# 
# *pretty sure adapted from this [source](http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/*)

# In[87]:


model = Sequential()

depth = 3
height = 28
width = 28
classes = 3

model = Sequential()
# first set of CONV => RELU => POOL
model.add(Convolution2D(20, (5, 5), padding="same",
            input_shape=(height, width, depth),data_format="channels_last"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second set of CONV => RELU => POOL
model.add(Convolution2D(40, (5, 5), padding="same",
            input_shape=(height, width, depth),data_format="channels_last"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# third set of CONV => RELU => POOL
model.add(Convolution2D(60, (5, 5), padding="same",
            input_shape=(height, width, depth),data_format="channels_last"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# fourth set of CONV => RELU => POOL
model.add(Convolution2D(100, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))


# model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# softmax classifier
model.add(Dense(classes))
model.add(Activation("softmax"))


# In[88]:


# train, fit the model
K.set_image_dim_ordering('tf')
print("[INFO] compiling model...")

adam = ks.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss="categorical_crossentropy", optimizer=adam,
    metrics=["accuracy"])

model.fit(trainData, trainLabels, epochs=50, batch_size=128, verbose=1)


# In[89]:


# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
    batch_size=64, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
    accuracy * 100))


# Not horrible, better than the baseline of 1/3. I have plenty more iamges of the other breeds that I plan to add into the model and hope that this will help the score a bit.

# # Results

# In[90]:


from matplotlib import pyplot as plt
from PIL import Image
import skimage.io as io


# In[91]:


# get predictions for each test image

imagePaths = list(paths.list_images("data/test"))
test_preds = []

for i in range(0, len(imagePaths)):
    
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    preds = [round(i*100,5) for i in probs[0]]
    # record actual breed
    if testLabels[i][0]==1.0:
        actual='Beagle'
    elif testLabels[i][1]==1.0:
        actual='Boston'
    else:
        actual='Boxer'
    test_preds.append([imagePaths[i],preds[0],preds[1],preds[2],actual])
    
# Boston: n02096585 = 1
# Beagle: n02088364 = 0
# Boxer:  n02108089 = 2


# In[92]:


import pandas as pd

test_preds = pd.DataFrame(test_preds,columns=['Img','Beagle','Boston','Boxer','Actual'])

# determine breed with highest probability
test_preds['max']=''

for i in range(0,len(test_preds)):
    a,b,c = test_preds[['Beagle','Boston','Boxer']].ix[i]
    if (a>b) & (a>c):
        test_preds['max'].ix[i]='Beagle'
    elif b>c:
        test_preds['max'].ix[i]='Boston'
    else:
        test_preds['max'].ix[i]='Boxer'


# In[93]:


# count true and false positives

TP_Boxer = 0
TP_Beagle = 0
TP_Boston = 0

FP_Boxer = 0
FP_Beagle = 0
FP_Boston = 0

for i in range(0,len(test_preds)):
    if (test_preds['max'].ix[i]=='Boxer'):
        if (test_preds.Actual.ix[i]=='Boxer'):
            TP_Boxer+=1
        else:
            FP_Boxer+=1
    if (test_preds['max'].ix[i]=='Beagle'):
        if (test_preds.Actual.ix[i]=='Beagle'):
            TP_Beagle+=1
        else:
            FP_Beagle+=1
    if (test_preds['max'].ix[i]=='Boston'):
        if (test_preds.Actual.ix[i]=='Boston'):
            TP_Boston+=1
        else:
            FP_Boston+=1


# In[94]:


print('True, False Positives by Breed:')
print('Boxer', (TP_Boxer, FP_Boxer))
print('Beagle', (TP_Beagle, FP_Beagle))
print('Boston', (TP_Boston, FP_Boston))


# In[95]:


total = len(testData)
from sklearn.metrics import confusion_matrix, classification_report

true_labels = le.fit_transform(testFolders)
# predicted_labels = [0 if dog=='Beagle' else 1 if dog=='Boston' else 2 for dog in test_preds['max'].tolist()]

cm = confusion_matrix(true_labels, predicted_labels, labels=[0,1,2])
cm = pd.DataFrame(cm,columns=['Beagle','Boston','Boxer'],index=['Predicted Beagle','Predicted Boston','Predicted Boxer'])


print("Classification Report")
print(classification_report(true_labels, predicted_labels))
print()
print("Confusion Matrix")
cm


# ### Not far from what I expected after the test evaluation. Not great.

# # Final Test: Boomer

# In[96]:


## test ##
imagePaths = list(paths.list_images("Boomer"))
boomerData = []
boomerFiles = []
# testLabels = []
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28,28))
    boomerData.append(image)
    boomerFiles.append(imagePath)
    if i > 0 and i % 10 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

boomerData = np.array(boomerData) / 255.0
print("[INFO] image processing complete")


# In[122]:


get_ipython().magic('matplotlib inline')
boomer = []
# randomly select a few testing digits
# for i in np.random.choice(np.arange(0, len(boomerData)), size=(32,)):
i = 0
for boom in boomerData:
    # classify the digit
    probs = model.predict(boomerData[np.newaxis, i])    
    image = io.imread(boomerFiles[i])
    preds = [round(i*100,5) for i in probs[0]]
    boomer.append([boomerFiles[i],preds[0],preds[1],preds[2]])
    # show the image and prediction
    print("[INFO] Predicted: Beagle = {}, Boston = {}, Boxer = {}".format(preds[0],preds[1],preds[2]))
    # create a grid of 3x3 images
    plt.subplot(530 + 1 + i)
    plt.imshow(image)
# show the plot
    plt.show()
    i+=1
# Beagle: n02088364 = 0    
# Boston: n02096585 = 1
# Boxer:  n02108089 = 2


# In[123]:


import pandas as pd

boomer = pd.DataFrame(boomer,columns=['Img','Beagle','Boston','Boxer'])

# determine the breed with the highest probability
boomer['max']=''

for i in range(0,len(boomer)):
    a,b,c = boomer[['Beagle','Boston','Boxer']].ix[i]
    if (a>b) & (a>c):
        boomer['max'].ix[i]='Beagle'
    elif b>c:
        boomer['max'].ix[i]='Boston'
    else:
        boomer['max'].ix[i]='Boxer'


# In[124]:


boomer['max'].groupby(boomer['max']).count()


# ### Knowing that the model is much better trained on Boxers, it's not suprising that most images are classified as Boxer.