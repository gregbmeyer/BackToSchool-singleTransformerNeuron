import numpy as np
from PIL import Image, ImageOps
import os
directory = os.path.realpath(os.path.dirname(__file__))
image_Label_list = []
image_X_list = []
def convertImageStdBW.py(imgT):
  im1 = ImageOps.grayscale(imgT)
  newsize = (50,50)
  im2 = im1.resize(newsize)
  img2Array = np.asarray(im2)
  img2Array = (img2Array-img2Array(min))/(img2Array.max()-img2Array.min())
  return img2Array

def convertImgToLabeledArray(labely):
  imagesPath = directory + '/' + labely
  print('image folder path: ', imagesPath)
  for imagesL in os.listdir(imagesPath):
    imagesL = Image.open(imagesPath + '/' + imagesL)
    image_Label_list.append(labely)
    image_X_list.append(im1)
  print('Number of images: ', len(image_X_list))
  return image_X_list, image_Label_list

print('Get image inputs for training the neuron.\n')
imageLabel = input('What is the folder Label for the first type of images?\n')
imListX1, imLblList1 = convertImgToLabeledArray(imageLabel)
imageLabelt = input('What is the folder Label for the second type of images?\n')
imListX2, imLblList2 = convertImgToLabeledArray(imageLabelt)
imListX = imListX1.append(imListX2)
imLblList = imLblList1.append(imLblList2)
neuronImageArrays = np.asarray(imListX)
np.save('neuronImageArrays.npy')
neuronImageLabels = np.asarray(imLblList)
np.save('neuronImageLabels.npy')
