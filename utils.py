import argparse
import h5py
import numpy as np
import datetime
import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt

def folder_to_h5(label, path):
    image_names = os.listdir(path)
    dataset_length = len(image_names)
    
    f = h5py.File('data.h5','a')
    output = f.create_dataset(label, (dataset_length, 402, 536, 3), dtype=np.uint8)
    for i in range(len(image_names)):
        image = image_names[i]
        npyImg = Image.open(path + "/" + image)
        npyImg = np.asarray(npyImg).reshape((402, 536, 3))
        output[i] = npyImg



def display(path):
    f = h5py.File(path, 'r')
    keys = [x.encode('UTF8') for x in f.keys()]
    for key in keys:
        data = f[key]
        length = data.shape[0]
        for i in range(length):
            img = data[i]
            plt.imshow(img)
            plt.show()

def getOnehotArray(key, keys):
    y = np.zeros((len(keys)))
    y[keys.index(key)] = 1
    return y

def convertOnehot(path):
    f = h5py.File(path, 'r')
    keys = [x.encode('UTF8') for x in f.keys()]
    total_length = 0
    for key in keys:
        data = f[key]
        length = data.shape[0]
        total_length += length

    f1 = h5py.File('pizza.h5','a')
    x = f1.create_dataset('x_dataset', (total_length, 402, 536, 3), dtype=np.uint8)
    y = f1.create_dataset('y_dataset', (total_length, 11))

    instance = 0
    for key in keys:
        data = f[key]
        length = data.shape[0]
        for i in range(length):
            img = data[i]
            print(img.shape)
            print(getOnehotArray(key, keys).shape)
            x[instance] = img
            y[instance] = getOnehotArray(key, keys)
            instance += 1

def display_onehot(path):
    f = h5py.File(path, 'r')
    x = f['x_dataset']
    y = f['y_dataset']
    length = x.shape[0]
    for i in range(length):
        image = x[i]
        label = y[i]

        plt.imshow(image)
        plt.show()
        print(label)

display_onehot('pizza.h5')
#convertOnehot('data.h5')
# display('data.h5')
# if __name__ == "__main__":
#     data = "dataset/" + "FPID_smallerPizza"
#     path = os.getcwd() + "/" + data
#     files = os.listdir(path)
#     for file in files:
#         print "Converting", file
#         folder_to_h5(file, path + "/" + file)

    
