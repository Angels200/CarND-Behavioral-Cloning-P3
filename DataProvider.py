
# coding: utf-8

# In[4]:

import errno
import json
import os
import datetime
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli

# Some useful constants
DRIVING_LOG_FILE = './data/driving_log.csv'
IMG_PATH = './data/'
STEERING_COEFFICIENT = 0.229

class DataProvider():
    
    def __init__(self, hxrate=.35,lxrate=.1,nshape=(64,64),shrange=200,flrate=.5,rotrate=15,shrate=.9):
        self.hxrate = hxrate
        self.lxrate = lxrate
        self.nshape = nshape
        self.shrange = shrange
        self.flrate = flrate
        self.rotrate = rotrate
        self.shrate = shrate
    
    def save(self, image, path):
        ts = time.time()
        fname = str(datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S'))+'.jpg'
        path = path +'/'+fname
        cv2.imwrite(path,image)
    
    def crop(self,image):
        hx = int(np.ceil(image.shape[0] * self.hxrate))
        lx = image.shape[0] - int(np.ceil(image.shape[0] * self.lxrate))
        return image[hx:lx, :]


    def resize(self,image):
        return scipy.misc.imresize(image, self.nshape)


    def rnflip(self,image, steering_angle):
        probability = bernoulli.rvs(self.flrate)
        if probability:
            return np.fliplr(image), -1 * steering_angle
        else:
            return image, steering_angle


    def rngamma(self,image):
        gamma = np.random.uniform(0.4, 1.5)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)


    def rnshear(self,image, steering_angle):
        """
        Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
        """
        rows, cols, ch = image.shape
        dx = np.random.randint(-self.shrange, self.shrange + 1)
        random_point = [cols / 2 + dx, rows / 2]
        pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
        pts2 = np.float32([[0, rows], [cols, rows], random_point])
        dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
        steering_angle += dsteering

        return image, steering_angle


    def rnrotation(self,image, steering_angle):
        rad = (np.pi / 180.0) * np.random.uniform(-self.rotrate, self.rotrate + 1)
        return rotate(image, angle, reshape=False), steering_angle + (-1) * rad


    def minmax(self,data, a=-0.5, b=0.5):
        return a + (b - a) * ((data - np.min(data)) / (np.max(data) - np.min(data)))


    def getnewimage(self,image, steering_angle):
        prob = bernoulli.rvs(self.shrate)
        if prob == 1:
            image, steering_angle = self.rnshear(image, steering_angle)
        image = self.crop(image)
        image, steering_angle = self.rnflip(image, steering_angle)
        image = self.rngamma(image)
        image = self.resize(image)
        return image, steering_angle


    def loadata(self,batch_size=64):
        data = pd.read_csv(DRIVING_LOG_FILE)
        nbsamples = len(data)
        indices = np.random.randint(0, nbsamples, batch_size)

        samples = []
        for index in indices:
            image = np.random.randint(0, 3)
            if image == 0:
                img = data.iloc[index]['left'].strip()
                angle = data.iloc[index]['steering'] + STEERING_COEFFICIENT
                samples.append((img, angle))

            elif image == 1:
                img = data.iloc[index]['center'].strip()
                angle = data.iloc[index]['steering']
                samples.append((img, angle))
            else:
                img = data.iloc[index]['right'].strip()
                angle = data.iloc[index]['steering'] - STEERING_COEFFICIENT
                samples.append((img, angle))

        return samples


    def getbatch(self,batch_size=64):
        while True:
            X_batch = []
            y_batch = []
            samples = self.loadata(batch_size)
            for img_file, angle in samples:
                raw_image = plt.imread(IMG_PATH + img_file)
                raw_angle = angle
                new_image, new_angle = self.getnewimage(raw_image, raw_angle)
                X_batch.append(new_image)
                y_batch.append(new_angle)
            assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'
            yield np.array(X_batch), np.array(y_batch)


# In[ ]:



