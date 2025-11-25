import torch
import numpy as np
import PIL
import cv2
import random

# TODO: implementation transformations for task3;
# You cannot directly use them from pytorch, but you are free to use functions from cv2 and PIL
class Padding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, img, **kwargs):
        
        pad = self.padding
        cv2_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img_pad = cv2.copyMakeBorder(cv2_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, (0,0,0))
        img = PIL.Image.fromarray(cv2.cvtColor(img_pad,cv2.COLOR_BGR2RGB))
        return img


class RandomCrop(object):
    def __init__(self, size):
        self.size = size


    def __call__(self, img, **kwargs):
        
        w,h = img.size
        tw = self.size
        th = self.size
        if w == tw and h == th:
            img = img.crop((0,0,w,h))

        i = random.randint(0,w-tw)
        j = random.randint(0,h-th)
        img = img.crop((i,j,i+tw,j+th))


        return img

class Cutout(object):
    def __init__(self,n_patch,size_patch):
        self.n_patch = n_patch
        self.size_patch = size_patch

    def __call__(self, img, **kwargs):
        w,h = img.size
        cv2_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        size_patch = self.size_patch


        for n in range(self.n_patch):
            y = random.randint(0,h)
            x = random.randint(0,w)
            y1 = np.clip(y - size_patch // 2, 0, h)
            y2 = np.clip(y + size_patch // 2, 0, h)
            x1 = np.clip(x - size_patch // 2, 0, w)
            x2 = np.clip(x + size_patch // 2, 0, w)
            cv2_img[x1:x2,y1:y2] = 0

        img = PIL.Image.fromarray(cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB))

        return img




class RandomFlip(object):
    def __init__(self, p=0.5):
        
        self.p = p
 
    def __call__(self, img, **kwargs):
        
        cv2_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        
        m = self.p
        list_flip = [0,-1,1]
        k = random.random()

        if k < m:
            flip_code = random.choice(list_flip)
            cv2_img_flip = cv2.flip(cv2_img,flip_code)
            img = PIL.Image.fromarray(cv2.cvtColor(cv2_img_flip,cv2.COLOR_BGR2RGB))


        return img
