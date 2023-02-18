from PIL import Image, ImageDraw, ImageOps 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from tqdm import tqdm
import random 
import os
from os import listdir
import json  

# I define the width and height of images, since height and width is different for each image, we need to predefine the desired shape
H, W = 224, 224
n_sample =  5000
plate_H = 128
plate_W = 32


# compose is a combination of random transforms to augment data. This transformations are Perspective, scaling, rotation, and shear.
composed = T.Compose([
    T.Resize(size= (plate_W, plate_H )),
     T.Pad(padding=((H - plate_H)//2, (W - plate_W)//2)),
     T.RandomPerspective(distortion_scale=0.4, p=0.5),
     T.RandomAffine(degrees=(-20, 20), translate=(0.3, 0.3), scale=(0.5, 0.8),  shear =(-20,20,-20,20) ) 
 ])

def find_corners(img, eps=1e-3):
    """
    Given the transformed plate image, identify the four corners of the plate
    """
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(img,1,255,0)
    orignal_thresh = thresh.copy()
    thresh = cv2.medianBlur(thresh,3)
    thresh= cv2.blur(thresh, (5,5))
    
    corners = cv2.goodFeaturesToTrack(thresh, 4, eps, 2, 5)
    corners = np.int0(corners)
    
    return corners, orignal_thresh

def mix_images(plate, back_ground , thresh):
    """
    Given the background image and thr plate , adding the two images together
    """
    back_ground = T.Resize(size=(W, H))(back_ground)
    back_ground = np.asarray(back_ground)
    
    thresh = cv2.bitwise_not(thresh)
    thresh[thresh>0]=1
    thresh = np.array([thresh,thresh,thresh]).transpose(1,2,0)
    
    mixed_image = back_ground  * thresh + plate
    return T.ToPILImage()(mixed_image)

    
    
    
    #Loading backgrounds and plates 
Back_grounds=[]
Plate_imgs=[]

back_ground_folder_dir = "backgrounds"
for images in os.listdir(back_ground_folder_dir):
    back_ground = Image.open('backgrounds/'+images)
    Back_grounds.append(back_ground)

    
plate_folder_dir = "plates" 
for images in os.listdir(plate_folder_dir):
    plate_img = Image.open('plates/'+images)
    Plate_imgs.append(plate_img)    
    
    
# Generating n_samples = 5000 training samples

corner_cordinates = {}   
for i in tqdm(range(n_sample)):
    back_ground = Back_grounds[random.randint(0,29)]
    plate_choice = random.randint(0,99)
    plate_img = Plate_imgs[plate_choice]

    sample = np.asarray(plate_img)
    sample = cv2.add(sample,10)

    plate_img = T.ToPILImage()(sample)
    img = composed(plate_img)
    img = np.asarray(img)

    corners, thresh= find_corners(img)
    img = mix_images(img, back_ground,thresh)
    if (len(corners) == 4):
        img.save('data/' + str(i) + '.jpg')
        corner_cordinates[str(i)] = corners.tolist()

save_file = open("coordinates.json", "w")  
json.dump(corner_cordinates, save_file)  
save_file.close()  