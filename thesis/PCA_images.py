import cv2
import numpy as np
from sklearn.decomposition import PCA
import os
from tqdm import tqdm

pca_save_path = '/home/nicolas/Documents/tesis/pca_imagenes/'
images_path = '/home/nicolas/Documents/tesis/imagenes_tesis/'
os.chdir(pca_save_path)

images = os.listdir(images_path)
images_full_path = []

for path in images:
    new_path = images_path + path
    images_full_path.append(new_path)

def transformImage(img):
    try:
        pic = cv2.imread(img, cv2.IMREAD_COLOR)
        blue, green, red = cv2.split(pic)
        img_compressed = getPCA(blue, green, red)
        filename = img.split('/')[-1]
        cv2.imwrite(filename, img_compressed) 
    except:
        pass

def getPCA(blue, green, red):
    pca = PCA(n_components=0.95)
    
    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)

    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)

    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)
    
    return (np.dstack((blue_inverted, green_inverted, red_inverted))).astype(np.uint8)

for image in tqdm(images_full_path):
    transformImage(image)