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
        img_pca = getPCA(pic, pic.shape[0], pic.shape[1])
        filename = img.split('/')[-1]
        cv2.imwrite(filename, img_pca) 
    except:
        pass

def getPCA(imgage, rows, columns):
    pca = PCA(n_components=0.95)
    pic_reshape = imgage.reshape(-1, 3)
    
    pic_transformed = pca.fit_transform(pic_reshape)
    pic_inverted = pca.inverse_transform(pic_transformed)
    
    pic_inverted_reshape = pic_inverted.reshape(rows, columns, 3).astype(np.uint8)
    
    return pic_inverted_reshape

for image in tqdm(images_full_path):
    transformImage(image)