import json
from PIL import Image, ImageDraw
import os
import cv2
from tqdm import tqdm

labels_path = r'/home/nicolas/Documents/tesis'
images_path = r'/home/nicolas/Documents/tesis/imagenes_tesis'
new_images_path = r'/home/nicolas/Documents/tesis/imagesWithRegions'

data = open(labels_path + '/labels.json',)
labels = json.load(data)

os.chdir(images_path)
images = os.listdir()

os.chdir(new_images_path)

for file in tqdm(images):
    try:
        path = images_path + '/' + file
        pic = cv2.imread(path,)
        img = Image.fromarray(pic)
        draw = ImageDraw.Draw(img)
        
        for i in list(labels[file]['regions'].keys()):
            coordinates = [(x,y) for x,y in zip(labels[file]['regions'][i]['shape_attributes']['all_points_x'], 
                                                labels[file]['regions'][i]['shape_attributes']['all_points_y'])]
            
            draw.polygon(tuple(coordinates), fill=(255, 0, 0), outline=(255, 0, 0))
        
        img.save(file)
    except:
        print(path)