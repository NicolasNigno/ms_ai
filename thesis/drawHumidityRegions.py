import json
from PIL import Image, ImageDraw
import os
import cv2
from tqdm import tqdm
import numpy as np

labels_path = r'/home/nicolas/Documents/tesis'
images_path = r'/home/nicolas/Documents/tesis/imagenes_tesis'
new_images_path = r'/home/nicolas/Documents/tesis/regions'

data = open(labels_path + '/labels.json',)
labels = json.load(data)

os.chdir(images_path)
images = os.listdir()
os.chdir(new_images_path)

"""
def isHUmidity(img_processed):
    humidity = []
    img_processed_bool = img_processed == np.array([255, 0, 0])
    possible_match = np.argwhere(img_processed_bool)
    
    points = pd.DataFrame(data=possible_match, columns=["y", "x", 'z'])
    points.drop('z', axis=1, inplace=True)
    points.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    
    points_array = np.array(points)
    dimensions = points_array.shape[0]
    
    for point in range(dimensions):
        if np.all(img_processed_bool[points_array[point][0], points_array[point][1]]):
            humidity.append(1)
        else:
            humidity.append(0)
    
    points['humidity'] = humidity
    points = points.copy().loc[points['humidity'] == 1].reset_index(drop=True)
    
    return points
"""

for file in tqdm(images):
    try:
        path = images_path + '/' + file
        pic = cv2.imread(path,)
        
        mask = np.zeros((pic.shape[0], pic.shape[1])).astype(np.uint8)
        img = Image.fromarray(mask)
        draw = ImageDraw.Draw(img)
        
        for i in list(labels[file]['regions'].keys()):
            coordinates = [(x,y) for x,y in zip(labels[file]['regions'][i]['shape_attributes']['all_points_x'], 
                                                labels[file]['regions'][i]['shape_attributes']['all_points_y'])]
            
            draw.polygon(tuple(coordinates), fill=(1), outline=(1))
        
        img_processed = np.array(img).astype(np.uint8)
        name = file.split('.')[0]
        np.savetxt(name, img_processed)
        # imageio.imwrite(file, img_processed)
        # cv2.imwrite(file, img_processed)
    except Exception as e:
        print(e)