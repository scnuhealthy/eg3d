import json
import numpy as np
import os
from PIL import Image
import cv2
input_json = 'transforms_train.json'
output_json = 'dataset.json'
img_size=128
input_dir = 'train'
output_dir = '00000'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
for file in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file)
    img = cv2.imread(file_path)
    img = cv2.resize(img,(img_size,img_size))
    out_path = os.path.join(output_dir, file)
    img = 255 - img
    cv2.imwrite(out_path, img)
    #img = Image.open(file_path)
    #img = img.resize((img_size,img_size), Image.LANCZOS)
    #img = img.convert('RGB')
    #img.save(out_path)

data = json.load(open(input_json))
camera_angle = data['camera_angle_x']
print(np.tan(0.5*camera_angle))
focal = 0.5*800/np.tan(0.5*camera_angle)
focal *= (img_size/800)
print(focal,img_size)
intrinsic_matrix = [focal,0,0.5,0,focal,0.5,0,0,1]

frames = data['frames']
labels = []
for frame in frames:
    frame_name = output_dir + '/' + frame['file_path'].split('/')[-1] + '.png'
    extrinsic_matrix = []
    for i in range(4):
        for j in range(4):
            extrinsic_matrix.append(frame['transform_matrix'][i][j])
    label = extrinsic_matrix + intrinsic_matrix 
    label = [frame_name, label]
    labels.append(label)
json.dump({'labels':labels}, open('dataset.json','w'),indent=2)