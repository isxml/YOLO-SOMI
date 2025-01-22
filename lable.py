import os
import numpy as np
import cv2
from PIL.ImagePath import Path

from utils.general import increment_path
from utils.plots import Colors


img_folder = "./datasets/image/"
img_list = os.listdir(img_folder)
img_list.sort()

label_folder = "./datasets/VisDrone/VisDrone2019-DET-val/labels/"
label_list = os.listdir(label_folder)
label_list.sort()

path = os.getcwd()
output_folder = str("./runs/detect/output")
save_dir = increment_path(output_folder)

colormap = Colors()  # create instance for 'from utils.plots import colors'

def xywh2xyxy(x, w1, h1, img):
    label, x, y, w, h = x
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[1], 2)
    return img


if __name__ == '__main__':
    for filename in img_list:
        image_path = os.path.join(img_folder, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_folder, txt_filename)
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        with open(label_path, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
        for x in lb:
            img = xywh2xyxy(x, w, h, img)
        cv2.imwrite(output_folder + '/' + '{}.png'.format(image_path.split('/')[-1][:-4]), img)
