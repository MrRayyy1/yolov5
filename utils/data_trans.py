#coding=utf-8
"""
1. Image_flip:翻转图片
2. Image_traslation:平移图片
3. Image_rotate:旋转图片
4. Image_noise:添加噪声
"""
import os
import cv2
import numpy as np
from random import choice
import random
import matplotlib.pyplot as plt

def Image_flip(img):
    """
    :param img:原始图片矩阵
    :return: 0-垂直； 1-水平； -1-垂直&水平
    """
    if img is None:
        return
    paras = [0, 1, -1]
    img_new = cv2.flip(img, choice(paras))
    return img_new

def Image_traslation(img):
    """
    :param img: 原始图片矩阵
    :return: [1, 0, 100]-宽右移100像素； [0, 1, 100]-高下移100像素
    """
    paras_wide = [[1, 0, 100], [1, 0, -100]]
    paras_height = [[0, 1, 100], [0, 1, -100]]
    rows, cols = img.shape[:2]
    img_shift = np.float32([choice(paras_wide), choice(paras_height)])
    border_value = tuple(int(x) for x in choice(choice(img)))
    img_new = cv2.warpAffine(img, img_shift, (cols, rows), borderValue=border_value)
    return img_new

def Image_rotate(img):
    """
    :param img:原始图片矩阵
    :return:旋转中心，旋转角度，缩放比例
    """
    rows, cols = img.shape[:2]
    rotate_core = (cols/2, rows/2)
    rotate_angle = [60, -60, 45, -45, 90, -90, 210, 240, -210, -240]
    paras = cv2.getRotationMatrix2D(rotate_core, choice(rotate_angle), 1)
    border_value = tuple(int(x) for x in choice(choice(img)))
    img_new = cv2.warpAffine(img, paras, (cols, rows), borderValue=border_value)
    return img_new

def Image_noise(img):
    """
    :param img:原始图片矩阵
    :return: 0-高斯噪声，1-椒盐噪声
    """
    paras = [0, 1]
    gaussian_class = choice(paras)
    noise_ratio = [0.05, 0.06, 0.08]
    if gaussian_class == 1:
        output = np.zeros(img.shape, np.uint8)
        prob = choice(noise_ratio)
        thres = 1 - prob
        #print('prob', prob)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output
    else:
        mean = 0
        var=choice([0.001, 0.002, 0.003])
        #print('var', var)
        img = np.array(img/255, dtype=float)
        noise = np.random.normal(mean, var**0.5, img.shape)
        out = img + noise
        if out.min() < 0:
            low_clip = -1
        else:
            low_clip = 0
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        return out

def image_crop(img, min_ratio=0.5, max_ratio = 1.0):
    h, w = img.shape[:2]

    ratio = random.random()

    scale = min_ratio + ratio * (max_ratio - min_ratio)

    new_h = int(h * scale)
    new_w = int(w * scale)

    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    image = img[y:y + new_h, x:x + new_w, :]

    return image



if __name__ == "__main__":
    """
    path_read: 读取原始数据集图片的位置;
    path_write：图片扩增后存放的位置；
    picture_size：图片之后存储的尺寸;
    enhance_hum: 需要通过扩增手段增加的图片数量
    """
    path_read = r"C:\Users\lenovo\Desktop\train"
    path_write = r"C:\Users\lenovo\Desktop\trans"
    enhance_num = 20
    image_list = [x for x in os.listdir(path_read)]
    img_dict = dict()
    for img in image_list:
        img_dict[img] = 0
    existed_img = len(image_list)
    while enhance_num > 0:
        img = choice(image_list)
        img_dict[img] += 1
        path = os.path.join(path_read, img)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        algorithm = [1, 2, 3, 4, 5]
        #algorithm = [4]
        random_process = choice(algorithm)
        if random_process == 1:
            image = Image_flip(image)
        elif random_process == 2:
            image = Image_traslation(image)
        elif random_process == 3:
            image = Image_rotate(image)
        elif random_process == 4:
            image = image_crop(image)
        else:
            image = Image_noise(image)
        #image_dir = path_write+str(enhance_num+existed_img-1).zfill(5)+'.jpeg'
        img_name = img.strip(".jpeg")
        image_dir = os.path.join(path_write, img_name+"_"+str(img_dict[img])+".jpeg")
        cv2.imwrite(image_dir,image)
        enhance_num -= 1
