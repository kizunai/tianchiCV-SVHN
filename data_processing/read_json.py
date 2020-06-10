import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser

#读取配置文件
cp = ConfigParser()
cp.read('../config.cfg')
data_location = cp.get('location', 'Data_location')
train_json = json.load(open(data_location + r'\train\mchar_train.json'))


# 数据标注处理
def parse_json(d):
    arr = np.array([
        d['top'], d['height'], d['left'],  d['width'], d['label']
    ])
    arr = arr.astype(int)
    return arr


# location = data_location + r"\train\mchar_train\000000.png"
img = cv2.imread(data_location + r"\train\mchar_train\000000.png")
arr = parse_json(train_json['000000.png'])

plt.figure(figsize=(10, 10))
plt.subplot(1, arr.shape[1]+1, 1)
plt.imshow(img)
plt.xticks([]); plt.yticks([]) # 坐标刻度，为空隐藏刻度

for idx in range(arr.shape[1]):
    plt.subplot(1, arr.shape[1]+1, idx+2)
    plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],arr[2, idx]:arr[2, idx]+arr[3, idx]])
    plt.title(arr[4, idx])
    plt.xticks([]); plt.yticks([])

plt.show()
