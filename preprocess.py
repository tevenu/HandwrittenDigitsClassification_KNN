import struct
import os
import numpy as np
from PIL import Image



def read_labels(filename):
    with open(filename, 'rb') as f:
        _ = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        labels = [struct.unpack('B', f.read(1))[0] for _ in range(num_labels)]
        return labels


def read_imgs(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


for i, img in enumerate(read_imgs(r"D:\大学文件\大三下\人工智能导论\Mnist\train-images.idx3-ubyte")):
    im = Image.fromarray(img)
    im.save(r'D:\大学文件\大三下\人工智能导论\Mnist\img_train/{}.png'.format(i))

for i, img in enumerate(read_imgs(r"D:\大学文件\大三下\人工智能导论\Mnist\test-images.idx3-ubyte")):
    im = Image.fromarray(img)
    im.save(r'D:\大学文件\大三下\人工智能导论\Mnist\img_test/{}.png'.format(i))

with open(r'D:\大学文件\大三下\人工智能导论\Mnist\labels_train.txt', 'w') as f:
    for label in read_labels(r"D:\大学文件\大三下\人工智能导论\Mnist\train-labels.idx1-ubyte"):
        f.write(str(label))
        f.write('\n')
    f.close()

with open(r'D:\大学文件\大三下\人工智能导论\Mnist\labels_test.txt', 'w') as f:
    for label in read_labels(r"D:\大学文件\大三下\人工智能导论\Mnist\test-labels.idx1-ubyte"):
        f.write(str(label))
        f.write('\n')
    f.close()
