import numpy as np
from PIL import Image


def img2vec(filename):
    # 打开图片并转换为灰度模式
    img = Image.open(filename).convert('L')
    # 把图片转换为numpy数组
    img_arr = np.array(img)
    # 把数组变成一个行向量
    img_vec = img_arr.reshape(1, -1)
    return img_vec


def save_train_img():
    file_root = r"D:\大学文件\大三下\人工智能导论\Mnist\img_train"
    train_vectors = np.empty(shape=[60000, 784])
    for i in range(0, 60000):
        file_path = file_root + '\\' + '%s.png' % i
        train_vectors[i] = img2vec(file_path)
        print(i)
    # 使用savetxt()函数来把矩阵存到一个文本文件中
    np.savetxt('train_vectors.txt', train_vectors)
    print("存储成功")


def main():
    save_train_img()


if __name__ == '__main__':
    main()
