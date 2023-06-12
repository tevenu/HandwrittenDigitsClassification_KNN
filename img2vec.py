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


# 给图像数据添加椒盐噪声
def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    image_array = np.array(image)
    height, width = image_array.shape
    salt_noise = np.random.rand(height, width) < salt_prob
    pepper_noise = np.random.rand(height, width) < pepper_prob
    image_array[salt_noise] = 255
    image_array[pepper_noise] = 0
    return Image.fromarray(image_array)


# 制作投毒数据集
def poison_train_img(poison_ratio):
    file_root = r"D:\大学文件\大三下\人工智能导论\Mnist\img_train"
    train_vectors = np.empty(shape=[60000, 784])
    for i in range(0, 60000):
        file_path = file_root + '\\' + '%s.png' % i
        image = Image.open(file_path).convert('L')
        # 数据投毒
        noisy_image = add_salt_pepper_noise(image, salt_prob=poison_ratio, pepper_prob=poison_ratio)
        # 把图片转换为numpy数组
        img_arr = np.array(noisy_image)
        # 把数组变成一个行向量
        img_vec = img_arr.reshape(1, -1)

        train_vectors[i] = img_vec
        print(i)  # 输出当前进度
    # 使用savetxt()函数来把矩阵存到一个文本文件中
    output_filename = f"poison_train_vectors_{str(poison_ratio)}.txt"
    np.savetxt(output_filename, train_vectors)
    print("投毒数据存储成功")


def main():
    # save_train_img()
    poison_train_img(0.1)


if __name__ == '__main__':
    main()
