import numpy as np
from PIL import Image


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
    file_root = r".\Mnist\img_train"
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


if __name__ == '__main__':
    poison_train_img(0.4)
