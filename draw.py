import matplotlib.pyplot as plt


def label_poison_draw(filename, type):
    # 读取文件并解析数据
    x = []  # 投毒比例
    y = []  # 准确率
    k_values = []  # 不同的k值

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                # 解析每一行数据
                line_parts = line.split(',')
                label_poison_ratio = line_parts[0].split(':')[1]
                accuracy = float(line_parts[3].split(':')[1])
                k = int(line_parts[1].split(':')[1])

                # 添加数据到相应的列表
                x.append(label_poison_ratio)
                y.append(accuracy)
                k_values.append(k)

    # 创建图形并绘制折线
    plt.figure(figsize=(10, 6))
    for k in set(k_values):
        k_x = [x[i] for i in range(len(x)) if k_values[i] == k]
        k_y = [y[i] for i in range(len(y)) if k_values[i] == k]
        plt.plot(k_x, k_y, label=f'k={k}')

    # 设置图形标题和轴标签
    if type == 'pepper':
        plt.title('椒盐投毒对模型的影响')
    elif type == 'label':
        plt.title('训练集标签投毒对模型的影响')
    else:
        print("type变量请输入pepper或label")
        return
    plt.xlabel('投毒比例')
    plt.ylabel('准确率')

    # 设置纵轴刻度密度
    plt.yticks([i / 100 for i in range(0, 105, 5)])

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()


def dif_k_acc():
    # 读取文件并解析数据
    k = []  # k值
    accuracy = []  # 准确率

    with open('test_result.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                # 解析每一行数据
                line_parts = line.split(',')
                k_value = int(line_parts[1].split(':')[1])
                accuracy_value = float(line_parts[3].split(':')[1])

                # 添加数据到相应的列表
                k.append(k_value)
                accuracy.append(accuracy_value)

    # 创建图形并绘制折线
    plt.figure(figsize=(10, 6))
    plt.plot(k, accuracy, marker='o')

    # 设置图形标题和轴标签
    plt.title('不同k值对模型的影响')
    plt.xlabel('k')
    plt.ylabel('准确率')

    # 显示图形
    plt.show()


def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # label_poison_draw("pepper_poison_result.txt", 'pepper')
    label_poison_draw("poison_test_result.txt", 'label')
    # dif_k_acc()


if __name__ == '__main__':
    main()
