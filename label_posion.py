import random


def poison_labels(input_file, poison_ratio):
    with open(input_file, 'r') as f:
        labels = f.readlines()

    num_poisoned = int(len(labels) * poison_ratio)
    indices = random.sample(range(len(labels)), num_poisoned)

    poisoned_labels = labels.copy()
    for idx in indices:
        original_label = int(labels[idx])
        while True:
            new_label = random.randint(0, 9)
            if new_label != original_label:
                poisoned_labels[idx] = str(new_label) + '\n'  # 添加换行符
                break

    output_filename = f"poisoned_labels_train_{int(poison_ratio * 100)}.txt"
    with open(output_filename, 'w') as f:
        f.writelines(poisoned_labels)


def main():
    # 设置要修改的标签比例
    poison_ratio = 0.5

    # 调用函数进行标签投毒
    poison_labels(r"D:\大学文件\大三下\人工智能导论\Mnist\labels_train.txt", poison_ratio)


if __name__ == "__main__":
    main()
