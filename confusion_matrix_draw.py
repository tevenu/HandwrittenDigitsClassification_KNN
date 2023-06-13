import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取混淆矩阵数据
confusion_matrix = np.loadtxt('confusion_matrix.txt', dtype=int)

# 绘制混淆矩阵热力图
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
