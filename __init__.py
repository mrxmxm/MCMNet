import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 生成示例数据
data = np.random.rand(10, 12)  # 生成 10x12 的随机数据
df = pd.DataFrame(data, columns=[f'Col{i}' for i in range(1, 13)])

# 使用 seaborn 绘制热力图并结合 matplotlib 进行进一步定制
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(df, annot=True, cmap='viridis', linewidths=.5)

# 设置标题和标签
plt.title('Heatmap with Seaborn and Matplotlib')
plt.xlabel('Columns')
plt.ylabel('Rows')

# 显示图形
plt.show()