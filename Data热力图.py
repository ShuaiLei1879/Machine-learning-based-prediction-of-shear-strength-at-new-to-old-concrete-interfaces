import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import learning_curve
import scipy.stats as stats
import seaborn as sns
import shap

# 加载数据
data = pd.read_csv('path_to_your_data.csv')  # 替换成你的实际文件路径

# 尝试转换为数值型，非数值型的转换成NaN
for col in data.columns[:-1]:  # 假设最后一列是目标值
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 处理缺失值：删除或填充
data = data.dropna()  # 删除包含缺失值的行

# 分离特征和目标变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 数据集的可视化
# 分别绘制每个特征的折线图和直方图
# for col in X.columns:
#     # 折线图
#     plt.figure(figsize=(10, 4))
#     plt.plot(X[col].sort_values().reset_index(drop=True))
#     plt.title(f'Line Plot of {col}')
#     plt.xlabel('Index')
#     plt.ylabel(f'{col}')
#     plt.grid(True)
#     plt.show()
#
#     # 直方图
#     plt.figure(figsize=(10, 4))
#     plt.hist(X[col], bins=30, alpha=0.75)
#     plt.title(f'Histogram of {col}')
#     plt.xlabel(f'{col}')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.show()

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 绘制特征之间的相关性矩阵热力图
# 转换为DataFrame并保留列名
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
correlation_matrix = X_scaled.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Heatmap of Feature Correlation')
plt.savefig('E:/Python_study/ProjectOutputs/Heatmap of Feature Correlation.png', dpi=400, bbox_inches='tight')
plt.show()