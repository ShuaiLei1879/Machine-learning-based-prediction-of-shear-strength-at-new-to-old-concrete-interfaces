import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建多项式特征
poly = PolynomialFeatures(degree=3)  # 可以调整多项式的度
X_poly = poly.fit_transform(X_scaled)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 创建线性回归模型来拟合多项式转换后的特征
model = LinearRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 评估模型
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print('训练集均方误差:', mse_train)
print('测试集均方误差:', mse_test)
print('训练集均方根误差:', rmse_train)
print('测试集均方根误差:', rmse_test)
print('训练集R平方:', r2_train)
print('测试集R平方:', r2_test)

# 绘制训练数据和测试数据的真实值与预测值
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train')
plt.scatter(y_test, y_test_pred, alpha=0.5, color='red', label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel('True Value')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

# 使用SHAP进行模型解释
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 绘制 SHAP 总结图
shap.summary_plot(shap_values, X_test, feature_names=poly.get_feature_names_out(input_features=data.columns[:-1]))

# 选择一个特征绘制 SHAP 依赖图
feature_index = 1  # 可以选择任何你想观察的特征索引
shap.dependence_plot(feature_index, shap_values.values, X_test, feature_names=poly.get_feature_names_out(input_features=data.columns[:-1]))
