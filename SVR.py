import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score,  mean_absolute_error
from sklearn.model_selection import learning_curve
import scipy.stats as stats
import shap

# 定义用于生成学习曲线的函数
def plot_learning_curves(model, X, y, kf, title='Learning Curves'):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=kf,
        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10),
        scoring='r2')

    # 计算训练和测试分数的平均值和标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    # 填充训练和测试分数之间的颜色
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # 绘制学习曲线
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig('E:/Python_study/ProjectOutputs/SVR/learning_curves.png', dpi=400, bbox_inches='tight')
    plt.show()

# 定义损失曲线的函数
def plot_loss_curves(model, X_train, y_train, X_test, y_test, step=1):
    train_errors = []
    test_errors = []
    m = len(X_train)

    for i in range(10, m, step):  # 逐步增加训练集大小
        model.fit(X_train[:i], y_train[:i])
        y_train_predict = model.predict(X_train[:i])
        y_test_predict = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train[:i], y_train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))

        train_errors.append(train_rmse)
        test_errors.append(test_rmse)

    plt.plot(range(10, 10 + step * len(train_errors), step), train_errors, "r-+", linewidth=2, label="Train RMSE")
    plt.plot(range(10, 10 + step * len(test_errors), step), test_errors, "b-", linewidth=2, label="Test RMSE")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.title("Loss Curves (RMSE)")
    plt.grid(True)
    plt.savefig('E:/Python_study/ProjectOutputs/SVR/Loss_Curves.png', dpi=400, bbox_inches='tight')
    plt.show()

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

# 转换为 DataFrame 并保留列名
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 使用 PCA 降维，保留95%的方差
pca = PCA(n_components=0.95, random_state=10)

X_scaled_pca = pca.fit_transform(X_scaled)
X_scaled_pca = pd.DataFrame(X_scaled_pca, columns=[f'PC{i+1}' for i in range(X_scaled_pca.shape[1])])

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled_pca, y, test_size=0.3, random_state=10)

# 定义一个固定的 KFold 实例
kf = KFold(n_splits=10, shuffle=True, random_state=10)

# 定义参数网格
param_grid = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],  # 考虑多种核函数，如 'rbf', 'linear', 'poly', 'sigmoid'
    'C': [5, 10, 15, 20],  # 调整C的值
    'gamma': [0.1, 0.2, 0.3, 0.4],  # gamma值的选择
    'epsilon': [0.5]  # epsilon的不同选择
}

# 使用网格搜索优化模型
grid_search = GridSearchCV(SVR(), param_grid, cv=kf, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证得分: {:.2f}".format(grid_search.best_score_))

# 使用最佳模型进行预测
best_svr = grid_search.best_estimator_
y_train_pred = best_svr.predict(X_train)
y_test_pred = best_svr.predict(X_test)

# 训练集预测结果
train_results = pd.DataFrame({
    'y_train_true': y_train,
    'y_train_pred': y_train_pred,
    'train_residuals': y_train - y_train_pred
})
train_results.to_csv("E:/Python_study/ProjectOutputs/SVR/train_predictions.csv", index=False)

# 测试集预测结果
test_results = pd.DataFrame({
    'y_test_true': y_test,
    'y_test_pred': y_test_pred,
    'test_residuals': y_test - y_test_pred
})
test_results.to_csv("E:/Python_study/ProjectOutputs/SVR/test_predictions.csv", index=False)


# 评估模型
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
explained_var_train = explained_variance_score(y_train, y_train_pred)
explained_var_test = explained_variance_score(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print('训练集均方误差:', mse_train)
print('测试集均方误差:', mse_test)
print('训练集均方根误差:', rmse_train)
print('测试集均方根误差:', rmse_test)
print('训练集R平方:', r2_train)
print('测试集R平方:', r2_test)
print('训练集解释方差分数:', explained_var_train)
print('测试集解释方差分数:', explained_var_test)
print('训练集平均绝对误差:', mae_train)
print('训练集平均绝对误差:', mae_test)

# 绘制训练数据和测试数据的真实值与预测值
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue', label='Train')
plt.scatter(y_test, y_test_pred, alpha=0.5, color='red', label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel('True Value', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.title('Actual vs Predicted with SVM Regression')
# 设置网格
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
# 设置坐标轴刻度指向内
plt.tick_params(axis='both', direction='in', length=6)
plt.savefig('E:/Python_study/ProjectOutputs/SVR/Actual vs Predicted with SVR Regression.png', dpi=400, bbox_inches='tight')
plt.legend()


# 绘制残差图
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, y_train - y_train_pred, alpha=0.5, color='blue', label='Train')
plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.5, color='orange', label='Test')
plt.hlines(y=0, xmin=min(y_train_pred.min(), y_test_pred.min()), xmax=max(y_train_pred.max(), y_test_pred.max()), colors='black', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.legend()
plt.savefig('E:/Python_study/ProjectOutputs/SVR/Residuals vs Predicted.png', dpi=400, bbox_inches='tight')
plt.show()

# 绘制Q-Q图
plt.figure(figsize=(10, 6))
stats.probplot(np.concatenate([y_train - y_train_pred, y_test - y_test_pred]), dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.savefig('E:/Python_study/ProjectOutputs/SVR/Q-Q Plot of Residuals.png', dpi=400, bbox_inches='tight')
plt.show()

# 在模型上绘制学习曲线
plot_learning_curves(best_svr, X_train, y_train, kf, title='Learning Curves for SVR Regression')

# 调用函数绘制损失曲线
plot_loss_curves(best_svr, X_train, y_train, X_test, y_test, step=1)


