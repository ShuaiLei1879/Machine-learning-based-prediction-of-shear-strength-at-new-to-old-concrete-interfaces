import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import learning_curve
import scipy.stats as stats
import shap

# ======================= 第 1 部分：定义可视化函数 (与原文相同) =======================
def plot_learning_curves(model, X, y, kf, title='Learning Curves'):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=kf,
        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10),
        scoring='r2')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig('E:/Python_study/ProjectOutputs/Lasso/learning_curves.png', dpi=400, bbox_inches='tight')
    plt.show()


def plot_loss_curves(model, X_train, y_train, X_test, y_test, step=1):
    train_errors = []
    test_errors = []
    m = len(X_train)

    for i in range(10, m, step):
        model.fit(X_train[:i], y_train[:i])
        y_train_predict = model.predict(X_train[:i])
        y_test_predict = model.predict(X_test)

        train_mse = mean_squared_error(y_train[:i], y_train_predict)
        test_mse = mean_squared_error(y_test, y_test_predict)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

    plt.plot(range(10, 10 + step * len(train_errors), step), train_errors, "r-+", linewidth=2, label="Train MSE")
    plt.plot(range(10, 10 + step * len(test_errors), step), test_errors, "b-", linewidth=2, label="Test MSE")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.title("Loss Curves (MSE)")
    plt.grid(True)
    plt.savefig('E:/Python_study/ProjectOutputs/Lasso/Loss_Curves.png', dpi=400, bbox_inches='tight')
    plt.show()


# ======================= 第 2 部分：数据加载与预处理 =======================
data = pd.read_csv('path_to_your_data.csv')  # 替换成你的实际文件路径

# 删除缺失值行
data = data.dropna()

# 将非数值特征转换为NaN，再次删除
for col in data.columns[:-1]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 多项式特征
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=10)

# ======================= 第 3 部分：网格搜索 =======================
# 1) 定义交叉验证策略
kf = KFold(n_splits=10, shuffle=True, random_state=10)

# 2) 定义参数网格
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],   # Lasso中最常调的参数
    'max_iter': [1000, 10000, 100000]
}

# 3) 实例化模型（基模型）
lasso = Lasso(random_state=10)  # 可以设置随机种子或不设置

# 4) 使用 GridSearchCV 搜索最佳参数
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='r2', cv=kf, n_jobs=-1, verbose=2)

# 5) 在训练集上执行网格搜索
grid_search.fit(X_train, y_train)

# 6) 查看最佳参数及其成绩
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation R^2:", grid_search.best_score_)

# 7) 获取最佳模型
best_model = grid_search.best_estimator_

# ======================= 第 4 部分：最终训练和评估 =======================
# 用最佳模型对训练集和测试集进行预测
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# 训练集预测结果
train_results = pd.DataFrame({
    'y_train_true': y_train,
    'y_train_pred': y_train_pred,
    'train_residuals': y_train - y_train_pred
})
train_results.to_csv("E:/Python_study/ProjectOutputs/Lasso/train_predictions.csv", index=False)

# 测试集预测结果
test_results = pd.DataFrame({
    'y_test_true': y_test,
    'y_test_pred': y_test_pred,
    'test_residuals': y_test - y_test_pred
})
test_results.to_csv("E:/Python_study/ProjectOutputs/Lasso/test_predictions.csv", index=False)

# 评估模型
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
explained_var_train = explained_variance_score(y_train, y_train_pred)
explained_var_test = explained_variance_score(y_test, y_test_pred)

print('训练集MSE:', mse_train)
print('测试集MSE:', mse_test)
print('训练集RMSE:', rmse_train)
print('测试集RMSE:', rmse_test)
print('训练集MAE:', mae_train)
print('测试集MAE:', mae_test)
print('训练集R²:', r2_train)
print('测试集R²:', r2_test)
print('训练集解释方差分数:', explained_var_train)
print('测试集解释方差分数:', explained_var_test)


# 绘制训练数据和测试数据的真实值与预测值
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue', label='Train')
plt.scatter(y_test, y_test_pred, alpha=0.5, color='red', label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel('True Value', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.title('Actual vs Predicted with Lasso Regression')
plt.legend()
# 设置网格
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
# 设置坐标轴刻度指向内
plt.tick_params(axis='both', direction='in', length=6)
plt.savefig('E:/Python_study/ProjectOutputs/Lasso/Actual vs Predicted with Lasso Regression.png', dpi=400, bbox_inches='tight')
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, y_train - y_train_pred, alpha=0.5, color='blue', label='Train')
plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.5, color='orange', label='Test')
plt.hlines(y=0, xmin=min(y_train_pred.min(), y_test_pred.min()), xmax=max(y_train_pred.max(), y_test_pred.max()), colors='black', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.legend()
plt.savefig('E:/Python_study/ProjectOutputs/Lasso/Residuals vs Predicted.png', dpi=400, bbox_inches='tight')
plt.show()

# 绘制Q-Q图
plt.figure(figsize=(10, 6))
stats.probplot(np.concatenate([y_train - y_train_pred, y_test - y_test_pred]), dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.savefig('E:/Python_study/ProjectOutputs/Lasso/Q-Q Plot of Residuals.png', dpi=400, bbox_inches='tight')
plt.show()

# 在模型上绘制学习曲线
plot_learning_curves(best_model, X_train, y_train, kf, title='Learning Curves for Lasso Regression')

# # 调用函数绘制损失曲线
# plot_loss_curves(best_model, X_train, y_train, X_test, y_test, step=1)

# # 使用SHAP进行模型解释
# explainer = shap.Explainer(model, X_train)
# shap_values = explainer(X_test)
#
# # 绘制 SHAP 总结图
# shap.summary_plot(shap_values, X_test, feature_names=poly.get_feature_names_out(data.columns[:-1]))
#
# # 选择一个特征绘制 SHAP 依赖图，例如特征0
# shap.dependence_plot(0, shap_values.values, X_test, feature_names=poly.get_feature_names_out(data.columns[:-1]))


