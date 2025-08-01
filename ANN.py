import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
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
    plt.savefig('E:/Python_study/ProjectOutputs/MLPRegressor/learning_curves.png', dpi=400, bbox_inches='tight')
    plt.show()

# 定义损失曲线的函数
def plot_loss_curves(model, X_train, y_train, X_test, y_test, step=1):
    train_errors = []  # 初始化训练误差列表
    test_errors = []   # 初始化测试误差列表
    m = len(X_train)

    # 减小步长，绘制更多点，使曲线更精细
    for i in range(10, m + 1, step):  # 每 step 个样本进行训练和测试
        model.fit(X_train[:i], y_train[:i])
        y_train_predict = model.predict(X_train[:i])
        y_test_predict = model.predict(X_test)

        # 记录训练集和测试集的均方误差 (MSE)
        train_mse = mean_squared_error(y_train[:i], y_train_predict)
        test_mse = mean_squared_error(y_test, y_test_predict)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

    # 绘制更加精细的损失曲线
    plt.plot(range(10, 10 + step * len(train_errors), step), train_errors, "r-+", linewidth=2, label="Train MSE")
    plt.plot(range(10, 10 + step * len(test_errors), step), test_errors, "b-", linewidth=2, label="Test MSE")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.title("Loss Curves (MSE)")
    plt.grid(True)  # 添加网格以便更容易观察
    plt.savefig('E:/Python_study/ProjectOutputs/MLPRegressor/Loss_Curves.png', dpi=400, bbox_inches='tight')
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

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=10)

# 定义一个固定的 KFold 实例
kf = KFold(n_splits=10, shuffle=True, random_state=10)

# 定义参数网格
param_grid = {
    'hidden_layer_sizes': [(10,), (20,),  (50,), (100,)],  # 减少隐藏层的神经元数量
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.01, 0.1, 1.0],  # 增加正则化强度
    'learning_rate': ['adaptive', 'invscaling', 'constant'],
    'learning_rate_init': [0.1, 0.2, 0.5],  # 调整学习率
    'max_iter': [1000, 2000, 4000]
}


# 创建 MLPRegressor 模型
mlp = MLPRegressor(random_state=10)

# 使用 GridSearchCV 进行参数调优
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=kf, n_jobs=-1, scoring='r2', verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# 使用最佳模型进行预测
best_mlp = grid_search.best_estimator_

# 在训练集和测试集上进行预测
y_train_pred = best_mlp.predict(X_train)
y_test_pred = best_mlp.predict(X_test)

# 训练集预测结果
train_results = pd.DataFrame({
    'y_train_true': y_train,
    'y_train_pred': y_train_pred,
    'train_residuals': y_train - y_train_pred
})
train_results.to_csv("E:/Python_study/ProjectOutputs/MLPRegressor/train_predictions.csv", index=False)

# 测试集预测结果
test_results = pd.DataFrame({
    'y_test_true': y_test,
    'y_test_pred': y_test_pred,
    'test_residuals': y_test - y_test_pred
})
test_results.to_csv("E:/Python_study/ProjectOutputs/MLPRegressor/test_predictions.csv", index=False)

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

print('训练集均方误差 (MSE):', mse_train)
print('测试集均方误差 (MSE):', mse_test)
print('训练集均方根误差 (RMSE):', rmse_train)
print('测试集均方根误差 (RMSE):', rmse_test)
print('训练集R平方 (R²):', r2_train)
print('测试集R平方 (R²):', r2_test)
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
plt.title('Actual vs Predicted with MLPRegressor')
plt.legend()
# 设置网格
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
# 设置坐标轴刻度指向内
plt.tick_params(axis='both', direction='in', length=6)
plt.savefig('E:/Python_study/ProjectOutputs/MLPRegressor/Actual vs Predicted with MLPRegressor.png', dpi=400, bbox_inches='tight')
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
plt.savefig('E:/Python_study/ProjectOutputs/MLPRegressor/Residuals vs Predicted.png', dpi=400, bbox_inches='tight')
plt.show()

# 绘制Q-Q图
plt.figure(figsize=(10, 6))
stats.probplot(np.concatenate([y_train - y_train_pred, y_test - y_test_pred]), dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.savefig('E:/Python_study/ProjectOutputs/MLPRegressor/Q-Q Plot of Residuals.png', dpi=400, bbox_inches='tight')
plt.show()

# 在模型上绘制学习曲线
plot_learning_curves(best_mlp, X_train, y_train, kf, title='Learning Curves for MLPRegressor')

# 调用函数绘制损失曲线
plot_loss_curves(best_mlp, X_train, y_train, X_test, y_test, step=10)

# # 使用 SHAP 进行模型解释
# # 对于 MLPRegressor，使用 KernelExplainer，建议使用一个样本子集
# shap_sample = X_train.sample(n=100, random_state=10)
# explainer = shap.KernelExplainer(best_mlp.predict, shap_sample)
# shap_values = explainer.shap_values(shap_sample, nsamples=100)
#
# # 可视化 SHAP 值的摘要图
# shap.summary_plot(shap_values, shap_sample, feature_names=X_scaled.columns)
#
# # 获取 SHAP 值的绝对平均值，选择前10个最重要的特征
# mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
# important_features = shap_sample.columns[np.argsort(mean_abs_shap_values)][-10:]  # 选择10个最重要的特征
#
# # 绘制 SHAP 依赖图
# for feature in important_features:
#     shap.dependence_plot(feature, shap_values, shap_sample, feature_names=X_scaled.columns)
#
# # 计算并绘制蜂巢-条状组合图
# # 获取特征名称
# feature_names = shap_sample.columns
#
# # 计算平均绝对 SHAP 值
# mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
#
# # 创建包含特征名称和平均绝对 SHAP 值的 DataFrame
# shap_summary = pd.DataFrame({
#     'Feature': feature_names,
#     'MeanAbsShapValue': mean_abs_shap_values
# })
#
# # 按平均绝对 SHAP 值升序排序
# shap_summary = shap_summary.sort_values(by='MeanAbsShapValue', ascending=True)
#
# # 获取排序后的特征名称列表
# sorted_features = shap_summary['Feature'].values
#
# # 准备绘制蜂巢图的数据
# shap_values_sorted = shap_values[:, [list(feature_names).index(f) for f in sorted_features]]
#
# # 创建图形和坐标轴
# fig, ax1 = plt.subplots(figsize=(7, len(sorted_features) * 0.4))
#
# # 绘制条状图
# ax1.barh(range(len(sorted_features)), shap_summary['MeanAbsShapValue'], color='#D8B6D1')
# ax1.set_yticks(range(len(sorted_features)))
# ax1.set_yticklabels(sorted_features)
# ax1.set_xlabel('Average Absolute SHAP Value')
#
# # 创建共享 y 轴的次坐标轴，用于绘制蜂巢图
# ax2 = ax1.twiny()
#
# # 绘制蜂巢图，使用颜色映射
# for i in range(len(sorted_features)):
#     shap_vals = shap_values_sorted[:, i]
#     y = np.random.normal(i, 0.1, size=shap_vals.shape)
#     ax2.scatter(shap_vals, y, alpha=0.5, c=shap_vals, cmap='plasma', s=10)
#
# # 设置 x 轴范围
# max_shap = np.abs(shap_values).max()
# ax2.set_xlim(-2, max_shap)
# ax2.set_xlabel('SHAP Values (Impact on Model Output)')
#
# # 隐藏次坐标轴的 y 轴标签
# ax2.set_yticks([])
#
# # 添加特征名称
# for i, feature in enumerate(sorted_features):
#     ax1.text(-0.01, i, feature, ha='right', va='center', color='black', fontsize=10)
#
# # 调整布局并显示图形
# plt.title('SHAP Hexbin-Bar Combined Plot')
# plt.tight_layout()
# plt.show()
