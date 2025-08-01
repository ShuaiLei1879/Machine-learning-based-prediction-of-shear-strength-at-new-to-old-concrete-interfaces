import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import learning_curve
import scipy.stats as stats
import shap
import joblib

# 定义用于生成学习曲线的函数
def plot_learning_curves(model, X, y, kf, title):
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
    plt.savefig('E:/Python_study/ProjectOutputs/AdaBoostRegressor/learning_curves.png', dpi=400, bbox_inches='tight')
    plt.show()

# 定义损失曲线的函数
def plot_loss_curves(model, X_train, y_train, X_test, y_test, max_estimators=300):
    train_errors = []  # 初始化训练误差列表
    test_errors = []  # 初始化测试误差列表 

    # 减小步长，绘制更多点，使曲线更精细
    for n_estimators in range(1, max_estimators + 1, 5):  # 每10个n_estimators作为间隔
        model.set_params(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        # 计算训练集和测试集的RMSE
        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))

        train_errors.append(train_rmse)
        test_errors.append(test_rmse)

    # 绘制更加精细的损失曲线
    plt.plot(range(1, max_estimators + 1, 5), train_errors, "r-+", linewidth=2, label="Train")
    plt.plot(range(1, max_estimators + 1, 5), test_errors, "b-", linewidth=2, label="Test")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Number of Estimators", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.title("Loss Curves")
    plt.grid(True)  # 添加网格以便更容易观察
    plt.savefig('E:/Python_study/ProjectOutputs/AdaBoostRegressor/loss_curves.png', dpi=400, bbox_inches='tight')
    plt.show()

# 加载数据
data = pd.read_csv('path_to_your_data.csv')  # 替换成你的实际文件路径

# 尝试转换为数值型，非数值型的转换成NaN
for col in data.columns[:-1]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 处理缺失值：删除或填充
data = data.dropna()

# 分离特征和目标变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train_original, X_test_original, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10
)
# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_original)
X_test = scaler.transform(X_test_original)

# 定义一个固定的 KFold 实例
kf = KFold(n_splits=10, shuffle=True, random_state=10)

# 定义参数网格
param_grid = {
    'n_estimators': [1200, 1400],
    'learning_rate': [0.01],
    'estimator__max_depth': [14],
    'estimator__min_samples_split': [2],
    'estimator__min_samples_leaf': [3],
    'loss': ['exponential']
}

# 创建 AdaBoostRegressor 模型
ada = AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=10)

# 使用 GridSearchCV 进行参数调优
grid_search = GridSearchCV(estimator=ada, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)
# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# 使用最佳模型进行预测
best_ada = grid_search.best_estimator_
# 在测试集上进行预测
y_train_pred = best_ada.predict(X_train)
y_test_pred = best_ada.predict(X_test)

test_df = X_test_original.copy()  # 保留列名
test_df['True'] = y_test.values
test_df['Pred'] = y_test_pred
test_df.to_csv("E:/Python_study/ProjectOutputs/AdaBoostRegressor/test_data_original_with_pred.csv", index=False)
print("已导出到 'test_data_original_with_pred.csv'")

# 训练集预测结果
train_results = pd.DataFrame({
    'y_train_true': y_train,
    'y_train_pred': y_train_pred,
    'train_residuals': y_train - y_train_pred
})
train_results.to_csv("E:/Python_study/ProjectOutputs/AdaBoostRegressor/train_predictions.csv", index=False)

# 测试集预测结果
test_results = pd.DataFrame({
    'y_test_true': y_test,
    'y_test_pred': y_test_pred,
    'test_residuals': y_test - y_test_pred
})
test_results.to_csv("E:/Python_study/ProjectOutputs/AdaBoostRegressor/test_predictions.csv", index=False)
joblib.dump(best_ada, 'E:/Python_study/ProjectOutputs/AdaBoostRegressor/best_ada_model.joblib')

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
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue',  label='Train')
plt.scatter(y_test, y_test_pred, alpha=0.5, color='red', label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel('True Value', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.title('Actual vs Predicted with AdaBoostRegressor')
plt.legend()
# 设置网格
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
# 设置坐标轴刻度指向内
plt.tick_params(axis='both', direction='in', length=6)

plt.savefig('E:/Python_study/ProjectOutputs/AdaBoostRegressor/Actual vs Predicted with AdaBoostRegressor.png', dpi=400, bbox_inches='tight')
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, y_train - y_train_pred, alpha=0.5, color='blue',  label='Train')
plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.5, color='orange', label='Test')
plt.hlines(y=0, xmin=min(y_train_pred.min(), y_test_pred.min()),
           xmax=max(y_train_pred.max(), y_test_pred.max()), colors='black', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.legend()
plt.savefig('E:/Python_study/ProjectOutputs/AdaBoostRegressor/Residuals vs Predicted.png', dpi=400, bbox_inches='tight')
plt.show()

# 绘制Q-Q图
plt.figure(figsize=(10, 6))
stats.probplot(np.concatenate([y_train - y_train_pred, y_test - y_test_pred]), dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.savefig('E:/Python_study/ProjectOutputs/AdaBoostRegressor/Q-Q Plot of Residuals.png', dpi=400, bbox_inches='tight')
plt.show()

# 在模型上绘制学习曲线
plot_learning_curves(best_ada, X_train, y_train, kf, title='Learning Curves for AdaBoostRegressor')
# 调用函数绘制损失曲线
plot_loss_curves(best_ada, X_train, y_train, X_test, y_test)

# # 使用KernelExplainer进行模型解释
# explainer = shap.KernelExplainer(best_ada.predict, shap.sample(X_train, 200))  # 使用训练数据的一个子集作为背景数据
# shap_values = explainer(shap.sample(X_train, 200))
#
# # 可视化 SHAP 值的摘要图
# shap.dependence_plot(0, shap_values.values, shap.sample(X_train, 200), feature_names=data.columns[:-1])
#
# # 计算并绘制蜂巢-条状组合图
# # 获取特征名称
# feature_names = X.columns
#
# # 计算平均绝对 SHAP 值
# mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
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
# shap_values_sorted = shap_values.values[:, [list(feature_names).index(f) for f in sorted_features]]
#
# # 创建图形和坐标轴
# fig, ax1 = plt.subplots(figsize=(7, len(feature_names) * 0.4))
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
# # 绘制蜂巢图，使用两种颜色
# for i in range(len(sorted_features)):
#     shap_vals = shap_values_sorted[:, i]
#     y = np.random.normal(i, 0.1, size=shap_vals.shape)
#     ax2.scatter(shap_vals, y, alpha=0.5, c=shap_vals, cmap='plasma', s=10)
#
# # 设置 x 轴范围
# max_shap = np.abs(shap_values.values).max()
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



















# # 绘制SHAP依赖图（对于第一个特征）
# shap.dependence_plot(0, shap_values.values, X_train, feature_names=data.columns[:-1])

# # 可视化第一个预测的SHAP值
# shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
#
# # 可视化整个测试集的SHAP值
# shap.summary_plot(shap_values, X_test, feature_names=data.columns[:-1])
#
# # 绘制特定特征的SHAP依赖图，这里以特征索引为0的特征为例
# shap.dependence_plot(0, shap_values, X_test, feature_names=data.columns[:-1])
#
# # 如果你有特定的特征名称，可以直接使用名称
# # shap.dependence_plot('feature_name', shap_values, X_test, feature_names=data.columns[:-1])
# plt.show()

