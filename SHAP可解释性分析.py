import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import learning_curve
import scipy.stats as stats
import seaborn as sns
import shap
import joblib
from tqdm import tqdm  # 用于显示进度条

# ========================== 第 1 部分：定义可视化函数 ==========================

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
    # plt.savefig('E:/Python_study/ProjectOutputs/XGBRegressor/learning_curves.png', dpi=400, bbox_inches='tight')
    plt.show()
    plt.clf()  # 清除当前图形，以释放内存

    # # 将学习曲线的数值保存到 CSV 文件
    # learning_data = pd.DataFrame({
    #     'Train Size': train_sizes,
    #     'Train Score Mean': train_scores_mean,
    #     'Train Score Std': train_scores_std,
    #     'Test Score Mean': test_scores_mean,
    #     'Test Score Std': test_scores_std
    # })
    #
    # learning_data.to_csv('E:/Python_study/ProjectOutputs/XGBRegressor/learning_curves_data.csv', index=False)
    # print("学习曲线的数值已保存到 'learning_curves_data.csv'")

# 定义损失曲线的函数
def plot_loss_curves(model, X_train, y_train, X_test, y_test, max_estimators=2000, step=10):
    train_errors = []  # 初始化训练误差列表
    test_errors = []  # 初始化测试误差列表

    # 减小步长，绘制更多点，使曲线更精细
    for n_estimators in range(1, max_estimators + 1, step):  # 每 step 个 n_estimators 作为间隔
        model.set_params(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        # 计算训练集和测试集的 MSE
        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))

        train_errors.append(train_rmse)
        test_errors.append(test_rmse)

    # 绘制更加精细的损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_estimators + 1, step), train_errors, "r-+", linewidth=2, label="Train RMSE")
    plt.plot(range(1, max_estimators + 1, step), test_errors, "b-", linewidth=2, label="Test RMSE")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Number of Estimators", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.title("Loss Curves (RMSE)")
    plt.grid(True)  # 添加网格以便更容易观察
    # plt.savefig('E:/Python_study/ProjectOutputs/XGBRegressor/Loss_Curves.png', dpi=400, bbox_inches='tight')
    plt.show()
    plt.clf()  # 清除当前图形，以释放内存

# ========================== 第 2 部分：读取并预处理数据 ==========================

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
X_train_original, X_test_original, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10
)
# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_original)
X_test = scaler.transform(X_test_original)
# 将 numpy 数组转成 DataFrame，保留列名
X_train_df = pd.DataFrame(X_train, columns=X_train_original.columns)
X_test_df = pd.DataFrame(X_test, columns=X_test_original.columns)

# 定义一个固定的 KFold 实例
kf = KFold(n_splits=10, shuffle=True, random_state=10)

# ========================== 第 3 部分：网格搜索 & 模型训练 ==========================

# 定义参数网格
param_grid = {
    'n_estimators': [1000],
    'learning_rate': [0.2],
    'max_depth': [2]
}

# 创建XGBRegressor模型，明确指定使用CPU
xgb = XGBRegressor(random_state=10, eval_metric='rmse', tree_method='hist') # 指定使用CPU

# 使用GridSearchCV搜索最佳参数
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=2)

try:
    grid_search.fit(X_train, y_train)
except XGBoostError as e:
    print("XGBoost 错误:", e)
except Exception as e:
    print("其他错误:", e)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# 使用最佳模型进行预测
best_xgb = grid_search.best_estimator_

# ========================== 第 4 部分：模型评估 & 可视化 ==========================

# 在测试集上进行预测
y_train_pred = best_xgb.predict(X_train)
y_test_pred = best_xgb.predict(X_test)

test_df = X_test_original.copy()  # 保留列名
test_df['True'] = y_test.values
test_df['Pred'] = y_test_pred
# test_df.to_csv("E:/Python_study/ProjectOutputs/test_data_original_with_pred.csv", index=False)
print("已导出到 'test_data_original_with_pred.csv'")

# # 训练集预测结果
# train_results = pd.DataFrame({
#     'y_train_true': y_train,
#     'y_train_pred': y_train_pred,
#     'train_residuals': y_train - y_train_pred
# })
# train_results.to_csv("E:/Python_study/ProjectOutputs/XGBRegressor/train_predictions.csv", index=False)
#
# # 测试集预测结果
# test_results = pd.DataFrame({
#     'y_test_true': y_test,
#     'y_test_pred': y_test_pred,
#     'test_residuals': y_test - y_test_pred
# })
# test_results.to_csv("E:/Python_study/ProjectOutputs/XGBRegressor/test_predictions.csv", index=False)
# joblib.dump(best_xgb, 'E:/Python_study/ProjectOutputs/XGBRegressor/best_xgb_model.joblib')


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
plt.scatter(y_test, y_test_pred, alpha=0.5, color='orange', label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel('True Value', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.title('Actual vs Predicted with XGBRegressor')
plt.legend()
# 设置网格
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
# 设置坐标轴刻度指向内
plt.tick_params(axis='both', direction='in', length=6)
# plt.savefig('E:/Python_study/ProjectOutputs/XGBRegressor/Actual vs Predicted with XGBRegressor.png', dpi=400, bbox_inches='tight')
plt.show()
plt.clf()  # 清除当前图形，以释放内存

# 绘制残差图
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, y_train - y_train_pred, alpha=0.5, color='blue',  label='Train')
plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.5, color='orange', label='Test')
plt.hlines(y=0, xmin=min(y_train_pred.min(), y_test_pred.min()), xmax=max(y_train_pred.max(), y_test_pred.max()), colors='black', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.legend()
# plt.savefig('E:/Python_study/ProjectOutputs/XGBRegressor/Residuals vs Predicted.png', dpi=400, bbox_inches='tight')
plt.show()
plt.clf()  # 清除当前图形，以释放内存

# 绘制Q-Q图
plt.figure(figsize=(10, 6))
stats.probplot(np.concatenate([y_train - y_train_pred, y_test - y_test_pred]), dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
# plt.savefig('E:/Python_study/ProjectOutputs/XGBRegressor/Q-Q Plot of Residuals.png', dpi=400, bbox_inches='tight')
plt.show()
plt.clf()  # 清除当前图形，以释放内存

# 在模型上绘制学习曲线
plot_learning_curves(best_xgb, X_train, y_train, kf, title='Learning Curves for XGBRegressor')

# # 调用函数绘制损失曲线
# plot_loss_curves(best_xgb, X_train, y_train, X_test, y_test)


# ================== SHAP可解释性分析 ==================
# 使用训练集数据计算 SHAP 值
explainer = shap.Explainer(best_xgb, X_train)
shap_values = explainer(X_train)

# ========== 1) 计算 & 保存 Bar Plot 数据 ==========

# 取所有样本、所有特征的 SHAP 值 => shap_values.values, 形状 [n_samples, n_features]
mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

# 拼到 DataFrame
bar_data = pd.DataFrame({
    'Feature': X_train_df.columns,
    'MeanAbsSHAP': mean_abs_shap
})

# 排序（从大到小）
bar_data = bar_data.sort_values(by='MeanAbsSHAP', ascending=False)

# 保存为 CSV
bar_data.to_csv("E:/Python_study/ProjectOutputs/SHAP/bar_data.csv", index=False)
print("Bar plot data saved to 'bar_data.csv'")

# ========== 2) 绘制并保存 Bar Plot 图 ==========

plt.figure()
shap.summary_plot(
    shap_values,
    X_train_df,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.gcf().set_size_inches(10, 8)
plt.xlabel("Mean Absolute SHAP Value", fontsize=16)
plt.ylabel("Features", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig("E:/Python_study/ProjectOutputs/SHAP/shap_bar_plot.png", dpi=400, bbox_inches='tight')
plt.show()

# 2) 绘制 SHAP 蜂巢散点图（beeswarm / dot plot）
plt.figure()
shap.summary_plot(
    shap_values,
    X_train_df,
    plot_type="dot",       # 或者省略 plot_type, 因为 dot 是默认
    max_display=20,        # 只显示前 20 个最重要特征
    color_bar=True,        # 是否显示右侧的颜色条
    # cmap="viridis",       # 一些版本可用, 更改配色
    show=False
)

# 再次用 Matplotlib 修改细节
plt.gcf().set_size_inches(10, 8)
# plt.title("My Custom SHAP Beeswarm Plot", fontsize=16)
plt.xlabel("SHAP Value (impact on model output)", fontsize=16)
plt.ylabel("Features", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig("E:/Python_study/ProjectOutputs/SHAP/shap_beeswarm_plot.png", dpi=400, bbox_inches='tight')
plt.show()


for i in range(X_train.shape[1]):
    shap.dependence_plot(i, shap_values.values, X_train,
                         feature_names=data.columns[:-1],
                         show=False,dot_size=20)
    # 2) 获取当前 Axes，并做定制
    ax = plt.gca()

    # 2.1 显示右、上轴
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    # 2.2 添加网格
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tick_params(axis='both', direction='in', length=6)
    # 3) 保存并清理
    plt.savefig(f"E:/Python_study/ProjectOutputs/SHAP/shap_dependence_plot_{i}.png",
                dpi=400, bbox_inches='tight')
    plt.clf()  # 清除当前图像，准备下一个循环

# 2 局部解释
#Force Plot
i = 669  # 第 i 条样本
shap_values_single = shap_values[i]
feature_names = X_train_df.columns

# round 以减少小数位
shap_values_single.values = np.round(shap_values_single.values, 3)
shap_values_single.base_values = round(shap_values_single.base_values, 3)

original_feat_values = X_train_original.iloc[i, :]

# 拼装 DataFrame
force_df = pd.DataFrame({
    'Feature': feature_names,
    'ShapValue': shap_values_single.values,
    'FeatureValue': X_train_df.iloc[i, :].round(3).values,
    'original_feat_values': X_train_original.iloc[i, :]
})

force_df = force_df.sort_values(by='ShapValue', ascending=False)  # 可按SHAP值排序
force_df.to_csv(f"E:/Python_study/ProjectOutputs/SHAP/force_sample_{i}.csv", index=False)

# Force Plot
plt.figure(figsize=(10,4))
force_fig = shap.force_plot(
    shap_values_single.base_values,
    shap_values_single.values,
    X_train_df.iloc[i, :].round(3),
    feature_names=feature_names,
    matplotlib=True
)
force_fig_html = shap.force_plot(
    shap_values_single.base_values,
    shap_values_single.values,
    X_train_df.iloc[i, :].round(3),
    feature_names=X_train_df.columns
)

plt.savefig(f"E:/Python_study/ProjectOutputs/SHAP/Force_Plot_sample_{i}.png", dpi=400, bbox_inches='tight')
shap.save_html(f"E:/Python_study/ProjectOutputs/SHAP/Force_Plot_sample_{i}.html", force_fig_html)
plt.show()


