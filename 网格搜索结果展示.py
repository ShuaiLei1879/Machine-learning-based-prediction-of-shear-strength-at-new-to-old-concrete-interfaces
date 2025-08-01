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
    plt.show()
    plt.clf()  # 清除当前图形，以释放内存

    # 将学习曲线的数值保存到 CSV 文件
    learning_data = pd.DataFrame({
        'Train Size': train_sizes,
        'Train Score Mean': train_scores_mean,
        'Train Score Std': train_scores_std,
        'Test Score Mean': test_scores_mean,
        'Test Score Std': test_scores_std
    })

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
    plt.title("Loss_Curves (RMSE)")
    plt.grid(True)  # 添加网格以便更容易观察
    plt.show()
    plt.clf()  # 清除当前图形，以释放内存

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

# 将数据分为训练集和测试集，这里分割30%为测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=10)
# 定义一个固定的 KFold 实例
kf = KFold(n_splits=10, shuffle=True, random_state=10)

# 定义参数网格, 0.1, 0.05, 0.01
param_grid = {
    'n_estimators': [1000],
    'learning_rate': [0.01],
    'max_depth': [8,10,12,14],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    'min_child_weight': [1],
    'gamma': [0.1],
    'reg_alpha': [0.1],
    'reg_lambda': [0.1]
}

# 创建XGBRegressor模型
xgb = XGBRegressor(random_state=10)

# 使用GridSearchCV搜索最佳参数
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# 使用最佳模型进行预测
best_xgb = grid_search.best_estimator_

# 在测试集上进行预测
y_train_pred = best_xgb.predict(X_train)
y_test_pred = best_xgb.predict(X_test)

# 评估模型
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
explained_var_train = explained_variance_score(y_train, y_train_pred)
explained_var_test = explained_variance_score(y_test, y_test_pred)

print('训练集均方误差:', mse_train)
print('测试集均方误差:', mse_test)
print('训练集均方根误差:', rmse_train)
print('测试集均方根误差:', rmse_test)
print('训练集R平方:', r2_train)
print('测试集R平方:', r2_test)
print('训练集解释方差分数:', explained_var_train)
print('测试集解释方差分数:', explained_var_test)



# 绘制训练数据和测试数据的真实值与预测值
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue', label='Train')
plt.scatter(y_test, y_test_pred, alpha=0.5, color='orange', label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel('True Value')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted with XGBRegressor')
plt.legend()
# 设置网格
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
# 设置坐标轴刻度指向内
plt.tick_params(axis='both', direction='in', length=6)
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
plt.show()
plt.clf()  # 清除当前图形，以释放内存

# 绘制Q-Q图
plt.figure(figsize=(10, 6))
stats.probplot(np.concatenate([y_train - y_train_pred, y_test - y_test_pred]), dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()
plt.clf()  # 清除当前图形，以释放内存

# 在模型上绘制学习曲线
plot_learning_curves(best_xgb, X_train, y_train, kf, title='Learning Curves for XGBRegressor')

# 调用函数绘制损失曲线
plot_loss_curves(best_xgb, X_train, y_train, X_test, y_test)


