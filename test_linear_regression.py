import feature_scaling
import linear_regression
import numpy as np

# self implemented
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
w, b = linear_regression.fit(X_train, y_train, num_iters=1000, alpha=5.0e-7)
print(f'w={w}, b={b}')
m, _ = X_train.shape
for i in range(m):
    print(f'prediction: {linear_regression.predict(X_train[i], w, b):0.2f}, target_value={y_train[i]}')


X_train_norm, m_, std_ = feature_scaling.zscore_normalize_scaler(X_train)
w1, b1 = linear_regression.fit(X_train_norm, y_train, num_iters=1000, alpha=0.01)
print(f'w1={w1}, b={b1}')
for i in range(m):
    print(f'prediction: {linear_regression.predict(X_train_norm[i], w1, b1):0.2f}, target_value={y_train[i]}')

# sklearn.linear_model.LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
print(f'sklearn_w={reg.coef_}, sklearn_b={reg.intercept_}')
predicts = reg.predict(X_train)
for i in range(m):
    print(f'prediction: {predicts[i]:0.2f}, target_value={y_train[i]}')
