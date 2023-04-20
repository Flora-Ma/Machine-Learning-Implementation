import feature_scaling
from regression import linear_regression
from regression import logistic_regression
import numpy as np

####################### Test linear regression #############################
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


######################### Test logistic regression ############################
# self implemented
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
w, b = logistic_regression.fit(X_train, y_train, num_iters=10000, alpha=0.1)
m, _ = X_train.shape
for i in range(m):
    p, y = logistic_regression.predict(X_train[i], w, b, threshold=0.5)
    print(f'prediction: p={p}, y={y}; target label={y_train[i]}')

# sklearn.linear_model.LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
p = clf.predict_proba(X_train)
label = clf.predict(X_train)
for i in range(m):
    print(f'prediction: p={p[i]}, y={label[i]}; target label={y_train[i]}')