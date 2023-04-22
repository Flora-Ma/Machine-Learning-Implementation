import logistic_regression
import numpy as np

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