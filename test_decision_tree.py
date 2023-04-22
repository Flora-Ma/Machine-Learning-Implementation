import numpy as np
import decision_tree

# test self implemented tree for binary classification.
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
decision_tree.build_tree_recursive(X_train, y_train, np.arange(len(X_train)), "Root", max_depth = 5, current_depth=0, tree=[])

# XGBoost
# Classification
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print(y_pred)
# Regression
from xgboost import XGBRegressor
X_train = np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,1],[1,1,1],[1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,1,1]])
y_train = np.array([7.2, 8.8, 15, 9.2, 8.4, 7.6, 11, 10.2, 18, 20])
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print(f'prediction:{y_pred}, original:{y_train}')
