import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from ThymeBoost import ThymeBoost as tb
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def prepare_train_test():
    train_df = pd.read_csv(r'D:\Python projects\M4\dataset\Hourly-train.csv')
    test_df = pd.read_csv(r'D:\Python projects\M4\dataset\Hourly-test.csv')
    train_df.index = train_df['V1']
    train_df = train_df.drop('V1', axis=1)
    test_df.index = test_df['V1']
    test_df = test_df.drop('V1', axis=1)
    return train_df, test_df


train_df, test_df = prepare_train_test()
x_train = train_df.iloc[:, :46]
y_train = train_df.iloc[:, 47]
x_test = test_df.iloc[:, :46]
y_test = test_df.iloc[:, 47]


estimators_list = [
    ('gbr', GradientBoostingRegressor()),
    ('ada', AdaBoostRegressor()),
    ('bag_huber',BaggingRegressor(base_estimator=HuberRegressor())),
    ('bag_theil',BaggingRegressor(base_estimator=TheilSenRegressor())),
    ('bag_gbr',BaggingRegressor(base_estimator=GradientBoostingRegressor())),
    ('bag_knn',BaggingRegressor(base_estimator=KNeighborsRegressor())),
    ('bag_svr',BaggingRegressor(base_estimator=SVR())),
]


model = StackingRegressor(estimators=estimators_list, final_estimator=TheilSenRegressor())
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# df_y_pred = pd.DataFrame({'y_pred': y_pred})
# df_y_pred.to_csv('y_pred.csv')
print(smape(y_pred, y_test))
