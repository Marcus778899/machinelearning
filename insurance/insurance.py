import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost
import optuna
import joblib
import warnings
warnings.filterwarnings("ignore")
try:
    from IPython import get_ipython

    get_ipython().magic("clear")
    get_ipython().magic("reset -f")
except:
    pass


# 檢查資料
df = pd.read_csv('./insurance/insurance.csv', header=0)
print(df.head())
print('='*100)
print(df.info())
print('='*100)
print(df.describe())
print('='*100)
print(df.isnull().sum())
print('='*100)
# 畫出相關係性熱圖
sns.pairplot(df, hue='region')
plt.show()


def preprocessor():
    column_label = ['sex', 'smoker', 'region']
    column_stander = ['age', 'bmi', 'children']
    transfrom = ColumnTransformer(
        transformers=[
            ('onehot', OrdinalEncoder(), column_label),
            ('scaler', StandardScaler(), column_stander)
        ])
    encode_data = pd.DataFrame(transfrom.fit_transform(
        df), columns=transfrom.get_feature_names_out())
    joblib.dump(transfrom, './insurance/transform_data.pkl')
    encode_data = pd.concat([df.charges, encode_data], axis=1)
    print(encode_data)
    return encode_data


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def create_fold():
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_split = list(cv.split(X_train, y_train))

    return cv_split


def model_select(X_train, y_train, cv_split):
    regressors = [('LinearRegression', LinearRegression()),
                  ('Ridge', Ridge(alpha=0.5)),
                  ('Lasso', Lasso(alpha=0.5)),
                  ('XGBoost', xgboost.XGBRegressor(random_state=42))]
    for name, regressor in regressors:
        mse = []
        r2 = []
        print(f'\nModel: {name}')
        for i, (train_index, test_index) in enumerate(cv_split):
            print("-" * 50)
            X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

            regressor.fit(X_train_cv, y_train_cv)
            y_pred = regressor.predict(X_test_cv)

            rmse = mean_squared_error(y_test_cv, y_pred, squared=False)
            print(f"fold {i + 1} rmse:{rmse}")
            score = r2_score(y_test_cv, y_pred)
            print(f"fold {i + 1} r2:{score}")
            # 分組的評估值
            mse.append(rmse)
            r2.append(score)
        print('<<' * 5 + "評估指標" + '>>'*5)
        print(f"MSE:{np.mean(mse):.2f} +/- {np.std(mse):.2f}")
        print(f"r2 score:{np.mean(r2):.2f}")
        print('='*100)

# 因為以上四個表現並不亮眼，再多試一個polynomial


def poly(X_train, y_train, cv_split):
    print(f'\nModel: Polynomial')
    poly = PolynomialFeatures(degree=2)
    mse = []
    r2 = []
    for i, (train_index, val_index) in enumerate(cv_split):
        X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        X_train_cv_poly = poly.fit_transform(X_train_cv)
        X_test_cv_poly = poly.transform(X_test_cv)

        poly_model = LinearRegression()
        poly_model.fit(X_train_cv_poly, y_train_cv)
        y_pred = poly_model.predict(X_test_cv_poly)

        rmse = mean_squared_error(y_test_cv, y_pred, squared=False)
        score = r2_score(y_test_cv, y_pred)

        print(f"fold {i + 1} rmse:{rmse}")
        print(f"fold {i + 1} r2:{score}")
        print('=' * 100)

        mse.append(rmse)
        r2.append(score)

    print('<<' * 5 + "評估指標" + '>>'*5)
    print(f"MSE:{np.mean(mse):.2f} +/- {np.std(mse):.2f}")
    print(f"r2 score:{np.mean(r2):.2f}")
    print('='*100)


def hyperparamter(trial):
    params = {"degree": trial.suggest_int("degree", 1, 10)}
    return params


def objective(trial):
    params = hyperparamter(trial)
    poly = PolynomialFeatures(degree=params["degree"])
    X_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    score = cross_val_score(model, X_poly, y_train, cv=5,
                            scoring="neg_mean_squared_error")
    return score.mean()


def create_model(best_params):
    poly_model = PolynomialFeatures(degree=best_params["degree"])
    X_poly = poly_model.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)

    X_test_poly = poly_model.transform(X_test)
    y_pred = model.predict(X_test_poly)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE:{rmse:.2f}")
    print(f"R2:{r2:.2f}")
    joblib.dump(poly_model, "./insurance/polynomialFeatures.pkl")
    joblib.dump(model, "./insurance/model.pkl")


if __name__ == '__main__':
    encode_data = preprocessor()
    X = encode_data.drop('charges', axis=1)
    y = encode_data['charges']
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    cv_split = create_fold()
    model_select(X_train, y_train, cv_split)
    poly(X_train, y_train, cv_split)
    # 多項式表現最佳r2_score是0.81
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    best_params = study.best_params
    print(best_params)
    create_model(best_params)
