import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
import joblib
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import xgboost
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
try:
    from IPython import get_ipython

    get_ipython().magic("clear")
    get_ipython().magic("reset -f")
except:
    pass


df_train = pd.read_csv('./houseprice/train.csv')
separator = "=" * 50
pd.options.display.max_rows = None
pd.options.display.max_columns = None


def check():
    print(
        f'{separator}\nThis dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns \n{separator}')
    print(f'The data types of the columns:\n{df_train.dtypes}\n{separator}')
    list_adjust_column = {}
    for col in df_train.columns:
        missing_check = df_train[col].isnull().sum()
        if missing_check > 0:
            list_adjust_column[col] = missing_check
    print(f'{df_train.describe()}\n{separator}')
    for key, value in list_adjust_column.items():
        print(f'{key} miss:{value}')
        print(
            f'valus_count:\n{df_train[key].value_counts().index}\n{separator}')


def processing():
    # 缺失值太多直接拿掉欄位
    df_train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence',
                   'FireplaceQu'], axis=1, inplace=True)

    '''剩餘需處理缺失值欄位
    LotFrontage房屋與街道之間的直線距離  
    MasVnrType外牆表層的類型。Done
    MasVnrArea外牆表層的面積。Done
    BsmtQual地下室的高度評分。Done
    BsmtCond地下室的現況評分。Done
    BsmtExposure地下室的透光性評分。Done
    BsmtFinType1 和 BsmtFinType2地下室完成區域的質量。Done
    Electrical電氣系統的類型。Done
    FireplaceQu壁爐的質量評分。Done
    GarageType車庫的位置。Done
    GarageYrBlt車庫建造的年份。Done
    GarageFinish車庫內部完成的質量。Done
    GarageCars車庫容車的數量。Done
    GarageArea車庫的面積。Done
    GarageQual 和 GarageCond車庫的質量和條件評分。Done
    '''

    # 外牆及車庫的部份則是用沒有和0來補上
    df_train['MasVnrType'] = df_train['MasVnrType'].fillna('noMas')
    df_train['MasVnrArea'] = df_train['MasVnrArea'].fillna(0)
    df_train['GarageType'] = df_train['GarageType'].fillna('noGarage')
    df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna(0)
    df_train['GarageFinish'] = df_train['GarageFinish'].fillna('noGarage')
    df_train['GarageCars'] = df_train['GarageCars'].fillna(0)
    df_train['GarageArea'] = df_train['GarageArea'].fillna(0)
    df_train['GarageQual'] = df_train['GarageQual'].fillna('noGarage')
    df_train['GarageCond'] = df_train['GarageCond'].fillna('noGarage')

    # 地下室的部份其缺失值佔總樣本的2.5%，但因為是小樣本不打算進行刪除動作，所以決定都以noBsmt補上
    df_train['BsmtQual'] = df_train['BsmtQual'].fillna('noBsmt')
    df_train['BsmtCond'] = df_train['BsmtCond'].fillna('noBsmt')
    df_train['BsmtExposure'] = df_train['BsmtExposure'].fillna('noBsmt')
    df_train['BsmtFinType1'] = df_train['BsmtFinType1'].fillna('noBsmt')
    df_train['BsmtFinType2'] = df_train['BsmtFinType2'].fillna('noBsmt')

    # Electrical缺一個，直接移除該樣本
    df_train.dropna(subset=['Electrical'], inplace=True)

    # LotFrontage
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df_train['LotFrontage'])
    plt.show()  # 發現平均數中位數接近，決定以平均數代替
    df_train['LotFrontage'] = df_train['LotFrontage'].fillna(70)
    # 這邊直接填70是因為測試樣本時不希望讓平均數有不同的情況


def select_features():
    delete_column = []
    # 類別資料
    for col in obj_list:
        chi_table = pd.crosstab(df_train[col], df_train['SalePrice'])
        chi2, p, _, _ = chi2_contingency(chi_table)
        if p > 0.05:
            print(f'{col} 卡方統計量:{chi2} p_value:{p}\n{separator}')
            delete_column.append(col)

    # 數值資料
    for col in num_list:
        test_col = df_train[col]
        correlation, p = pearsonr(test_col, df_train['SalePrice'])
        if p > 0.05:
            print(f'{col} 皮爾森相關係數:{correlation} p_value:{p}\n{separator}')
            delete_column.append(col)

    print(f'需要刪除的欄位數量是{len(delete_column)}\n{separator}')

    df_train.drop(delete_column, axis=1, inplace=True)


def feature_engineer():
    col_for_transform = ['Condition2', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'Heating', 'CentralAir', 'KitchenQual',
                         'MSZoning', 'Street', 'LotShape', 'LotConfig', 'Neighborhood', 'MasVnrType', 'Foundation', 'GarageFinish', 'SaleType', 'SaleCondition']
    label_encode = LabelEncoder()
    df_train_transform = df_train.copy()
    for col in col_for_transform:
        df_train_transform[col] = label_encode.fit_transform(
            df_train_transform[col])
    joblib.dump(label_encode, './houseprice/label_transform.pkl')

    scaler = MinMaxScaler()
    df_train_transform = pd.DataFrame(scaler.fit_transform(
        df_train_transform), columns=df_train_transform.columns)
    joblib.dump(scaler, './houseprice/MinMaxScaler.pkl')

    return df_train_transform


def split_train_test():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def fold():
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    cv_split = list(cv.split(X_train, y_train))

    return cv_split


def model_selection():
    model = [('LinearRegression', LinearRegression()), ('Ridge', Ridge(alpha=0.5)),
             ('Lasso', Lasso(alpha=0.5)), ('XGBoost', xgboost.XGBRegressor(random_state=42))]
    for index, (name, regressor) in enumerate(model):
        mse = []
        r2 = []
        print(f'\nModel: {name}\n{separator}')
        for i, (train_index, test_index) in enumerate(cv_split):
            X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
            regressor.fit(X_train_cv, y_train_cv)
            y_pred = regressor.predict(X_test_cv)
            MSE = mean_squared_error(y_test_cv, y_pred)
            R2 = r2_score(y_test_cv, y_pred)
            print(f'fold {i + 1} MSE:{MSE} R2:{R2}')
            mse.append(MSE)
            r2.append(R2)
        print(separator)
        print(f'<<' * 5 + "指標數值" + '>>'*5)
        print(f"MSE:{np.mean(mse)} +/- {np.std(mse)}")
        print(f"r2 score:{np.mean(r2):.2f}")

# 決定使用ridge的方法進行超參調整


def hyperparameter():
    params = {"alpha": [1e-5, 1e-1], "solver": ["auto", "svd", "cholesky", "lsqr",
                                                "sparse_cg", "sag", "saga"], "tol": np.logspace(-5, -1, num=5)}
    return params


def best_params():
    grid_search = GridSearchCV(Ridge(), params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE:{mse}\nR2:{r2}")

    joblib.dump(grid_search, "./houseprice/ridge_model.pkl")

    return grid_search


def visualize(X_test, y_test, model):
    y_pred = model.predict(X_test)

    # 分布圖
    plt.figure(figsize=(10, 10))
    sns.distplot(y_test, label="actual")
    sns.distplot(y_pred, label="predicted")
    plt.legend()
    plt.show()

    # 散點圖
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=y_test, y=y_pred)
    sns.regplot(x=y_test, y=y_pred, scatter=False,
                color='r', label='Regression Line')
    plt.title('Actual vs. Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.show()

    # 殘差圖
    resduial = y_test - y_pred
    plt.figure(figsize=(10, 10))
    sns.residplot(x=y_pred, y=resduial, lowess=True, color="g")
    plt.title("Residuals vs. Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()


if __name__ == "__main__":
    check()
    processing()
    obj_list = []
    num_list = []
    for index, content in df_train.dtypes.items():
        if content == 'object':
            obj_list.append(index)
        else:
            num_list.append(index)
    select_features()
    df_train_transform = feature_engineer()
    X = df_train_transform.drop(['SalePrice'], axis=1)
    y = df_train_transform['SalePrice']
    X_test, X_train, y_test, y_train = split_train_test()
    cv_split = fold()
    model_selection()
    params = hyperparameter()
    grid_search = best_params()
    visualize(X_test, y_test, grid_search)
