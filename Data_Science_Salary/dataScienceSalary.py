def load_data():
    import pandas as pd

    df = pd.read_csv("./Data_Science_Salary/Latest_Data_Science_Salaries .csv")
    print(df.info())
    print("-" * 50)
    # print(df["Employee Residence"].unique().tolist())
    # print(df["Job Title"].unique().tolist())
    # print(df["Employment Type"].unique().tolist())
    # print(df["Expertise Level"].unique().tolist())
    # print(df["Company Location"].unique().tolist())
    # print(df["Company Size"].unique().tolist())
    return df


def EDA(df):
    experience_salary = (
        df.groupby("Experience Level")["Salary in USD"].mean().reset_index()
    )
    print(df.groupby("Experience Level")["Salary in USD"].describe().astype(int))
    print("=" * 100)
    print("遺失值")
    print(df.isnull().sum())
    print("=" * 50)
    print("工作經驗薪水級距")
    print(experience_salary)
    from matplotlib import pyplot as plt

    # 職稱與薪水的柱狀圖
    plt.figure(figsize=(20, 10))
    plt.bar(
        df.groupby("Job Title")["Salary in USD"].mean().index,
        df.groupby("Job Title")["Salary in USD"].mean(),
    )
    plt.xlabel("Job Title")
    plt.ylabel("Salary in USD")
    plt.title("Job Title and Salary")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # 經驗與薪水的柱狀圖
    plt.bar(experience_salary["Experience Level"], experience_salary["Salary in USD"])
    plt.xlabel("Experience Level")
    plt.ylabel("Salary in USD")
    plt.title("Experience Level and Salary")
    plt.show()

    # 工作職稱數量前15名
    print("=" * 50)
    print(f"職稱{df['Job Title'].value_counts()}")
    plt.bar(
        df["Job Title"].value_counts().nlargest(15).index,
        df["Job Title"].value_counts().nlargest(15),
    )
    plt.xlabel("Job Title")
    plt.ylabel("Count")
    plt.title("Job Title")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # 工作類型數量
    print("=" * 50)
    print(f"工作類型{df['Employment Type'].value_counts()}")

    # 專業技術等級
    print("=" * 50)
    print(f"專業技術{df['Expertise Level'].value_counts()}")


def label_code(df):
    from sklearn.preprocessing import OrdinalEncoder
    import pandas as pd

    df_ = df.drop(["Employee Residence", "Salary"], axis=1)

    # 等級尺度標籤
    oe = OrdinalEncoder()
    col_oe = ["Experience Level", "Expertise Level", "Company Size"]
    oe_df = pd.DataFrame(
        oe.fit_transform(df_[col_oe]), columns=oe.get_feature_names_out(col_oe)
    )
    # OneHotEncoding
    col_ohe = ["Job Title", "Employment Type", "Salary Currency", "Company Location"]
    ohe_df = pd.get_dummies(df_[col_ohe], columns=col_ohe, prefix=col_ohe)
    df_ = pd.concat([df_.drop(col_ohe + col_oe, axis=1), oe_df, ohe_df], axis=1)

    # 先進行切分，test_set最後用來進行測試
    from sklearn.model_selection import train_test_split

    train_set, test_set = train_test_split(df_, test_size=0.2, random_state=123)
    x = train_set.drop("Salary in USD", axis=1)
    y = train_set["Salary in USD"]

    # 5折交叉驗證
    from sklearn.model_selection import KFold

    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    cv_split = list(cv.split(x, y))
    return cv_split, x, y, test_set


def model(cv_split, x, y):
    # 未調整超參前，挑選適合模型
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
        AdaBoostRegressor,
        HistGradientBoostingRegressor,
    )
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    regressors = [
        ("CatBoost", CatBoostRegressor(random_state=123, verbose=False)),
        ("XGboost", XGBRegressor(random_state=123)),
        ("Ada Boost", AdaBoostRegressor(random_state=123)),
        (
            "Histogram-based Gradient Boosting",
            HistGradientBoostingRegressor(random_state=123),
        ),
        ("Random Forest", RandomForestRegressor(random_state=123)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=123)),
    ]
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    for name, clf in regressors:
        MSE = []
        r2_scores = []
        print(f"\n{name} Regressor:\n")
        for i, (train_index, test_index) in enumerate(cv_split):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            print(f"fold {i + 1} rmse:{rmse}")
            r2 = r2_score(y_test, y_pred)
            print(f"fold {i + 1} r2:{r2}")
            MSE.append(mean_squared_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
            print("==" * 50)

        if i == len(cv_split) - 1:
            mean_score = np.mean(MSE)
            fold_std = np.std(MSE)
            mean_r2 = np.mean(r2_scores)
            print(f"MSE:{mean_score:.2f} +/- {fold_std:.2f}")
            print(f"r2:{mean_r2:.2f}")
            print("==" * 50)


def HistGradient(trial):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np

    params = {
        "loss": trial.suggest_categorical(
            "loss", ["absolute_error", "poisson", "squared_error"]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 1.0, step=0.2),
        "max_iter": trial.suggest_int("max_iter", 100, 1000, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 30, 10000, step=100),
        "max_depth": trial.suggest_int("max_depth", 30, 10000, step=100),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 1000, step=15),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 0.01, 100, step=0.05
        ),
    }
    clf = HistGradientBoostingRegressor(**params, random_state=123)
    rmse_score_hist = []
    for train_index, val_index in cv_split:
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        rmse_score_hist.append(rmse)

    return np.mean(rmse_score_hist)


def CatBoost(trial):
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np

    params = {
        "loss_function": trial.suggest_categorical(
            "loss_function", ["MAE", "RMSE", "Poisson"]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, step=0.01),
        "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10, step=0.1),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10, step=0.1),
        "eval_metric": "RMSE",
    }
    clf = CatBoostRegressor(**params, random_state=123)
    rmse_score_cat = []
    for train_index, val_index in cv_split:
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        rmse_score_cat.append(rmse)

    return np.mean(rmse_score_cat)


def StackingRegressor(best_params_hist, best_params_cat):
    from sklearn.ensemble import StackingRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    import numpy as np

    estimators = [
        (
            "HistGradientBoostingRegressor",
            HistGradientBoostingRegressor(random_state=123, **best_params_hist),
        ),
        ("CatBoostRegressor", CatBoostRegressor(random_state=123, **best_params_cat)),
    ]
    model_final = StackingRegressor(estimators=estimators, n_jobs=5)
    scores = []
    r2_scores = []
    for i, (train_index, test_index) in enumerate(cv_split):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model_final.fit(x_train, y_train)
        y_pred = model_final.predict(x_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        scores.append(rmse)
        r2_scores.append(r2)
        print("===================================================")
        if i == len(cv_split) - 1:
            mean_score = np.mean(scores)
            fold_std = np.std(scores)
            mean_r2 = np.mean(r2_scores)
            print(f"MSE:{mean_score:.2f} +/- {fold_std:.2f}")
            print(f"r2:{mean_r2:.2f}")
    import joblib

    joblib.dump(model_final, "model_final.pkl")


def model_test(test_set):
    import joblib

    load_model = joblib.load("model_final.pkl")
    x_test = test_set.drop(["Salary in USD"], axis=1)
    y_true = test_set["Salary in USD"]
    y_pred = load_model.predict(x_test)

    import pandas as pd

    predictions = pd.DataFrame(
        {"id": test_set.index, "Real Salary": y_true, "Predicted Salary": y_pred}
    )

    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    print("\nTest Results with Meta Model: \n")
    rmse = mean_squared_error(
        predictions["Real Salary"], predictions["Predicted Salary"], squared=False
    )
    print(f"RMSE = {rmse:.2f}")
    r2 = r2_score(predictions["Real Salary"], predictions["Predicted Salary"])
    print(f"r2 scroes = {r2:.2f}")

    # 畫圖
    import plotly.graph_objects as go
    import numpy as np

    slope, intercept = np.polyfit(
        predictions["Real Salary"], predictions["Predicted Salary"], 1
    )
    line = slope * predictions["Real Salary"] + intercept
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions["Real Salary"],
            y=predictions["Predicted Salary"],
            mode="markers",
            name="Points",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions["Real Salary"], y=line, mode="lines", name="Regression Line"
        )
    )
    fig.update_traces(marker=dict(size=12, color="green"))
    fig.update_layout(
        title="Real Salary vs Predicted Salary",
        xaxis_title="Real Salary",
        yaxis_title="Predicted Salary",
        height=750,
        width=850,
        margin=dict(t=250, l=80),
        template="simple_white",
    )
    fig.show()


if __name__ == "__main__":
    df = load_data()
    EDA(df)
    cv_split, x, y, test_set = label_code(df)
    model(cv_split, x, y)

    import optuna

    # HistGradientBoostingRegressor
    study = optuna.create_study(direction="minimize")
    study.optimize(HistGradient, n_trials=100, show_progress_bar=True)
    print(f"\nHistogram-based Gradient Boosting Regressor:\n")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("Best RMSE score : ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    best_params_hist = study.best_params

    print("=" * 50)
    # CatBoostRegressor
    study2 = optuna.create_study(direction="minimize")
    study2.optimize(CatBoost, n_trials=100, show_progress_bar=True)
    print(f"\nCatBoost Regressor:\n")
    print("Number of finished trials: ", len(study2.trials))
    print("Best trial:")
    trial = study2.best_trial
    print("Best RMSE score : ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    best_params_cat = study2.best_params
    StackingRegressor(best_params_hist, best_params_cat)
    model_test(test_set)
