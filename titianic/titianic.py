import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib


def load_data():
    df_train = pd.read_csv("./titianic/train.csv", header=0)
    print(df_train.isna().sum())
    print("-" * 50)
    print(df_train.describe())
    print("-" * 50)
    df_train.Age.fillna(df_train.Age.mode()[0], inplace=True)
    df_train.dropna(subset=["Embarked"], inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    return df_train


def EDA(df_train):
    # 性別比較
    women = df_train.loc[df_train.Sex == "female"]["Survived"]
    rate_women = sum(women) * 100 // len(women)
    print(f"女性存活率為{rate_women}%")
    men = df_train.loc[df_train.Sex == "male"]["Survived"]
    rate_men = sum(men) * 100 // len(men)
    print(f"男性存活率為{rate_men}%")

    # 相關程度視覺化
    feature = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    sns.set()
    sns.pairplot(df_train[feature], hue="Sex", height=1.5)
    plt.show()

    # 票價與存活的關聯
    plt.rcParams["font.family"] = "SimSun"
    ax = sns.kdeplot(
        df_train.Fare[(df_train["Survived"] == 1)], color="Red", fill=True)
    ax = sns.kdeplot(
        df_train.Fare[(df_train["Survived"] == 0)], ax=ax, color="Blue", fill=True
    )
    ax.legend(["存活", "死亡"], loc="upper right")
    ax.set_ylabel("Survived")
    ax.set_xlabel("Fare")
    ax.set_title("票價存活狀況")
    plt.show()

    # 年紀與存活關聯
    ax = sns.kdeplot(
        df_train.Age[(df_train["Survived"] == 1)], color="Red", fill=True)
    ax = sns.kdeplot(
        df_train.Age[(df_train["Survived"] == 0)], ax=ax, color="Blue", fill=True
    )
    ax.legend(["存活", "死亡"], loc="upper right")
    ax.set_ylabel("Survived")
    ax.set_xlabel("Age")
    ax.set_title("年齡存活狀況")
    plt.show()


def processing(df_train):
    df_train.drop(["Name", "Ticket", "Cabin"], inplace=True, axis=1)

    encoder = OneHotEncoder()
    X = df_train[["Age", "Fare", "Pclass", "SibSp", "Parch"]]
    encode_data = encoder.fit_transform(df_train[["Sex", "Embarked"]])
    encode_data = pd.DataFrame(
        encode_data.toarray(),
        columns=encoder.get_feature_names_out(["Sex", "Embarked"]),
    )
    X = pd.concat([X, encode_data], axis=1)
    y = df_train["Survived"]
    return X, y


def model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"{X_train.shape}{X_test.shape}{y_train.shape}{y_test.shape}")
    regressors = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("SVC", SVC(kernel="rbf", random_state=42)),
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(criterion="gini", random_state=42),
        ),
        ("RandomForestClassifier", RandomForestClassifier(random_state=42)),
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
        ("XGBClassifier", XGBClassifier(random_state=42)),
    ]

    for name, classifer in regressors:
        print(f"\n{name} Regressor:")
        classifer.fit(X_train, y_train)
        pred = classifer.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        print(f"Accuracy score = {accuracy:.2f}")
    return X_train, X_test, y_train, y_test


def RandomForest(X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": [10, 50, 100, 150, 200],
        "max_depth": [5, 10, 15, 20],
        "max_leaf_nodes": [10, 20, 30],
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=params,
        scoring="accuracy",
        cv=5,
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    y_pred = grid_search.predict(X_test)
    print(f"Accuracy score = {accuracy_score(y_test, y_pred):.2f}")

    joblib.dump(grid_search, "./titianic/titanic_model.pkl")


if __name__ == "__main__":
    df_train = load_data()
    EDA(df_train)
    X, y = processing(df_train)
    X_train, X_test, y_train, y_test = model(X, y)
    RandomForest(X_train, X_test, y_train, y_test)
