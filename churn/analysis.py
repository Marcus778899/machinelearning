import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimSun']
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

df = pd.read_csv("./churn/customer_churn_data.csv")
print(df.head())
print('-' * 50)


def data_structure():
    # 型別轉換(SeniorCitizen,TotalCharges)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
    print(df.TotalCharges.value_counts())
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(df.info())
    print('-' * 100)
    print(df.describe().iloc[1, :])
    print('-' * 100)
    return df


def EDA():
    # 不同的合約期間，停留月數和付費情況是否有差異
    columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    print(df.groupby('Contract')[columns].mean().reset_index())
    print('-' * 100)

    # 不同性別&contract，人數是否有所差異(看來是沒有)
    print(df.pivot_table(index=['gender', 'Contract'],
          values='customerID', aggfunc='count'))
    print('-' * 100)


def processing(df):
    print(df.isna().sum())
    df.dropna(inplace=True)

    # onehotencodeing
    df = df.iloc[:, 1:]  # customer_id移除
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
    df_dummy = pd.get_dummies(df)
    print(df_dummy.head())
    print('-' * 100)
    return df_dummy


def DescriptiveStatistics(df):
    colors = ['blue', 'orange']
    # 查看名單內性別比例
    ax = (df['gender'].value_counts()*100.0 / len(df)
          ).plot(kind='bar', stacked=True, rot=0, color=colors)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel('% Customers')
    ax.set_xlabel('Gender')
    ax.set_ylabel('% Customers')
    ax.set_title('Gender Distribution')

    totals = []

    for i in ax.patches:
        totals.append(i.get_width())

    total = sum(totals)

    for i in ax.patches:
        ax.text(i.get_x()+.15, i.get_height()-3.5,
                str(round((i.get_height()/total), 1))+'%',
                fontsize=12,
                color='white',
                weight='bold')
    plt.show()

    # 查看老年人比例
    plt.pie(df['SeniorCitizen'].value_counts()*100/len(df),
            labels=['NO', 'YES'],
            autopct='%1.1f%%',
            colors=colors)
    plt.ylabel('SeniorCitizen')
    plt.title('SeniorCitizen Distribution')
    plt.show()

    # 查看客戶停留月數對續約的關聯
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, sharey=True, figsize=(20, 6))

    ax = sns.distplot(df[df['Contract'] == 'Month-to-month']['tenure'],
                      hist=True, kde=False,
                      bins=int(180/5), color='turquoise',
                      hist_kws={'edgecolor': 'black'},
                      kde_kws={'linewidth': 4},
                      ax=ax1)
    ax.set_ylabel('Customers count')
    ax.set_xlabel('Tenure (months)')
    ax.set_title('Month to Month Contract')

    ax = sns.distplot(df[df['Contract'] == 'One year']['tenure'],
                      hist=True, kde=False,
                      bins=int(180/5), color='steelblue',
                      hist_kws={'edgecolor': 'black'},
                      kde_kws={'linewidth': 4},
                      ax=ax2)
    ax.set_xlabel('Tenure (months)', size=14)
    ax.set_title('One Year Contract', size=14)

    ax = sns.distplot(df[df['Contract'] == 'Two year']['tenure'],
                      hist=True, kde=False,
                      bins=int(180/5), color='darkblue',
                      hist_kws={'edgecolor': 'black'},
                      kde_kws={'linewidth': 4},
                      ax=ax3)

    ax.set_xlabel('Tenure (months)')
    ax.set_title('Two Year Contract')
    plt.show()

    # 箱形圖
    sns.boxplot(x=df.Churn, y=df.tenure)
    plt.show()


def modeling():
    # 標準化
    X = df_dummy.drop('Churn', axis=1)
    y = df_dummy['Churn']
    features = X.columns
    print(len(features))
    standarscaler = MinMaxScaler()
    standarscaler.fit(X)
    X = standarscaler.transform(X)
    print(X.shape)
    print(y.shape)

    # 分割測試集和訓練集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print('-' * 100)

    # 列出所有可用模型
    regressors = [("LogisticRegression", LogisticRegression()), ("SVC", SVC(probability=True)), ("DecisionTreeClassifier", DecisionTreeClassifier(
    )), ("RandomForestClassifier", RandomForestClassifier()), ("XGBClassifier", XGBClassifier()), ("KNeighborsClassifier", KNeighborsClassifier())]

    models = []
    for name, classifer in regressors:
        print(f"\n{name} Regressor:")
        classifer.fit(X_train, y_train)
        models.append((name, classifer))
        pred = classifer.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        print(f"Accuracy score = {accuracy:.2f}")

    # ROC Curve
    plt.figure(figsize=(10, 8))
    for name, model in models:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

    return X_train, X_test, y_train, y_test


def logistic():
    print(f'\nLogistic Regression Regressor:')
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}
    gridSearch = GridSearchCV(estimator=LogisticRegression(
    ), param_grid=params, scoring='accuracy', cv=5)
    gridSearch.fit(X_train, y_train)
    print("Best parameters:", gridSearch.best_params_)
    print("Best cross-validation score:", gridSearch.best_score_)
    y_pred = gridSearch.predict(X_test)
    param['LogisticRegression'] = gridSearch.best_params_
    accuracy['LogisticRegression'] = accuracy_score(y_test, y_pred)


def svc():
    print(f'\nSVC Regressor:')
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    gridSearch = GridSearchCV(
        estimator=SVC(), param_grid=params, scoring='accuracy', cv=5)
    gridSearch.fit(X_train, y_train)
    print("Best parameters:", gridSearch.best_params_)
    print("Best cross-validation score:", gridSearch.best_score_)
    y_pred = gridSearch.predict(X_test)
    param['SVC'] = gridSearch.best_params_
    accuracy['SVC'] = accuracy_score(y_test, y_pred)


def decisiontree():
    print(f'\nDescision Tree Regressor:')
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7,
                            8, 9, 10], 'min_samples_split': [2, 5, 10]}
    gridSearch = GridSearchCV(estimator=DecisionTreeClassifier(
    ), param_grid=params, scoring='accuracy', cv=5)
    gridSearch.fit(X_train, y_train)
    print("Best parameters:", gridSearch.best_params_)
    print("Best cross-validation score:", gridSearch.best_score_)
    y_pred = gridSearch.predict(X_test)
    param['DecisionTreeClassifier'] = gridSearch.best_params_
    accuracy['DecisionTreeClassifier'] = accuracy_score(y_test, y_pred)


def randomforest():
    print(f'\nRandom Forest Regressor:')
    params = {'n_estimators': [10, 50, 100, 150, 200],
              'max_depth': [5, 10, 15, 20],
              'max_leaf_nodes': [10, 20, 30]}
    gridSearch = GridSearchCV(estimator=RandomForestClassifier(
    ), param_grid=params, scoring='accuracy', cv=5)
    gridSearch.fit(X_train, y_train)
    print("Best parameters:", gridSearch.best_params_)
    print("Best cross-validation score:", gridSearch.best_score_)
    y_pred = gridSearch.predict(X_test)
    param['RandomForestClassifier'] = gridSearch.best_params_
    accuracy['RandomForestClassifier'] = accuracy_score(y_test, y_pred)


def xgboost():
    print(f'\nXgboost Regressor:')
    params = {'n_estimators': [10, 50, 100, 150, 200], 'max_depth': [
        5, 10, 15, 20], 'learning_rate': [0.01, 0.1, 0.2]}
    gridSearch = GridSearchCV(estimator=XGBClassifier(
    ), param_grid=params, scoring='accuracy', cv=5)
    gridSearch.fit(X_train, y_train)
    print("Best parameters:", gridSearch.best_params_)
    print("Best cross-validation score:", gridSearch.best_score_)
    y_pred = gridSearch.predict(X_test)
    param['XGBClassifier'] = gridSearch.best_params_
    accuracy['XGBClassifier'] = accuracy_score(y_test, y_pred)


def kneighbors():
    print(f'\nKneighbors Regressor:')
    params = {'n_neighbors': [3, 5, 7, 10], 'weights': [
        'uniform', 'distance'], 'p': [1, 2]}
    gridSearch = GridSearchCV(estimator=KNeighborsClassifier(
    ), param_grid=params, scoring='accuracy', cv=5)
    gridSearch.fit(X_train, y_train)
    print("Best parameters:", gridSearch.best_params_)
    print("Best cross-validation score:", gridSearch.best_score_)
    y_pred = gridSearch.predict(X_test)
    param['KNeighborsClassifier'] = gridSearch.best_params_
    accuracy['KNeighborsClassifier'] = accuracy_score(y_test, y_pred)


def model_test(model):
    y_pred = model.predict(X_test)
    print(f'\n{model.__class__.__name__}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    # 混淆矩陣
    conf = confusion_matrix(y_test, y_pred)
    cond_df = pd.DataFrame(
        conf, index=['no Churn', 'Churn'], columns=['NO', "YES"])
    print(cond_df)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cond_df, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f'{model.__class__.__name__} Confusion Matrix')
    plt.show()
    TP = cond_df.iloc[1, 1]
    FP = cond_df.iloc[0, 1]
    TN = cond_df.iloc[0, 0]
    FN = cond_df.iloc[1, 0]

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F1-score:", f1_score)


if __name__ == '__main__':
    df = data_structure()
    EDA()
    DescriptiveStatistics(df)
    df_dummy = processing(df)
    X_train, X_test, y_train, y_test = modeling()
    param = {}
    accuracy = {}
    logistic()
    svc()
    decisiontree()
    randomforest()
    xgboost()
    kneighbors()
    print(param)
    print(accuracy)

    model_selection = LogisticRegression(**param['LogisticRegression'])
    model = model_selection.fit(X_train, y_train)

    model_test(model)
    joblib.dump(model, './churn/logistic_regression.pkl')
