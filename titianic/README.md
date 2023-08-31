# 欄位說明

1. **PassengerId**: 乘客的唯一識別號，每個乘客都有一個獨一無二的 ID。

2. **Survived**: 乘客是否倖存，通常用 0 表示未倖存，用 1 表示倖存。

3. **Pclass**: 艙等（Passenger Class），表示乘客所在的艙位等級，可能是 1（頭等艙）、2（二等艙）或 3（三等艙）。

4. **Name**: 乘客的姓名。

5. **Sex**: 乘客的性別，通常用 "male" 表示男性，用 "female" 表示女性。

6. **Age**: 乘客的年齡。

7. **SibSp**: 乘客在船上的兄弟姐妹或配偶數量（Siblings/Spouses Aboard）。

8. **Parch**: 乘客在船上的父母或子女數量（Parents/Children Aboard）。

9. **Ticket**: 船票號碼。

10. **Fare**: 支付的船票費用。

11. **Cabin**: 乘客的艙位號碼。

12. **Embarked**: 登船港口，表示乘客登船的地點，可能是 "C"（Cherbourg）、"Q"（Queenstown）或 "S"（Southampton）。

# 各模型效能
- LogisticRegression Regressor:
  Accuracy score = 0.82

- SVC Regressor:
  Accuracy score = 0.66

- DecisionTreeClassifier Regressor:
  (No accuracy score provided)

- RandomForestClassifier Regressor:
  Accuracy score = 0.85

- KNeighborsClassifier Regressor:
  Accuracy score = 0.69

- XGBClassifier Regressor:
  Accuracy score = 0.83
## 最後選用隨機森林RandomForest，其最佳參數如下
- Best parameters: {'max_depth': 15, 'max_leaf_nodes': 30, 'n_estimators': 100}
- Best cross-validation score: 0.8228593872741555
- Accuracy score = 0.86

# 模型測試
**<span style="color:red">Accuracy score = 0.91</span>**

## 繪製出confusion_matrix可得到下列評估量
- Accuracy: 0.9114832535885168
- Precision: 0.9236641221374046
- Recall: 0.8175675675675675
- Specificity: 0.9629629629629629
- F1-score: 0.8673835125448027