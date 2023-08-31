import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("./titianic/test.csv", header=0)

model = joblib.load("./titianic/titanic_model.pkl")
df_ans = pd.read_csv("./titianic/submission.csv", header=0)
df.drop(["Name", "Ticket", "Cabin"], inplace=True, axis=1)
df.Age.fillna(df.Age.mode()[0], inplace=True)
df.Fare.fillna(df.Fare.mode()[0], inplace=True)


encoder = OneHotEncoder()
X = df[["Age", "Fare", "Pclass", "SibSp", "Parch"]]
encode_data = encoder.fit_transform(df[["Sex", "Embarked"]])
encode_data = pd.DataFrame(
    encode_data.toarray(),
    columns=encoder.get_feature_names_out(["Sex", "Embarked"]),
)
X = pd.concat([X, encode_data], axis=1)
y = df_ans.Survived.values

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
print(f"Accuracy score = {accuracy:.2f}")

# 混淆矩陣

conf_matrix = confusion_matrix(y, y_pred)

conf_matrix_df = pd.DataFrame(
    conf_matrix, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]
)
print(conf_matrix_df)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix Heatmap")
plt.show()

TP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]

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
