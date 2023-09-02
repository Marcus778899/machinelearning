import pandas as pd
import joblib

column = "Job Title,Employment Type,Experience Level,Expertise Level,Salary,Salary Currency,Company Location,Salary in USD,Employee Residence,Company Size,Year"
data = "Data Scientist,Full-Time,Senior,Expert,130000,United States Dollar,United States,130000,United States,Medium,2023"

column = column.split(",")
data = data.split(",")
df = pd.DataFrame([data], columns=column)

preprocessor = joblib.load("./Data_Science_Salary/label.pkl")
label_cloumn = [
    "Experience Level",
    "Expertise Level",
    "Company Size",
    "Job Title",
    "Employment Type",
    "Salary Currency",
    "Company Location",
]

encode_data = pd.DataFrame(
    preprocessor.transform(df[label_cloumn]).toarray(),
    columns=preprocessor.get_feature_names_out(label_cloumn),
)

df = pd.concat([df[["Year", "Salary in USD"]], encode_data], axis=1)

X = df.drop("Salary in USD", axis=1)
y = df["Salary in USD"]

model = joblib.load("./Data_Science_Salary/model_final.pkl")
predictions = model.predict(X)
predictions = pd.DataFrame({"Predicted Salary": predictions, "Real Salary": y})
print(predictions)
