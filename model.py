#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# load dataset
df = pd.read_csv(r"C:\Users\HW\Desktop\Diabetes prediction\diabetes.csv")

print(df.head())

#EDA (exploratory data analysis)
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()

#Data visualization
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

#Target value analysis
print(df["Outcome"].value_counts())

#Define features and target
x = df.drop("Outcome", axis=1)
y = df["Outcome"]

#Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Machine Learning Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

#make prediction
y_pred = model.predict(x_test)
print("Predictions:", y_pred)

#Evaulate model performance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#Confusion Matrix Visualization
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    model, x_test, y_test,
    cmap="Blues",
    display_labels=["Non-Diabetic", "Diabetic"])
plt.show()

 
#save the model
import joblib

joblib.dump(model, "diabetes_model.pkl")
print("Model saved successfully")

#Load Model and Predict New Patient
import joblib

#Load Model
loaded_model = joblib.load("diabetes_model.pkl")

#New patient data
new_data = [[2,120,70,30,100,25.5,0.5,33]]

#Prediction
prediction = loaded_model.predict(new_data)

print("Prediction:", "Diabetic" if prediction[0]==1 else "Non-Diabetic")
 
