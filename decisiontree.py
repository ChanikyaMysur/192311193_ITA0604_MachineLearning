# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Sample Dataset
data = pd.read_csv("/Student_Pass_Fail_Dataset_100.csv")

# Create DataFrame
df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Features and Target
X = df[['Attendance', 'Assignments', 'Internal_Marks']]
y = df['Result']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Decision Tree Model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict New Student
new_student = [[75, 7, 36]]
prediction = model.predict(new_student)

print("\nPrediction for New Student:")
print("Attendance = 75%")
print("Assignments = 7")
print("Internal Marks = 36")

print("Result:", prediction[0])

# Plot Decision Tree
plt.figure(figsize=(12,8))
plot_tree(model,
          feature_names=['Attendance','Assignments','Internal_Marks'],
          class_names=model.classes_,
          filled=True,
          rounded=True)

plt.title("Decision Tree for Pass/Fail Prediction")
plt.show()