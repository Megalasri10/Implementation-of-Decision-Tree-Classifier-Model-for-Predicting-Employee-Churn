# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Get the independent variable X and dependent variable Y.
2.Calculate the mean of the X -values and the mean of the Y -values.
3.Find the slope m of the line of best fit using the formula.
image
<img width="296" height="134" alt="Screenshot 2026-04-24 142329" src="https://github.com/user-attachments/assets/00ec297a-73f3-4933-b6ba-fdc120b1a3ca" />

4. Compute the y -intercept of the line by using the formula:
image
<img width="209" height="51" alt="Screenshot 2026-04-24 142402" src="https://github.com/user-attachments/assets/932e5671-5bd9-4e4b-95b5-6bf41081c6d2" />

5. Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b and plot the scatterplot.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Megala M S
RegisterNumber:  212225040230
# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 2: Create Dataset
data = {
    "Age": [25, 30, 35, 40, 28, 32, 45, 50, 29, 38],
    "Salary": [30000, 40000, 50000, 60000, 35000,
               45000, 70000, 80000, 38000, 52000],
    "Years_at_Company": [1, 3, 5, 10, 2, 4, 12, 15, 2, 6],
    "Churn": [1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display Dataset
print("Dataset:\n")
print(df)

# Step 3: Split Features and Target
X = df[["Age", "Salary", "Years_at_Company"]]
y = df["Churn"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Decision Tree Classifier
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Evaluation:")

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(importance)

# Step 9: Visualize Decision Tree
plt.figure(figsize=(12,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True
)

plt.title("Decision Tree for Employee Churn Prediction")
plt.show()

# Step 10: Custom Prediction
employee_data = [[30, 45000, 3]]

prediction = model.predict(employee_data)

print("\nCustom Employee Prediction:")

if prediction[0] == 1:
    print("Employee is Likely to Leave the Company")
else:
    print("Employee is Likely to Stay in the Company")
*/
```

## Output:
![decision tree classifier model](sam.png)

<img width="557" height="362" alt="image" src="https://github.com/user-attachments/assets/006b771b-5606-4f7f-8d60-b5f037bf6889" />
<img width="782" height="333" alt="image" src="https://github.com/user-attachments/assets/f7672179-9406-4038-8e7f-c2de0eabb9a6" />
<img width="1247" height="802" alt="image" src="https://github.com/user-attachments/assets/ddd8cacc-67f8-4ceb-82f9-ed33da927851" />
<img width="506" height="122" alt="image" src="https://github.com/user-attachments/assets/965ebc71-4a46-450d-8884-c64d8aa0c483" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
