# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import required libraries such as Pandas, Scikit-learn, Seaborn, and Matplotlib.
2. Load the dataset (food_items_binary.csv) using Pandas.
3. Display the dataset and check the available columns.
4. Select the required features (Calories, Total Fat, Saturated Fat, Sugars, Dietary Fiber, Protein).
5. Define the target variable as class.
6. Split the dataset into training and testing sets using train_test_split().
7. Normalize the feature values using StandardScaler.
8. Train the Support Vector Machine (SVM) model using the training data.
9. Predict the class labels for the test dataset.
10. Evaluate the model using accuracy score, classification report, and confusion matrix

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: Sanjeev Kumar K
RegisterNumber:  25012334

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features = ['Calories', 'Total Fat','Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target='class'
X = data[features]
y=data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm=SVC()
param_grid = {
    'C':[0.1,1,10,100],
    'kernel': ['linear','rbf'],
    'gamma': ['scale','auto'] 
}
grid_search = GridSearchCV(svm,param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train,y_train)
best_model = grid_search.best_estimator_
print("Name: Sanjeev Kumar")
print("Register Number: 25012334")
print("Best Parameters:",grid_search.best_params_)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Name: Sanjeev Kumar K")
print("Register Number: 25012334")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test, y_pred))
conf_matrix= confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

*/
```

## Output:
![alt text](<Screenshot 2026-03-10 133608.png>) 
![alt text](<Screenshot 2026-03-10 133627.png>)

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
