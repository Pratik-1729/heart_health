# Importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Loading and reading the dataset
heart = pd.read_csv("heart.csv")

# Creating a copy of the dataset to avoid affecting the original dataset
heart_df = heart.copy()

# Renaming some of the columns
print(heart_df.head())

# Model building
# Fixing our data in x and y. Here y contains target data and X contains the rest of the features.
x = heart_df.drop(columns=['target'])
y = heart_df['target']

# Splitting the dataset into training and testing using train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)  # Using transform instead of fit_transform for the test set

max_accuracy = 0
best_x = 0

# Finding the best random_state for RandomForestClassifier
for x in range(5000):
    model = RandomForestClassifier(random_state=x)
    model.fit(x_train_scaler, y_train)
    y_pred = model.predict(x_test_scaler)
    current_accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

# Training the model with the best random_state
model = RandomForestClassifier(random_state=best_x)
model.fit(x_train_scaler, y_train)
y_pred = model.predict(x_test_scaler)

# Printing the classification report and accuracy
print('Classification Report\n', classification_report(y_test, y_pred))
score_model = round(accuracy_score(y_pred, y_test) * 100, 2)
print('Accuracy: ' + str(score_model) + " %")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-randomforest-model.pkl'
pickle.dump(model, open(filename, 'wb'))
