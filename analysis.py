# Customer Churn Analysis Project
# Internship Task - Python Code

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("customer_churn.csv")

# Display First 5 Rows
print(df.head())

# Dataset Information
print(df.info())

# Check Missing Values
print(df.isnull().sum())

# Remove Missing Values
df = df.dropna()

# Convert Categorical Columns into Numerical
label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# Target Column
# Assuming "Churn" column contains Yes/No values
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Factors Affecting Churn:\n")
print(feature_importance)

# Plot Feature Importance
plt.figure(figsize=(10,6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance for Customer Churn")
plt.show()

# Churn Rate Visualization
plt.figure(figsize=(5,5))
df['Churn'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['skyblue', 'orange']
)

plt.title("Customer Churn Distribution")
plt.ylabel("")
plt.show()

# Retention Suggestions
print("\nRetention Improvement Suggestions:")
print("1. Increase customer engagement through personalized offers.")
print("2. Improve customer support response time.")
print("3. Provide loyalty rewards for long-term users.")
print("4. Target low activity users with re-engagement campaigns.")
print("5. Reduce subscription pricing issues with flexible plans.")
