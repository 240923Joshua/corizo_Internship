import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    data = pd.read_csv(r"C:\Users\PROBOOK\Downloads\cardio.csv", sep=';')
    
    data['age'] = (data['age'] / 365).round().astype(int)
    
    data = data[
        (data['height'] > 120) & (data['height'] < 200) &
        (data['weight'] > 30) & (data['weight'] < 200) &
        (data['ap_hi'] >= 50) & (data['ap_hi'] <= 250) &
        (data['ap_lo'] >= 50) & (data['ap_lo'] <= 200) &
        (data['ap_lo'] <= data['ap_hi'])
    ]
    
    plt.figure(figsize=(6,4))
    plt.hist(data['age'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.show()
    
    plt.figure(figsize=(6,4))
    plt.hist(data['height'], bins=20, color='lightgreen', edgecolor='black')
    plt.title('Height Distribution')
    plt.xlabel('Height (cm)')
    plt.ylabel('Count')
    plt.show()
    
    plt.figure(figsize=(6,4))
    plt.hist(data['weight'], bins=20, color='salmon', edgecolor='black')
    plt.title('Weight Distribution')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Count')
    plt.show()
    
    plt.figure(figsize=(6,4))
    plt.hist(data['ap_hi'], bins=20, color='orchid', edgecolor='black')
    plt.title('Systolic Blood Pressure (ap_hi) Distribution')
    plt.xlabel('ap_hi (systolic)')
    plt.ylabel('Count')
    plt.show()
    
    plt.figure(figsize=(6,4))
    plt.hist(data['ap_lo'], bins=20, color='gold', edgecolor='black')
    plt.title('Diastolic Blood Pressure (ap_lo) Distribution')
    plt.xlabel('ap_lo (diastolic)')
    plt.ylabel('Count')
    plt.show()
    
    plt.figure(figsize=(6,4))
    sns.countplot(x='cardio', data=data, palette='pastel')
    plt.title('Count of Cardiovascular Disease (cardio)')
    plt.xlabel('Cardio (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.show()
    
    features = data.drop(columns=['id', 'cardio'])
    corr_matrix = features.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Features')
    plt.show()
    
    X = data.drop(columns=['id', 'cardio'])
    y = data['cardio']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5)),
        ('Support Vector Machine (Linear)', LinearSVC(max_iter=10000, random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42))
    ]
    
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.2f}")
    
if __name__ == "__main__":
    main()

