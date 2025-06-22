#importing the necessary dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


#load dataset
data = load_breast_cancer()

#convert to dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)
#add target column
df['target'] = data.target

#explore data
# print(df.head())
print(df['target'].value_counts())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1)
y = df['target']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#training multiple models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

lr_model = LogisticRegression(max_iter=10000)
rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()

#train model
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)  
gb_model.fit(X_train, y_train)

#evaluate the models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#models
models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model
}


for name , model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))
    print("\n")



#saving models and scaler for web app deployment
import os
import joblib

# Make sure 'models/' folder exists
if not os.path.exists('models'):
    os.makedirs('models')

# Save models into 'models' folder
joblib.dump(lr_model, 'models/lr_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(gb_model, 'models/gb_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
