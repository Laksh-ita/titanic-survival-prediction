import os
import joblib
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Titanic dataset
df = sns.load_dataset('titanic')
df = df.dropna(subset=['age', 'embarked'])
df = df.drop(columns=['deck'])  # Drop 'deck' due to too many missing values

# Step 2: Feature selection
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

X = df[features]
y = df[target]

# Step 3: Encode categorical features
X['sex'] = LabelEncoder().fit_transform(X['sex'])
X['embarked'] = LabelEncoder().fit_transform(X['embarked'])

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train models
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Step 6: Save models using joblib
joblib.dump(lr_model, 'logistic_model.pkl')
joblib.dump(dt_model, 'decision_tree_model.pkl')

# Step 7: Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_model.predict(X_test)))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_model.predict(X_test)))

# Step 8: Create the Streamlit app
# Now, let's set up the Streamlit app

st.title("Titanic Survival Prediction")

# File paths
model_path_lr = 'logistic_model.pkl'
model_path_dt = 'decision_tree_model.pkl'

# Check if the model files exist
if not os.path.exists(model_path_lr) or not os.path.exists(model_path_dt):
    st.error("Model files not found! Please make sure the models are saved correctly.")
else:
    # Load models
    lr_model = joblib.load(model_path_lr)
    dt_model = joblib.load(model_path_dt)

    # Step 9: Create input fields
    pclass = st.number_input('Pclass (Passenger Class)', min_value=1, max_value=3)
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.number_input('Age', min_value=0.0)
    sibsp = st.number_input('Siblings/Spouses', min_value=0)
    parch = st.number_input('Parents/Children', min_value=0)
    fare = st.number_input('Fare', min_value=0.0)
    embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

    # Encode inputs
    sex = 1 if sex == 'male' else 0
    embarked = {'C': 0, 'Q': 1, 'S': 2}[embarked]

    # Prepare features
    features = [[pclass, sex, age, sibsp, parch, fare, embarked]]

    # Prediction
    if st.button('Predict'):
        pred_lr = lr_model.predict(features)[0]
        pred_dt = dt_model.predict(features)[0]

        st.write("Logistic Regression Prediction: ", "Survived" if pred_lr == 1 else "Not Survived")
        st.write("Decision Tree Prediction: ", "Survived" if pred_dt == 1 else "Not Survived")
