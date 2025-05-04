# import os
# import joblib
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Step 1: Load Titanic dataset
# df = sns.load_dataset('titanic')
# df = df.dropna(subset=['age', 'embarked'])
# df = df.drop(columns=['deck'])  # Drop 'deck' due to too many missing values

# # Step 2: Feature selection
# features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
# target = 'survived'

# X = df[features]
# y = df[target]

# # Step 3: Encode categorical features
# X['sex'] = LabelEncoder().fit_transform(X['sex'])
# X['embarked'] = LabelEncoder().fit_transform(X['embarked'])

# # Step 4: Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Train models
# lr_model = LogisticRegression(max_iter=1000)
# lr_model.fit(X_train, y_train)

# dt_model = DecisionTreeClassifier()
# dt_model.fit(X_train, y_train)

# # Step 6: Save models using joblib
# joblib.dump(lr_model, 'logistic_model.pkl')
# joblib.dump(dt_model, 'decision_tree_model.pkl')

# # Step 7: Evaluation
# print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_model.predict(X_test)))
# print("Decision Tree Accuracy:", accuracy_score(y_test, dt_model.predict(X_test)))

# # Step 8: Create the Streamlit app
# # Now, let's set up the Streamlit app

# st.title("Titanic Survival Prediction")

# # File paths
# model_path_lr = 'logistic_model.pkl'
# model_path_dt = 'decision_tree_model.pkl'

# # Check if the model files exist
# if not os.path.exists(model_path_lr) or not os.path.exists(model_path_dt):
#     st.error("Model files not found! Please make sure the models are saved correctly.")
# else:
#     # Load models
#     lr_model = joblib.load(model_path_lr)
#     dt_model = joblib.load(model_path_dt)

#     # Step 9: Create input fields
#     pclass = st.number_input('Pclass (Passenger Class)', min_value=1, max_value=3)
#     sex = st.selectbox('Sex', ['male', 'female'])
#     age = st.number_input('Age', min_value=0.0)
#     sibsp = st.number_input('Siblings/Spouses', min_value=0)
#     parch = st.number_input('Parents/Children', min_value=0)
#     fare = st.number_input('Fare', min_value=0.0)
#     embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

#     # Encode inputs
#     sex = 1 if sex == 'male' else 0
#     embarked = {'C': 0, 'Q': 1, 'S': 2}[embarked]

#     # Prepare features
#     features = [[pclass, sex, age, sibsp, parch, fare, embarked]]

#     # Prediction
#     if st.button('Predict'):
#         pred_lr = lr_model.predict(features)[0]
#         pred_dt = dt_model.predict(features)[0]

#         st.write("Logistic Regression Prediction: ", "Survived" if pred_lr == 1 else "Not Survived")
#         st.write("Decision Tree Prediction: ", "Survived" if pred_dt == 1 else "Not Survived")
import streamlit as st
import pandas as pd
import joblib
import os

# Load models
model_path_lr = 'logistic_model.pkl'
model_path_dt = 'decision_tree_model.pkl'

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

# Apply custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f4f6f7;
            font-family: Arial, sans-serif;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            font-size: 28px;
            margin-bottom: 20px;
        }
        .form-container {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 400px;
            margin: auto;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        h3 {
            color: #27ae60;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚢 Titanic Survival Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="form-container">', unsafe_allow_html=True)

# Input form
with st.form("input_form"):
    pclass = st.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x == 1 else f"{x}nd Class" if x == 2 else f"{x}rd Class")
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.number_input("Age", min_value=0.0, step=1.0)
    sibsp = st.number_input("Siblings/Spouse", min_value=0, step=1)
    parch = st.number_input("Parents/Children", min_value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, step=1.0)
    embarked = st.selectbox("Embarked From", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
    submitted = st.form_submit_button("Predict")

st.markdown('</div>', unsafe_allow_html=True)

# Predict if submitted
if submitted:
    if not os.path.exists(model_path_lr) or not os.path.exists(model_path_dt):
        st.error("Model files not found. Please make sure they are saved in the same directory.")
    else:
        # Encoding
        sex = 1 if sex.lower() == 'male' else 0
        embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
        embarked = embarked_map[embarked]

        # Prepare input
        input_data = [[pclass, sex, age, sibsp, parch, fare, embarked]]

        # Load models
        lr_model = joblib.load(model_path_lr)
        dt_model = joblib.load(model_path_dt)

        # Make predictions
        pred_lr = lr_model.predict(input_data)[0]
        pred_dt = dt_model.predict(input_data)[0]

        st.markdown("### 🎯 Predictions:")
        st.success(f"**Logistic Regression:** {'Survived' if pred_lr == 1 else 'Not Survived'}")
        st.success(f"**Decision Tree:** {'Survived' if pred_dt == 1 else 'Not Survived'}")
