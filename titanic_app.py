
# import streamlit as st
# import pandas as pd
# import joblib
# import os

# # Load models
# model_path_lr = 'logistic_model.pkl'
# model_path_dt = 'decision_tree_model.pkl'

# st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

# st.title("Titanic Survival Predictor")

# # Input form
# with st.form("input_form"):
#     pclass = st.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x == 1 else f"{x}nd Class" if x == 2 else f"{x}rd Class")
#     sex = st.selectbox("Sex", ["Female", "Male"])
#     age = st.number_input("Age", min_value=0.0, step=1.0)
#     sibsp = st.number_input("Siblings/Spouse", min_value=0, step=1)
#     parch = st.number_input("Parents/Children", min_value=0, step=1)
#     fare = st.number_input("Fare", min_value=0.0, step=1.0)
#     embarked = st.selectbox("Embarked From", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
#     submitted = st.form_submit_button("Predict")

# # Predict if submitted
# if submitted:
#     if not os.path.exists(model_path_lr) or not os.path.exists(model_path_dt):
#         st.error("Model files not found. Please make sure they are saved in the same directory.")
#     else:
#         # Encoding
#         sex = 1 if sex.lower() == 'male' else 0
#         embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
#         embarked = embarked_map[embarked]

#         # Prepare input
#         input_data = [[pclass, sex, age, sibsp, parch, fare, embarked]]

#         # Load models
#         lr_model = joblib.load(model_path_lr)
#         dt_model = joblib.load(model_path_dt)

#         # Make predictions
#         pred_lr = lr_model.predict(input_data)[0]
#         pred_dt = dt_model.predict(input_data)[0]

#         st.markdown("###  Predictions:")
#         st.success(f"**Logistic Regression:** {'Survived' if pred_lr == 1 else 'Not Survived'}")
#         st.success(f"**Decision Tree:** {'Survived' if pred_dt == 1 else 'Not Survived'}")
import streamlit as st
import pandas as pd
import joblib
import os

# Load models
model_path_lr = 'logistic_model.pkl'
model_path_dt = 'decision_tree_model.pkl'

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("Titanic Survival Predictor")

# Input form with the clear_on_submit parameter
with st.form("input_form", clear_on_submit=True):
    pclass = st.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x == 1 else f"{x}nd Class" if x == 2 else f"{x}rd Class")
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.number_input("Age", min_value=0.0, step=1.0)
    sibsp = st.number_input("Siblings/Spouse", min_value=0, step=1)
    parch = st.number_input("Parents/Children", min_value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, step=1.0)
    embarked = st.selectbox("Embarked From", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
    submitted = st.form_submit_button("Predict")

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

        st.markdown("###  Predictions:")
        st.success(f"**Logistic Regression:** {'Survived' if pred_lr == 1 else 'Not Survived'}")
        st.success(f"**Decision Tree:** {'Survived' if pred_dt == 1 else 'Not Survived'}")

        # Form will be cleared automatically due to 'clear_on_submit=True'
