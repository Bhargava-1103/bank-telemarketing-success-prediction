import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load preprocessor and model
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("model (2).pkl", "rb") as f:
    model = pickle.load(f)

# Feature Engineering
def create_features(df):
    df = df.copy()
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 20, 30, 40, 50, 60, 100],
        labels=['0-20', '21-30', '31-40', '41-50', '51-60', '60+']
    )

    try:
        bins = pd.qcut(df['balance'], q=5, retbins=True, duplicates='drop')[1]
        labels = ['very_low', 'low', 'medium', 'high', 'very_high'][:len(bins) - 1]
        df['balance_group'] = pd.cut(df['balance'], bins=bins, labels=labels, include_lowest=True)
    except Exception:
        df['balance_group'] = 'unknown'

    df['campaign_intensity'] = df['campaign'] / (df['pdays'].replace(-1, 999) + 1)
    df['campaign_intensity'] = df['campaign_intensity'].clip(upper=df['campaign_intensity'].quantile(0.99))

    df['contact_rate'] = df['previous'] / (df['pdays'].replace(-1, 999) + 1)
    df['contact_rate'] = df['contact_rate'].clip(upper=df['contact_rate'].quantile(0.99))

    df['age_balance'] = df['age'] * df['balance']
    df['age_balance'] = df['age_balance'].clip(upper=df['age_balance'].quantile(0.99))

    df['duration_campaign'] = df['duration'] * df['campaign']
    df['duration_campaign'] = df['duration_campaign'].clip(upper=df['duration_campaign'].quantile(0.99))

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill')  # Fallback to avoid NaN errors

    return df

# Streamlit UI
st.title("Bank Term Deposit Subscription Predictor")
st.markdown("Predict whether a client will subscribe to a term deposit based on campaign data.")

with st.form("prediction_form"):
    st.subheader("Enter Customer Details:")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician',
        'unemployed', 'unknown'
    ])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.selectbox("Credit in Default?", ['yes', 'no'])
    balance = st.number_input("Balance", value=0)
    housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
    loan = st.selectbox("Has Personal Loan?", ['yes', 'no'])
    contact = st.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])
    month = st.selectbox("Month of Contact", [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ])
    weekday = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
    poutcome = st.selectbox("Previous Campaign Outcome", ['failure', 'nonexistent', 'success'])
    duration = st.number_input("Last Call Duration (seconds)", value=100)
    campaign = st.number_input("Contacts During This Campaign", value=1)
    pdays = st.number_input("Days Since Last Contact (-1 = never)", value=-1)
    previous = st.number_input("Previous Campaign Contacts", value=0)

    submit = st.form_submit_button("Predict")

if submit:
    try:
        user_input = pd.DataFrame([{
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'weekday': weekday,
            'poutcome': poutcome,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous
        }])

        features = create_features(user_input)
        X = preprocessor.transform(features)
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"The client is **likely** to subscribe. Confidence: {confidence:.2%}")
        else:
            st.warning(f"The client is **unlikely** to subscribe. Confidence: {(1 - confidence):.2%}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
