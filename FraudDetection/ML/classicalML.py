# streamlit_app.py

import streamlit as st
import pandas as pd
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt


st.set_page_config(page_title="XGBoost Fraud Detector", layout="wide")
st.title("💳 XGBoost Fraud Detection App")

# --- SECTION 1: Load Dataset ---
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\archive\Financial_datasets_log.csv")
    return data

df = load_data()

# --- SECTION 2: Preprocess ---
st.subheader("📊 Raw Data")
st.dataframe(df.head(10))

# Feature selection
feature_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = df[feature_cols]
y = df['isFraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# --- SECTION 3: Train Model ---
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)
joblib.dump((model, scaler), "fraud_model.joblib")

# --- SECTION 4: Prediction & Metrics ---
df['fraud_pred'] = model.predict(X_scaled)
df['fraud_score'] = model.predict_proba(X_scaled)[:, 1]

st.subheader("✅ Prediction Results")
st.dataframe(df[['step', 'type', 'amount', 'fraud_pred', 'fraud_score']].head(10))

st.subheader("📈 Model Metrics")
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC", f"{roc_auc_score(y_test, y_score):.3f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred, zero_division=0))

# --- SECTION 5: Single Transaction Prediction ---
st.subheader("🧾 Predict a Single Transaction")

with st.form("single_txn"):
    amount = st.number_input("Amount", value=1000.0)
    oldbalanceOrg = st.number_input("Old Balance (Sender)", value=10000.0)
    newbalanceOrig = st.number_input("New Balance (Sender)", value=8000.0)
    oldbalanceDest = st.number_input("Old Balance (Receiver)", value=0.0)
    newbalanceDest = st.number_input("New Balance (Receiver)", value=2000.0)
    submit = st.form_submit_button("🚨 Predict")

if submit:
    model, scaler = joblib.load("fraud_model.joblib")
    input_df = pd.DataFrame([{
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"📈 Fraud Probability: {prob:.4f}")
    if pred == 1:
        st.error("⚠️ Transaction is likely FRAUDULENT")
    else:
        st.success("✅ Transaction is likely LEGITIMATE")


    # Plot XGBoost Feature Importance
    model.get_booster().feature_names = X_train.columns.tolist()
    # fig, ax = plt.subplots()
    # plot_importance(model, ax=ax, importance_type='weight')
    # st.pyplot(fig)
    plot_importance(model, importance_type='gain')
    plt.tight_layout()
    plt.show()


    # explainer = shap.Explainer(model)
    # shap_values = explainer(input_scaled)
    #
    # st.subheader("🧠 SHAP Waterfall Explanation")
    # fig = plt.figure(figsize=(10, 5))
    # # shap.plots.waterfall(shap_values[0], show=False)
    # st.pyplot(fig)