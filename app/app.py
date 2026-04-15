"""💳 Credit Card Fraud Detection Dashboard"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import requests
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fraud Detection AI", page_icon="💳", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 2.8rem; color: #1F3864; text-align: center; font-weight: 800; margin-bottom: 0; }
    .sub-header { text-align: center; color: #4472C4; font-size: 1.1rem; margin-top: 0; }
    .fraud-alert { background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #e74c3c; color: #c62828; }
    .safe-alert { background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #2ecc71; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.pkl')
    rf = joblib.load('models/random_forest.pkl')
    xgb = joblib.load('models/xgboost.pkl')
    nn = keras.models.load_model('models/neural_network.keras')
    rf_res = joblib.load('models/rf_results.pkl')
    xgb_res = joblib.load('models/xgb_results.pkl')
    nn_res = joblib.load('models/nn_results.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return scaler, rf, xgb, nn, rf_res, xgb_res, nn_res, explainer, feature_names

@st.cache_data
def load_test_samples():
    X_test = joblib.load('data/processed/X_test.pkl')
    y_test = joblib.load('data/processed/y_test.pkl')
    return X_test[y_test == 1].values, X_test[y_test == 0].values

scaler, rf, xgb, nn, rf_res, xgb_res, nn_res, explainer, feature_names = load_models()
fraud_samples, legit_samples = load_test_samples()

st.sidebar.image("https://img.icons8.com/fluency/96/fraud.png", width=80)
st.sidebar.title("💳 Fraud AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🎯 Live Prediction", "🔌 API Mode", "📊 Model Performance", "📈 Dataset Explorer", "ℹ️ About"])
st.sidebar.markdown("---")
st.sidebar.info("Built with Python, Scikit-learn, XGBoost, TensorFlow, SHAP, FastAPI & Streamlit")


if page == "🏠 Home":
    st.markdown('<p class="main-header">💳 Credit Card Fraud Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time fraud scoring with explainable AI + REST API</p>', unsafe_allow_html=True)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Best ROC-AUC", f"{xgb_res['roc_auc']:.4f}", "XGBoost")
    c2.metric("🚨 Fraud Caught", f"{xgb_res['recall']*100:.1f}%", "Recall")
    c3.metric("✅ Precision", f"{xgb_res['precision']*100:.1f}%", "When flagged")
    c4.metric("📊 Transactions", "284,807", "Training dataset")
    st.markdown("---")
    a, b = st.columns(2)
    with a:
        st.subheader("🎓 What This Is")
        st.markdown("End-to-end ML system: 3 models, SHAP explainability, FastAPI backend, Streamlit frontend.\n\n**Tech Stack:** Python, Scikit-learn, XGBoost, TensorFlow, SMOTE, SHAP, FastAPI, Streamlit")
    with b:
        st.subheader("⚡ Why It Matters")
        st.markdown("- **$30+ billion** lost globally yearly\n- Catches **85%+ of fraud**\n- **Explainable** decisions (SHAP)\n- **Production architecture** (REST API + UI)")
    st.markdown("---")
    st.subheader("📊 Model Performance at a Glance")
    comp_df = pd.DataFrame({'Model': ['Random Forest', 'XGBoost', 'Neural Network'],
        'Accuracy': [rf_res['accuracy'], xgb_res['accuracy'], nn_res['accuracy']],
        'Precision': [rf_res['precision'], xgb_res['precision'], nn_res['precision']],
        'Recall': [rf_res['recall'], xgb_res['recall'], nn_res['recall']],
        'F1': [rf_res['f1_score'], xgb_res['f1_score'], nn_res['f1_score']],
        'ROC-AUC': [rf_res['roc_auc'], xgb_res['roc_auc'], nn_res['roc_auc']]})
    st.dataframe(comp_df.style.format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1': '{:.4f}', 'ROC-AUC': '{:.4f}'}).background_gradient(cmap='RdYlGn', subset=['Accuracy','Precision','Recall','F1','ROC-AUC']), use_container_width=True)


elif page == "🎯 Live Prediction":
    st.markdown('<p class="main-header">🎯 Live Fraud Detection</p>', unsafe_allow_html=True)
    st.markdown("Load a REAL transaction — the AI scores it and **explains why**.")
    st.markdown("---")
    if 'tx' not in st.session_state:
        st.session_state['tx'] = legit_samples[0]
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📝 Pick a Real Transaction")
        b1, b2 = st.columns(2)
        if b1.button("🟢 Load Real LEGIT Transaction", use_container_width=True):
            idx = np.random.randint(0, len(legit_samples))
            st.session_state['tx'] = legit_samples[idx]
            st.success(f"Loaded LEGIT transaction #{idx}")
        if b2.button("🔴 Load Real FRAUD Transaction", use_container_width=True):
            idx = np.random.randint(0, len(fraud_samples))
            st.session_state['tx'] = fraud_samples[idx]
            st.error(f"Loaded FRAUD transaction #{idx}")
        st.markdown("**Transaction feature values (first 10):**")
        feat_vals = [f"{v:.3f}" for v in st.session_state['tx'][:10]]
        st.dataframe(pd.DataFrame({'Feature': feature_names[:10], 'Value': feat_vals}), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("🤖 Model Choice")
        model_choice = st.selectbox("Pick the model:", ["XGBoost (🏆 best)", "Random Forest", "Neural Network", "Ensemble (all 3 voting)"])
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)
        predict_btn = st.button("🔍 ANALYZE TRANSACTION", type="primary", use_container_width=True)
    st.markdown("---")
    if predict_btn:
        X = np.array(st.session_state['tx']).reshape(1, -1)
        if model_choice.startswith("XGBoost"):
            prob = xgb.predict_proba(X)[0][1]
            used = "XGBoost"
        elif model_choice.startswith("Random"):
            prob = rf.predict_proba(X)[0][1]
            used = "Random Forest"
        elif model_choice.startswith("Neural"):
            prob = float(nn.predict(X, verbose=0)[0][0])
            used = "Neural Network"
        else:
            p1 = xgb.predict_proba(X)[0][1]
            p2 = rf.predict_proba(X)[0][1]
            p3 = float(nn.predict(X, verbose=0)[0][0])
            prob = (p1 + p2 + p3) / 3
            used = f"Ensemble (XGB={p1:.3f}, RF={p2:.3f}, NN={p3:.3f})"
        is_fraud = prob >= threshold
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            if is_fraud:
                st.markdown(f'<div class="fraud-alert"><h2>🚨 FRAUD DETECTED</h2><h3>Probability: {prob*100:.2f}%</h3><p>Recommend blocking.</p><small>Model: {used} | Threshold: {threshold}</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-alert"><h2>✅ LEGITIMATE</h2><h3>Fraud probability: {prob*100:.2f}%</h3><p>Approve.</p><small>Model: {used} | Threshold: {threshold}</small></div>', unsafe_allow_html=True)
        with cc2:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, title={'text': "Fraud Risk Score (%)"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#e74c3c" if is_fraud else "#2ecc71"},
                    'steps': [{'range': [0, 30], 'color': "#d4edda"}, {'range': [30, 70], 'color': "#fff3cd"}, {'range': [70, 100], 'color': "#f8d7da"}],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': threshold * 100}}))
            fig.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("🔍 Why did the model decide this?")
        st.caption("SHAP values show which features pushed the decision.")
        with st.spinner("Computing SHAP values..."):
            shap_values = explainer.shap_values(X)
        contributions = pd.DataFrame({'Feature': feature_names, 'Value': X[0], 'SHAP': shap_values[0]})
        contributions['AbsSHAP'] = contributions['SHAP'].abs()
        top_10 = contributions.nlargest(10, 'AbsSHAP').sort_values('SHAP')
        fig_shap = go.Figure(go.Bar(x=top_10['SHAP'], y=top_10['Feature'], orientation='h',
            marker_color=['#e74c3c' if s > 0 else '#2ecc71' for s in top_10['SHAP']],
            text=[f"{v:+.3f}" for v in top_10['SHAP']], textposition='auto'))
        fig_shap.update_layout(title="Top 10 Features Driving This Decision",
            xaxis_title="SHAP value (positive = fraud)", height=450)
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown("##### 💬 Plain English Explanation")
        if is_fraud:
            drivers = contributions[contributions['SHAP'] > 0].nlargest(3, 'SHAP')
            explanation = "**🚨 Flagged as FRAUD because:**\n\n"
            for _, row in drivers.iterrows():
                explanation += f"- **{row['Feature']}** = `{row['Value']:.3f}` pushed toward fraud (SHAP: +{row['SHAP']:.3f})\n"
        else:
            drivers = contributions[contributions['SHAP'] < 0].nsmallest(3, 'SHAP')
            explanation = "**✅ Approved because:**\n\n"
            for _, row in drivers.iterrows():
                explanation += f"- **{row['Feature']}** = `{row['Value']:.3f}` indicated legitimate (SHAP: {row['SHAP']:.3f})\n"
        st.markdown(explanation)


elif page == "🔌 API Mode":
    st.markdown('<p class="main-header">🔌 API Mode (Microservice Architecture)</p>', unsafe_allow_html=True)
    st.markdown("This page calls our **FastAPI backend** instead of using models directly.")
    st.markdown("---")
    API_URL = "http://localhost:8000"
    st.subheader("📡 API Status Check")
    col_a, col_b = st.columns([1, 3])
    if col_a.button("🔄 Check API Health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            if r.status_code == 200:
                col_b.success(f"✅ API online! {r.json()}")
            else:
                col_b.error(f"❌ Status {r.status_code}")
        except Exception as e:
            col_b.error(f"❌ Cannot reach API. Is FastAPI running? Error: {e}")
    st.markdown("---")
    st.subheader("🎯 Predict via API Call")
    if 'tx_api' not in st.session_state:
        st.session_state['tx_api'] = legit_samples[0]
    bb1, bb2 = st.columns(2)
    if bb1.button("🟢 Load LEGIT (via API)", use_container_width=True):
        idx = np.random.randint(0, len(legit_samples))
        st.session_state['tx_api'] = legit_samples[idx]
        st.success(f"Loaded LEGIT #{idx}")
    if bb2.button("🔴 Load FRAUD (via API)", use_container_width=True):
        idx = np.random.randint(0, len(fraud_samples))
        st.session_state['tx_api'] = fraud_samples[idx]
        st.error(f"Loaded FRAUD #{idx}")
    api_model = st.selectbox("Model to call:", ["xgboost", "random_forest", "neural_network", "ensemble"])
    api_threshold = st.slider("API threshold", 0.0, 1.0, 0.5, 0.05)
    if st.button("🚀 SEND REQUEST TO API", type="primary", use_container_width=True):
        payload = {"features": [float(v) for v in st.session_state['tx_api']], "model": api_model, "threshold": api_threshold}
        try:
            with st.spinner("Calling API..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                st.markdown("##### 📥 Raw JSON Response from API:")
                st.json(data)
                rc1, rc2 = st.columns(2)
                with rc1:
                    if data['is_fraud']:
                        st.markdown(f'<div class="fraud-alert"><h2>{data["verdict"]}</h2><h3>Probability: {data["fraud_probability"]*100:.2f}%</h3><p>Model: {data["model_used"]}</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="safe-alert"><h2>{data["verdict"]}</h2><h3>Probability: {data["fraud_probability"]*100:.4f}%</h3><p>Model: {data["model_used"]}</p></div>', unsafe_allow_html=True)
                with rc2:
                    st.markdown("**Top 5 Explanations:**")
                    st.dataframe(pd.DataFrame(data['top_explanations']), use_container_width=True, hide_index=True)
                st.success(f"✅ API responded in {response.elapsed.total_seconds()*1000:.0f}ms")
            else:
                st.error(f"❌ Status {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"❌ Failed: {e}")
    st.info(f"📚 Live API docs: {API_URL}/docs")


elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown("---")
    metrics_df = pd.DataFrame([
        {'Model': 'Random Forest', **{k: rf_res[k] for k in ['accuracy','precision','recall','f1_score','roc_auc','pr_auc']}},
        {'Model': 'XGBoost', **{k: xgb_res[k] for k in ['accuracy','precision','recall','f1_score','roc_auc','pr_auc']}},
        {'Model': 'Neural Network', **{k: nn_res[k] for k in ['accuracy','precision','recall','f1_score','roc_auc','pr_auc']}}])
    fig = px.bar(metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'), x='Metric', y='Score', color='Model', barmode='group',
        color_discrete_sequence=['#2ecc71', '#9b59b6', '#e67e22'], title="Model Comparison — All Metrics")
    fig.update_layout(height=500, yaxis=dict(range=[0.7, 1]))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(metrics_df.set_index('Model').style.format('{:.4f}').background_gradient(cmap='RdYlGn'), use_container_width=True)
    st.subheader("🔍 Confusion Matrices")
    cm_cols = st.columns(3)
    for col, results, name, cmap in zip(cm_cols, [rf_res, xgb_res, nn_res], ['Random Forest', 'XGBoost', 'Neural Net'], ['Blues', 'Purples', 'Oranges']):
        cm = np.array(results['confusion_matrix'])
        fig = px.imshow(cm, text_auto=True, aspect="auto", x=['Pred Legit', 'Pred Fraud'], y=['Actual Legit', 'Actual Fraud'], color_continuous_scale=cmap, title=name)
        col.plotly_chart(fig, use_container_width=True)


elif page == "📈 Dataset Explorer":
    st.markdown('<p class="main-header">📈 Dataset Explorer</p>', unsafe_allow_html=True)
    st.markdown("---")
    @st.cache_data
    def load_data():
        return pd.read_csv('data/creditcard.csv')
    with st.spinner("Loading 284,807 transactions..."):
        df = load_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Cases", f"{df['Class'].sum():,}", f"{df['Class'].mean()*100:.3f}%")
    c3.metric("Features", f"{df.shape[1] - 1}")
    fig = px.pie(values=df['Class'].value_counts().values, names=['Legitimate', 'Fraud'],
        color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
    feat = st.selectbox("Pick a feature to explore:", ['Amount'] + [f'V{i}' for i in [17,14,12,10,16,11,4,3,7]])
    fig = px.histogram(df.sample(min(20000, len(df))), x=feat, color='Class', nbins=60, opacity=0.7,
        color_discrete_sequence=['#2ecc71', '#e74c3c'], barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)


elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
## 🎯 Project Goal
Build a production-grade credit card fraud detection system with zero cloud cost.

## 🛠️ Pipeline
1. **Data** — 284,807 real European transactions
2. **EDA** — statistical analysis, correlations
3. **Preprocessing** — RobustScaler, stratified split, SMOTE
4. **Modeling** — 3 independent models (RF, XGBoost, NN)
5. **Evaluation** — precision, recall, F1, ROC-AUC, PR-AUC
6. **Explainability** — SHAP values for every prediction
7. **API** — FastAPI REST backend
8. **UI** — Streamlit dashboard frontend

## 📈 Key Findings
- Extreme class imbalance (0.172% fraud) required SMOTE
- V14, V17, V12, V10 are strongest fraud predictors
- **XGBoost beat deep learning** on this tabular dataset
- Microservice architecture: API can be called by mobile, web, anywhere
""")

st.markdown("---")
st.markdown("<center><small>Fraud Detection AI © 2026</small></center>", unsafe_allow_html=True)