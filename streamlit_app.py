import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Fraud Detection Dashboard")
st.write("Interface de démonstration pour tester l’API de détection de fraude bancaire en temps réel.")

st.markdown("---")

# =========================
# Exemples de transactions
# =========================
normal_example = {
    "Time": 10000.0,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62,
    "threshold": 0.3
}

fraud_example = {
    "Time": 406.0,
    "V1": -2.312227,
    "V2": 1.951992,
    "V3": -1.609851,
    "V4": 3.997906,
    "V5": -0.522188,
    "V6": -1.426545,
    "V7": -2.537387,
    "V8": 1.391657,
    "V9": -2.770089,
    "V10": -2.772272,
    "V11": 3.202033,
    "V12": -2.899907,
    "V13": -0.595222,
    "V14": -4.289254,
    "V15": 0.389724,
    "V16": -1.140747,
    "V17": -2.830056,
    "V18": -0.016822,
    "V19": 0.416956,
    "V20": 0.126911,
    "V21": 0.517232,
    "V22": -0.035049,
    "V23": -0.465211,
    "V24": 0.320198,
    "V25": 0.044519,
    "V26": 0.177840,
    "V27": 0.261145,
    "V28": -0.143276,
    "Amount": 0.00,
    "threshold": 0.3
}

if "transaction_data" not in st.session_state:
    st.session_state.transaction_data = normal_example.copy()

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("Charger exemple normal"):
        st.session_state.transaction_data = normal_example.copy()

with col2:
    if st.button("Charger exemple fraude"):
        st.session_state.transaction_data = fraud_example.copy()

with col3:
    st.info("Choisis un exemple, puis modifie les valeurs si nécessaire avant de lancer la prédiction.")

st.markdown("---")

data = st.session_state.transaction_data

# =========================
# Formulaire
# =========================
left, right = st.columns(2)

with left:
    st.subheader("Informations principales")
    data["Time"] = st.number_input("Time", value=float(data["Time"]), format="%.6f")
    data["Amount"] = st.number_input("Amount", value=float(data["Amount"]), format="%.6f")
    data["threshold"] = st.slider("Threshold", min_value=0.0, max_value=1.0, value=float(data["threshold"]), step=0.01)

with right:
    st.subheader("API")
    st.code(API_URL, language="text")
    st.caption("L’API FastAPI doit être lancée sur localhost:8000")

st.markdown("### Variables V1 à V28")

feature_cols = st.columns(4)
for i in range(1, 29):
    col = feature_cols[(i - 1) % 4]
    key = f"V{i}"
    with col:
        data[key] = st.number_input(key, value=float(data[key]), format="%.6f")

st.session_state.transaction_data = data

st.markdown("---")

# =========================
# Aperçu JSON
# =========================
with st.expander("Voir les données envoyées à l’API"):
    st.json(data)

# =========================
# Prédiction
# =========================
if st.button("Lancer la prédiction", use_container_width=True):
    try:
        response = requests.post(API_URL, json=data, timeout=10)

        if response.status_code == 200:
            result = response.json()

            st.success("Prédiction réalisée avec succès.")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Fraud probability", f"{result['fraud_probability']:.6f}")
            r2.metric("Is fraud", "Oui" if result["is_fraud"] else "Non")
            r3.metric("Threshold used", f"{result['threshold_used']:.2f}")
            r4.metric("Model", result["model_name"])

            st.markdown("### Résultat détaillé")
            result_df = pd.DataFrame([result])
            st.dataframe(result_df, use_container_width=True)

            if result["is_fraud"]:
                st.error("⚠️ Transaction classée comme frauduleuse.")
            else:
                st.info("✅ Transaction classée comme non frauduleuse.")

        else:
            st.error(f"Erreur API : {response.status_code}")
            st.text(response.text)

    except requests.exceptions.ConnectionError:
        st.error("Impossible de contacter l’API. Vérifie que FastAPI est bien lancée sur http://localhost:8000")
    except requests.exceptions.Timeout:
        st.error("La requête a expiré.")
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")