import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="IDS", layout="wide")

@st.cache_resource
def load_assets():
    try:
        # Al usar TF 2.15, esto debería cargar sin errores de DTypePolicy
        m = tf.keras.models.load_model("modelo_cnn.h5", compile=False)
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        s = joblib.load("scaler.pkl")
        f = joblib.load("features.pkl")
        return m, s, f
    except Exception as e:
        st.error(f"Error de carga: {e}")
        st.stop()

st.title("Monitor de Sistema de Detección de Intrusos")
st.subheader("Análisis de Tráfico con CNN")

if not all(os.path.exists(x) for x in ["modelo_cnn.h5", "scaler.pkl", "features.pkl"]):
    st.error("Faltan archivos (.h5 o .pkl) en GitHub.")
    st.stop()

model, scaler, features_list = load_assets()
archivo = st.file_uploader("Cargar tráfico de red (CSV)", type=["csv"])

if archivo:
    df_raw = pd.read_csv(archivo, nrows=5000)
    df_raw.columns = df_raw.columns.str.strip()

    if st.button("Iniciar Monitoreo"):
        df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
        X_input = df_clean[features_list]
        X_scaled = scaler.transform(X_input)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        registros_normales = 0
        registros_maliciosos = 0
        preds_acumuladas = []

        st.divider()
        col_total, col_normal, col_malware = st.columns(3)
        mt_total = col_total.metric("Total Analizado", 0)
        mt_normal = col_normal.metric("Tráfico Normal", 0)
        mt_malware = col_malware.metric("Intrusiones", 0)

        tabla_flujos = st.empty()
        grafico_cnt = st.empty()
        historial_completo = []
        progreso = st.progress(0)

        for i in range(len(X_scaled)):
            entrada_modelo = X_scaled[i:i + 1]
            probabilidad = model.predict(entrada_modelo, verbose=0)[0][0]
            prediccion = 1 if probabilidad > 0.5 else 0
            preds_acumuladas.append(prediccion)

            if prediccion == 0:
                registros_normales += 1
                estado = "NORMAL"
            else:
                registros_maliciosos += 1
                estado = "ATAQUE"

            mt_total.metric("Total Analizado", i + 1)
            mt_normal.metric("Tráfico Normal", registros_normales)
            mt_malware.metric("Intrusiones", registros_maliciosos)

            historial_completo.insert(0, {"ID": i+1, "Resultado": estado, "Confianza": f"{probabilidad*100:.2f}%"})
            tabla_flujos.dataframe(pd.DataFrame(historial_completo[:10]), use_container_width=True)

            if (i + 1) % 50 == 0:
                fig, ax = plt.subplots()
                ax.pie([registros_normales, registros_maliciosos], labels=["Normal", "Ataque"], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
                grafico_cnt.pyplot(fig)
                plt.close(fig)

            progreso.progress((i + 1) / len(X_scaled))
            time.sleep(0.001)
