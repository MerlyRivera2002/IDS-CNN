import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monitor IDS", layout="wide")

@st.cache_resource
def load_assets():
    try:
        # Cargamos el nuevo formato nativo .keras
        m = tf.keras.models.load_model("modelo_cnn.keras")
        s = joblib.load("scaler.pkl")
        f = joblib.load("features.pkl")
        return m, s, f
    except Exception as e:
        st.error(f"Error al cargar archivos nuevos: {e}")
        st.stop()

st.title("🛡️ Sistema de Detección de Intrusos (IDS)")
st.subheader("Análisis de Tráfico con CNN - Tesis")

# Verificación de archivos
if not os.path.exists("modelo_cnn.keras"):
    st.error("⚠️ Sube 'modelo_cnn.keras' a tu GitHub.")
    st.stop()

model, scaler, features_list = load_assets()

archivo = st.file_uploader("📂 Cargar dataset CSV para monitoreo", type=["csv"])

if archivo:
    df_raw = pd.read_csv(archivo, nrows=5000)
    df_raw.columns = df_raw.columns.str.strip()

    if st.button("🚀 Iniciar Análisis"):
        df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
        X_input = df_clean[features_list]
        X_scaled = scaler.transform(X_input)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        total = len(X_scaled)
        normal, ataque = 0, 0
        historial = []

        # Métricas principales
        st.divider()
        c1, c2, c3 = st.columns(3)
        mt1 = c1.metric("Analizados", 0)
        mt2 = c2.metric("Normal", 0)
        mt3 = c3.metric("Ataques", 0)

        t_col, g_col = st.columns([2, 1])
        tabla = t_col.empty()
        grafico = g_col.empty()
        progreso = st.progress(0)

        for i in range(total):
            # Predicción
            prob = model.predict(X_scaled[i:i+1], verbose=0)[0][0]
            es_ataque = 1 if prob > 0.5 else 0

            if es_ataque == 0:
                normal += 1
                res = "✅ NORMAL"
            else:
                ataque += 1
                res = "🚨 ATAQUE"

            # Actualizar métricas
            mt1.metric("Analizados", i + 1)
            mt2.metric("Normal", normal)
            mt3.metric("Ataques", ataque)

            historial.insert(0, {
                "ID": i + 1,
                "Resultado": res,
                "Confianza": f"{prob*100 if es_ataque else (1-prob)*100:.2f}%"
            })
            tabla.dataframe(pd.DataFrame(historial[:10]), use_container_width=True)

            if (i + 1) % 50 == 0 or i == total - 1:
                fig, ax = plt.subplots(figsize=(4,4))
                ax.pie([normal, ataque], labels=["Normal", "Ataque"], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
                grafico.pyplot(fig)
                plt.close(fig)

            progreso.progress((i + 1) / total)
            time.sleep(0.001)

        st.success("🎯 Análisis completado con éxito.")
