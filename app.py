import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import matplotlib.pyplot as plt

# Configuración de la interfaz
st.set_page_config(page_title="IDS - Dashboard de Tesis", layout="wide")

@st.cache_resource
def load_assets():
    try:
        # ARQUITECTURA AJUSTADA (4 capas detectadas en el archivo)
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(78, 1)),
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'), # Capa extra para cuadrar
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Intentamos cargar los pesos. Si falla el conteo, usamos el cargador automático
        # que ahora debería funcionar porque ya tienes Python 3.11 y TF 2.15
        try:
            m.load_weights("modelo_cnn.h5")
            m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        except:
            # Plan B: Carga directa (Ahora que el entorno es compatible)
            m = tf.keras.models.load_model("modelo_cnn.h5", compile=False)
            m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        s = joblib.load("scaler.pkl")
        f = joblib.load("features.pkl")
        return m, s, f
    except Exception as e:
        st.error(f"Error al montar el modelo: {e}")
        st.stop()

# --- INTERFAZ ---
st.title("🛡️ Monitor de Sistema de Detección de Intrusos")
st.subheader("Análisis de Tráfico de Red con Inteligencia Artificial (CNN)")

# Verificación de archivos
if not all(os.path.exists(x) for x in ["modelo_cnn.h5", "scaler.pkl", "features.pkl"]):
    st.error("⚠️ Faltan archivos en GitHub: Sube modelo_cnn.h5, scaler.pkl y features.pkl.")
    st.stop()

model, scaler, features_list = load_assets()

archivo = st.file_uploader("📂 Cargar dataset de tráfico (CSV)", type=["csv"])

if archivo:
    df_raw = pd.read_csv(archivo, nrows=5000)
    df_raw.columns = df_raw.columns.str.strip()

    if st.button("🚀 Iniciar Monitoreo"):
        # Preprocesamiento rápido
        df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
        X_input = df_clean[features_list]
        X_scaled = scaler.transform(X_input)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        reg_normal, reg_ataque = 0, 0
        preds = []

        st.divider()
        c1, c2, c3 = st.columns(3)
        m1 = c1.metric("Total Analizado", 0)
        m2 = c2.metric("Seguro", 0)
        m3 = c3.metric("Ataques", 0, delta_color="inverse")

        col_t, col_g = st.columns([2, 1])
        with col_t:
            tabla = st.empty()
        with col_g:
            grafico = st.empty()

        historial = []
        progreso = st.progress(0)

        for i in range(len(X_scaled)):
            entrada = X_scaled[i:i + 1]
            prob = model.predict(entrada, verbose=0)[0][0]
            
            es_ataque = 1 if prob > 0.5 else 0
            preds.append(es_ataque)

            if es_ataque == 0:
                reg_normal += 1
                res = "✅ NORMAL"
                conf = (1 - prob) * 100
            else:
                reg_ataque += 1
                res = "🚨 ATAQUE"
                conf = prob * 100

            m1.metric("Total Analizado", i + 1)
            m2.metric("Seguro", reg_normal)
            m3.metric("Ataques", reg_ataque, delta=f"+{reg_ataque}" if reg_ataque > 0 else 0)

            historial.insert(0, {"ID": i + 1, "Resultado": res, "Confianza": f"{conf:.2f}%"})
            tabla.dataframe(pd.DataFrame(historial[:10]), use_container_width=True)

            if (i + 1) % 50 == 0 or i == len(X_scaled) - 1:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie([reg_normal, reg_ataque], labels=["Seguro", "Ataque"], 
                       autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
                grafico.pyplot(fig)
                plt.close(fig)

            progreso.progress((i + 1) / len(X_scaled))
            time.sleep(0.001)

        st.success("Análisis finalizado.")
