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
        # 1. Definimos la arquitectura manual para que no dependa de la versión de Keras
        # Esta es la estructura estándar de una CNN para IDS (78 características)
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(78, 1)),
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # 2. Intentamos cargar solo los pesos (esto evita el error de batch_shape)
        try:
            m.load_weights("modelo_cnn.h5")
        except:
            # Si los pesos no encajan con la manual, intentamos la carga directa 
            # pero con el parche de compatibilidad incluido
            from tensorflow.keras.layers import InputLayer
            class CustomInputLayer(InputLayer):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('batch_shape', None)
                    kwargs.pop('shape', None)
                    super().__init__(*args, **kwargs)
            
            m = tf.keras.models.load_model(
                "modelo_cnn.h5", 
                custom_objects={'InputLayer': CustomInputLayer}, 
                compile=False
            )
            
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Carga de transformadores
        s = joblib.load("scaler.pkl")
        f = joblib.load("features.pkl")
        return m, s, f
    except Exception as e:
        st.error(f"Error crítico en el servidor: {e}")
        st.stop()

# --- INTERFAZ DE USUARIO ---
st.title("🛡️ Monitor de Sistema de Detección de Intrusos")
st.subheader("Análisis de Tráfico de Red en Tiempo Real con CNN")

# Verificación de archivos necesarios
if not all(os.path.exists(x) for x in ["modelo_cnn.h5", "scaler.pkl", "features.pkl"]):
    st.error("⚠️ Error: No se encuentran los archivos .h5 o .pkl en el repositorio de GitHub.")
    st.stop()

model, scaler, features_list = load_assets()

archivo = st.file_uploader("📂 Cargar archivo de tráfico (CSV)", type=["csv"])

if archivo:
    # Cargamos 5,000 registros para una demostración fluida
    df_raw = pd.read_csv(archivo, nrows=5000)
    df_raw.columns = df_raw.columns.str.strip()

    if st.button("🚀 Iniciar Análisis de Seguridad"):
        # Preprocesamiento de datos
        df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
        X_input = df_clean[features_list]
        X_scaled = scaler.transform(X_input)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        total_registros = len(X_scaled)
        reg_normal = 0
        reg_ataque = 0
        preds = []

        st.divider()
        c1, c2, c3 = st.columns(3)
        m1 = c1.metric("Flujos Analizados", 0)
        m2 = c2.metric("Tráfico Seguro", 0)
        m3 = c3.metric("Ataques Detectados", 0, delta_color="inverse")

        st.divider()
        col_t, col_g = st.columns([2, 1])
        with col_t:
            st.info("Log de Eventos Recientes")
            tabla = st.empty()
        with col_g:
            st.info("Distribución de Amenazas")
            grafico = st.empty()

        historial = []
        progreso = st.progress(0)

        # Bucle de monitoreo simulado
        for i in range(total_registros):
            entrada = X_scaled[i:i + 1]
            prob = model.predict(entrada, verbose=0)[0][0]
            
            # Clasificación
            es_ataque = 1 if prob > 0.5 else 0
            preds.append(es_ataque)

            if es_ataque == 0:
                reg_normal += 1
                resultado = "✅ NORMAL"
                conf = (1 - prob) * 100
            else:
                reg_ataque += 1
                resultado = "🚨 ATAQUE"
                conf = prob * 100

            # Actualizar Dashboard
            m1.metric("Flujos Analizados", i + 1)
            m2.metric("Tráfico Seguro", reg_normal)
            m3.metric("Ataques Detectados", reg_ataque, delta=f"+{reg_ataque}" if reg_ataque > 0 else 0)

            # Historial simplificado
            historial.insert(0, {
                "ID": i + 1,
                "Puerto Destino": int(df_clean.iloc[i].get("Destination Port", 0)),
                "Veredicto": resultado,
                "Confianza": f"{conf:.2f}%"
            })
            tabla.dataframe(pd.DataFrame(historial[:15]), use_container_width=True)

            # Actualizar gráfico cada 50 registros para no saturar
            if (i + 1) % 50 == 0 or i == total_registros - 1:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie([reg_normal, reg_ataque], labels=["Seguro", "Ataque"], 
                       autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
                grafico.pyplot(fig)
                plt.close(fig)

            progreso.progress((i + 1) / total_registros)
            time.sleep(0.001)

        st.success("✅ Monitoreo finalizado con éxito.")

        # Métricas de tesis
        if "Label" in df_clean.columns:
            st.divider()
            from sklearn.metrics import accuracy_score, classification_report
            y_real = df_clean["Label"].iloc[:total_registros].astype(str).str.upper().apply(
                lambda x: 0 if "BENIGN" in x else 1)
            
            acc = accuracy_score(y_real, preds)
            st.metric("Precisión del Modelo (Accuracy)", f"{acc:.4f}")
            st.text("Matriz de Clasificación Detallada:")
            st.text(classification_report(y_real, preds, target_names=["Normal", "Ataque"]))
