import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import matplotlib.pyplot as plt

# Configuración de la interfaz
st.set_page_config(page_title="IDS", layout="wide")

@st.cache_resource
def load_assets():
    # --- PARCHE DEFINITIVO DE COMPATIBILIDAD ---
    from tensorflow.keras.layers import InputLayer
    
    class CustomInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            # Borramos los argumentos que causan el error en Keras 3
            kwargs.pop('batch_shape', None)
            kwargs.pop('shape', None)
            kwargs.pop('config', None)
            super().__init__(*args, **kwargs)

    # Registramos el objeto personalizado para que load_model lo use
    custom_objects = {'InputLayer': CustomInputLayer}
    # --- FIN DEL PARCHE ---

    try:
        # Cargamos el modelo con el parche y sin compilar
        m = tf.keras.models.load_model(
            "modelo_cnn.h5", 
            custom_objects=custom_objects, 
            compile=False
        )
        # Re-compilamos para que funcione en este entorno
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    except Exception as e:
        st.error(f"Error técnico de compatibilidad: {e}")
        st.stop()
        
    s = joblib.load("scaler.pkl")
    f = joblib.load("features.pkl")
    return m, s, f

st.title("Monitor de Sistema de Detección de Intrusos")
st.subheader("Análisis de Tráfico con CNN")

# Verificación de archivos en el repo
if not all(os.path.exists(x) for x in ["modelo_cnn.h5", "scaler.pkl", "features.pkl"]):
    st.error("Faltan archivos técnicos (.h5, .pkl) en el repositorio de GitHub.")
    st.stop()

model, scaler, features_list = load_assets()

archivo = st.file_uploader("Cargar tráfico de red (CSV)", type=["csv"])

if archivo:
    # Procesamos 5000 filas para una buena demo
    df_raw = pd.read_csv(archivo, nrows=5000)
    df_raw.columns = df_raw.columns.str.strip()

    if st.button("Iniciar Monitoreo"):
        # Limpieza de datos
        df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
        X_input = df_clean[features_list]
        X_scaled = scaler.transform(X_input)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        total_registros = len(X_scaled)
        registros_normales = 0
        registros_maliciosos = 0
        preds_acumuladas = []

        st.divider()
        st.subheader("Dashboard de Monitoreo Activo")

        col_total, col_normal, col_malware = st.columns(3)
        mt_total = col_total.metric("Total Analizado", 0)
        mt_normal = col_normal.metric("Tráfico Normal", 0, delta="Seguro")
        mt_malware = col_malware.metric("Intrusiones", 0, delta="Riesgo", delta_color="inverse")

        st.divider()
        col_tabla, col_grafico = st.columns([2, 1])

        with col_tabla:
            st.write("Registro en Tiempo Real")
            tabla_flujos = st.empty()

        with col_grafico:
            st.write("Distribución de Eventos")
            grafico_cnt = st.empty()

        historial_completo = []
        max_demo_rows = min(5000, total_registros)
        progreso = st.progress(0)

        for i in range(max_demo_rows):
            fila_completa = df_clean.iloc[i].copy()
            entrada_modelo = X_scaled[i:i + 1]

            # Inferencia
            probabilidad = model.predict(entrada_modelo, verbose=0)[0][0]
            prediccion_binaria = 1 if probabilidad > 0.5 else 0
            preds_acumuladas.append(prediccion_binaria)

            if prediccion_binaria == 0:
                registros_normales += 1
                estado_texto = "NORMAL"
                confianza_valor = (1 - probabilidad) * 100
            else:
                registros_maliciosos += 1
                estado_texto = "ATAQUE"
                confianza_valor = probabilidad * 100

            # Actualizar métricas
            mt_total.metric("Total Analizado", i + 1)
            mt_normal.metric("Tráfico Normal", registros_normales)
            mt_malware.metric("Intrusiones", registros_maliciosos)

            datos_simplificados = {
                "ID": i + 1,
                "Puerto": int(fila_completa.get("Destination Port", 0)),
                "Resultado": estado_texto,
                "Confianza": f"{confianza_valor:.2f}%"
            }

            historial_completo.insert(0, datos_simplificados)
            # Mostramos los últimos 50 para que no se trabe el navegador
            tabla_flujos.dataframe(pd.DataFrame(historial_completo[:50]), use_container_width=True)

            if (i + 1) % 25 == 0 or i == max_demo_rows - 1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie([registros_normales, registros_maliciosos],
                       labels=["Normal", "Ataque"],
                       autopct='%1.1f%%',
                       colors=['#2ecc71', '#e74c3c'],
                       startangle=140)
                grafico_cnt.pyplot(fig)
                plt.close(fig)

            progreso.progress((i + 1) / max_demo_rows)
            time.sleep(0.005)

        st.success("Análisis completo.")

        # Reporte de métricas finales (si existe la etiqueta real)
        if "Label" in df_clean.columns:
            st.divider()
            st.subheader("Reporte Final de Rendimiento")
            from sklearn.metrics import accuracy_score, classification_report

            labels_reales = df_clean["Label"].iloc[:max_demo_rows].astype(str).str.upper().apply(
                lambda x: 0 if "BENIGN" in x else 1)

            acc = accuracy_score(labels_reales, preds_acumuladas)
            st.metric("Precisión Global (Accuracy)", f"{acc:.4f}")
            st.text("Matriz de Clasificación:")
            st.text(classification_report(labels_reales, preds_acumuladas, target_names=["Normal", "Ataque"]))
