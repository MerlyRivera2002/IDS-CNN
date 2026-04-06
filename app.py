import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import matplotlib.pyplot as plt

#Configuración de la interfaz
st.set_page_config(page_title="IDS", layout="wide")


@st.cache_resource
def load_assets():
    m = tf.keras.models.load_model("modelo_cnn.h5")
    s = joblib.load("scaler.pkl")
    f = joblib.load("features.pkl")
    return m, s, f


st.title("Monitor de Sistema de Detección de Intrusos")
st.subheader("Análisis de Tráfico con CNN")

if not all(os.path.exists(x) for x in ["modelo_cnn.h5", "scaler.pkl", "features.pkl"]):
    st.error("Faltan archivos de dependencias técnicos (.h5, .pkl).")
    st.stop()

model, scaler, features_list = load_assets()

archivo = st.file_uploader("Cargar tráfico de red", type=["csv"])

if archivo:
    # Cargamos 2000 filas para que la PC no sufra
    df_raw = pd.read_csv(archivo, nrows=2000)
    df_raw.columns = df_raw.columns.str.strip()

    if st.button("Iniciar Monitoreo"):
        # Preprocesamiento rápido
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

        # Métricas principales en grande
        col_total, col_normal, col_malware = st.columns(3)
        mt_total = col_total.metric("Total Flujos Analizados", 0)
        mt_normal = col_normal.metric("Tráfico Normal", 0, delta="Seguro")
        mt_malware = col_malware.metric("Detección de Intrusiones", 0, delta="Riesgo", delta_color="inverse")

        st.divider()
        col_tabla, col_grafico = st.columns([2, 1])

        with col_tabla:
            st.write("Registro de Análisis")
            tabla_flujos = st.empty()  # Espacio para la tabla

        with col_grafico:
            st.write("Distribución de Tráfico")
            grafico_cnt = st.empty()

        # Lista donde guardaremos
        historial_completo = []

        max_demo_rows = min(2000, total_registros)
        progreso = st.progress(0)

        for i in range(max_demo_rows):
            fila_completa = df_clean.iloc[i].copy()
            entrada_modelo = X_scaled[i:i + 1]

            # Predicción CNN
            probabilidad = model.predict(entrada_modelo, verbose=0)[0][0]
            prediccion_binaria = 1 if probabilidad > 0.05 else 0
            preds_acumuladas.append(prediccion_binaria)

            if prediccion_binaria == 0:
                registros_normales += 1
                estado_texto = "NORMAL"
                confianza_valor = (1 - probabilidad) * 100
            else:
                registros_maliciosos += 1
                estado_texto = "ATAQUE"
                confianza_valor = probabilidad * 100

            # Actualizar métricas superiores
            mt_total.metric("Total Flujos Analizados", i + 1)
            mt_normal.metric("Tráfico Normal", registros_normales)
            mt_malware.metric("Detección de Intrusiones", registros_maliciosos)

            # Limpieza de la tabla
            datos_simplificados = {
                "ID": i + 1,
                "Puerto": int(fila_completa.get("Destination Port", 0)),
                "Resultado": estado_texto,
                "Porcentaje": f"{confianza_valor:.2f}%"
            }

            # Guardamos en el historial
            historial_completo.insert(0, datos_simplificados)

            # Mostramos la tabla
            tabla_flujos.dataframe(pd.DataFrame(historial_completo), use_container_width=True, height=400)

            # Gráfico de pastel (se actualiza cada 20 para no trabar la PC)
            if (i + 1) % 20 == 0 or i == max_demo_rows - 1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie([registros_normales, registros_maliciosos],
                       labels=["Normal", "Ataque"],
                       autopct='%1.1f%%',
                       colors=['#2ecc71', '#e74c3c'],
                       startangle=140)
                grafico_cnt.pyplot(fig)
                plt.close(fig)

            progreso.progress((i + 1) / max_demo_rows)
            time.sleep(0.01)

        st.success("Análisis completo. El historial está disponible arriba para su revisión.")

        # Reporte Final de Métricas
        if "Label" in df_clean.columns:
            st.divider()
            st.subheader("Reporte Final de Rendimiento")
            from sklearn.metrics import accuracy_score, classification_report

            labels_reales = df_clean["Label"].iloc[:max_demo_rows].astype(str).str.upper().apply(
                lambda x: 0 if "BENIGN" in x else 1)

            acc = accuracy_score(labels_reales, preds_acumuladas)

            c1, c2 = st.columns(2)
            c1.metric("Precisión Global (Accuracy)", f"{acc:.4f}")

            st.text("Matriz de Clasificación Detallada:")
            st.text(classification_report(labels_reales, preds_acumuladas, target_names=["Normal", "Ataque"]))