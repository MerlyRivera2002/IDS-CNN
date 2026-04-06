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
    # CAMBIO CLAVE: Se carga sin compilar para evitar errores de versión en la nube
    try:
        m = tf.keras.models.load_model("modelo_cnn.h5", compile=False)
        # Lo compilamos aquí mismo para que esté listo para predecir
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()
        
    s = joblib.load("scaler.pkl")
    f = joblib.load("features.pkl")
    return m, s, f

st.title("Monitor de Sistema de Detección de Intrusos")
st.subheader("Análisis de Tráfico con CNN")

# Verificación de archivos
if not all(os.path.exists(x) for x in ["modelo_cnn.h5", "scaler.pkl", "features.pkl"]):
    st.error("Faltan archivos de dependencias técnicos (.h5, .pkl) en el repositorio.")
    st.stop()

model, scaler, features_list = load_assets()

archivo = st.file_uploader("Cargar tráfico de red", type=["csv"])

if archivo:
    # Ajustado a 5000 para que el análisis sea más completo
    df_raw = pd.read_csv(archivo, nrows=5000)
    df_raw.columns = df_raw.columns.str.strip()

    if st.button("Iniciar Monitoreo"):
        # Preprocesamiento
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
        mt_total = col_total.metric("Total Flujos Analizados", 0)
        mt_normal = col_normal.metric("Tráfico Normal", 0, delta="Seguro")
        mt_malware = col_malware.metric("Detección de Intrusiones", 0, delta="Riesgo", delta_color="inverse")

        st.divider()
        col_tabla, col_grafico = st.columns([2, 1])

        with col_tabla:
            st.write("Registro de Análisis")
            tabla_flujos = st.empty()

        with col_grafico:
            st.write("Distribución de Tráfico")
            grafico_cnt = st.empty()

        historial_completo = []
        
        # Ajustado a 5000 también aquí
        max_demo_rows = min(5000, total_registros)
        progreso = st.progress(0)

        for i in range(max_demo_rows):
            fila_completa = df_clean.iloc[i].copy()
            entrada_modelo = X_scaled[i:i + 1]

            # Predicción
            probabilidad = model.predict(entrada_modelo, verbose=0)[0][0]
            # Umbral de detección ajustado
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

            mt_total.metric("Total Flujos Analizados", i + 1)
            mt_normal.metric("Tráfico Normal", registros_normales)
            mt_malware.metric("Detección de Intrusiones", registros_maliciosos)

            datos_simplificados = {
                "ID": i + 1,
                "Puerto": int(fila_completa.get("Destination Port", 0)),
                "Resultado": estado_texto,
                "Confianza": f"{confianza_valor:.2f}%"
            }

            historial_completo.insert(0, datos_simplificados)
            # Mostrar solo los últimos 50 en la tabla para que no se ponga lento
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
            # Un poquito de delay para que se vea el efecto en vivo
            time.sleep(0.005)

        st.success("Análisis completo.")

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
