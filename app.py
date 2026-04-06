import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import matplotlib.pyplot as plt
import logic  # <--- IMPORTAMOS TU NUEVO ARCHIVO DE LÓGICA

# Configuración inicial
st.set_page_config(page_title="IDS Tesis 2026", layout="wide")

# --- SISTEMA DE LOGIN ---
if 'perfil' not in st.session_state:
    st.session_state.perfil = None

st.sidebar.title("🔐 Acceso al Sistema")
if st.session_state.perfil is None:
    user = st.sidebar.text_input("Usuario")
    pw = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Ingresar"):
        if user == "admin" and pw == "tesis2026":
            st.session_state.perfil = "Administrador"
            st.rerun()
        elif user == "viewer" and pw == "consulta":
            st.session_state.perfil = "Usuario"
            st.rerun()
        else:
            st.sidebar.error("Datos incorrectos")
    st.stop()
else:
    st.sidebar.write(f"Conectado como: **{st.session_state.perfil}**")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.perfil = None
        st.rerun()

# --- CARGA DE MODELO (IGUAL QUE ANTES) ---
@st.cache_resource
def load_assets():
    try:
        m = tf.keras.models.load_model("modelo_cnn.keras")
        s = joblib.load("scaler.pkl")
        f = joblib.load("features.pkl")
        return m, s, f
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

model, scaler, features_list = load_assets()

# --- NAVEGACIÓN POR PESTAÑAS ---
tab1, tab2, tab3 = st.tabs(["🚀 Monitor en Vivo", "📊 Historial Diario", "🎯 Evaluación de Desempeño"])

# --- PESTAÑA 1: MONITOR (Solo Admin puede cargar) ---
with tab1:
    st.title("🛡️ Monitor de Intrusiones")
    if st.session_state.perfil == "Administrador":
        archivo = st.file_uploader("Subir tráfico de red (CSV)", type=["csv"])
        if archivo:
            df_raw = pd.read_csv(archivo, nrows=5000)
            df_raw.columns = df_raw.columns.str.strip()
            
            if st.button("Iniciar Escaneo"):
                t_inicio = time.time() # EMPIEZA EL CRONÓMETRO
                
                df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
                X_scaled = scaler.transform(df_clean[features_list])
                X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

                # Simulación de monitoreo (resumida para no saturar)
                normal, ataque = 0, 0
                preds = []
                for i in range(len(X_scaled)):
                    prob = model.predict(X_scaled[i:i+1], verbose=0)[0][0]
                    p_binaria = 1 if prob > 0.5 else 0
                    preds.append(p_binaria)
                    if p_binaria == 0: normal += 1
                    else: ataque += 1
                
                t_fin = time.time() # TERMINA EL CRONÓMETRO
                
                # CÁLCULOS DE LÓGICA (LLAMANDO A logic.py)
                t_total, t_reg = logic.calcular_eficiencia(t_inicio, t_fin, len(X_scaled))
                logic.guardar_en_historial("historial.csv", archivo.name, len(X_scaled), ataque, t_total)
                
                st.success(f"Análisis terminado en {t_total} seg. ({t_reg} seg/registro)")
                
                # Gráfico rápido de resultados
                fig, ax = plt.subplots()
                ax.pie([normal, ataque], labels=["Normal", "Ataque"], autopct='%1.1f%%', colors=['green', 'red'])
                st.pyplot(fig)
                
                # Guardamos las predicciones en sesión para la pestaña de métricas
                st.session_state.ultimas_preds = preds
                st.session_state.ultimo_df = df_clean
    else:
        st.warning("Solo el Administrador puede cargar nuevos datos para analizar.")

# --- PESTAÑA 2: HISTORIAL (Dashboard por día) ---
with tab2:
    st.title("📅 Reporte Histórico")
    if os.path.exists("historial.csv"):
        hist = pd.read_csv("historial.csv")
        st.dataframe(hist, use_container_width=True)
        # Aquí podrías poner un gráfico de barras comparando días
        st.bar_chart(hist.set_index("Fecha")["Ataques_Detectados"])
    else:
        st.info("Aún no hay registros en el historial.")

# --- PESTAÑA 3: EVALUACIÓN (Capítulo 4) ---
with tab3:
    st.title("🎯 Métricas de Eficiencia")
    if 'ultimas_preds' in st.session_state:
        df = st.session_state.ultimo_df
        if 'Label' in df.columns:
            y_real = df['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
            cm, reporte = logic.generar_metricas_detalladas(y_real, st.session_state.ultimas_preds)
            
            st.write("### Matriz de Confusión")
            st.write(cm)
            st.write("### Reporte de Clasificación")
            st.json(reporte)
        else:
            st.error("El dataset no tiene columna 'Label' para calcular exactitud.")
    else:
        st.info("Realiza un escaneo en la pestaña 1 para ver las métricas aquí.")
