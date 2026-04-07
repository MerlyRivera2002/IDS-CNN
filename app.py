import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import plotly.express as px # Gráficos modernos
import logic 

# Configuración visual
st.set_page_config(page_title="IDS Tesis 2026", layout="wide")

# --- LOGIN ---
if 'perfil' not in st.session_state: st.session_state.perfil = None

if st.session_state.perfil is None:
    st.title("🛡️ DASHBOARD IDS - EDICIÓN TESIS") # <--- SI NO VES ESTO, NO SE ACTUALIZÓ
    u = st.text_input("Usuario")
    p = st.text_input("Clave", type="password")
    if st.button("Ingresar"):
        if u == "admin" and p == "tesis2026": 
            st.session_state.perfil = "Administrador"
            st.rerun()
    st.stop()

# --- ASSETS ---
@st.cache_resource
def load_assets():
    m = tf.keras.models.load_model("modelo_cnn.keras")
    s = joblib.load("scaler.pkl")
    f = joblib.load("features.pkl")
    return m, s, f

model, scaler, features_list = load_assets()

# --- PESTAÑAS ---
tab1, tab2 = st.tabs(["🚀 MONITOR & MÉTRICAS", "📅 BITÁCORA POR DÍAS"])

with tab1:
    st.header("Análisis de Tráfico y Métricas del Dataset")
    archivo = st.file_uploader("Subir archivo CSV", type=["csv"])
    
    if archivo:
        if st.button("▶️ EJECUTAR ANÁLISIS"):
            with st.spinner("Calculando métricas y procesando modelo..."):
                t_ini = time.time()
                df = pd.read_csv(archivo, nrows=2000) # Límite de 2000 para rapidez
                df.columns = df.columns.str.strip()
                df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
                X = scaler.transform(df_clean[features_list]).reshape(-1, len(features_list), 1)
                
                # Predicciones
                prob = model.predict(X, verbose=0)
                preds = (prob > 0.5).astype(int).flatten()
                t_fin = time.time()

            # --- MÉTRICAS VISUALES (PANEL 1) ---
            st.success("✅ Análisis finalizado exitosamente")
            
            c1, c2, c3 = st.columns(3)
            ataques = int(np.sum(preds))
            normales = len(preds) - ataques
            
            with c1:
                st.write("### Distribución de Tráfico")
                fig_pie = px.pie(names=["Normal", "Ataque"], values=[normales, ataques], 
                                color_discrete_sequence=["#2ecc71", "#e74c3c"], hole=0.5)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                st.write("### Puertos más Frecuentes")
                top_p = logic.analizar_puertos(df_clean, preds)
                if top_p is not None:
                    st.bar_chart(top_p)
            
            with c3:
                # MATRIZ DE CONFUSIÓN CON MAPA DE CALOR (HEATMAP)
                if 'Label' in df_clean.columns:
                    st.write("### Matriz de Confusión")
                    y_real = df_clean['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
                    cm, _ = logic.generar_metricas_detalladas(y_real, preds)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                      x=['Pred: Normal', 'Pred: Ataque'], 
                                      y=['Real: Normal', 'Real: Ataque'])
                    st.plotly_chart(fig_cm, use_container_width=True)

            # TABLA DE DATOS AL FINAL
            st.divider()
            st.subheader("📋 Tabla de Inspección de Datos")
            df_clean['ESTADO_IA'] = ["⚠️ ATAQUE" if p == 1 else "✅ NORMAL" for p in preds]
            st.dataframe(df_clean, use_container_width=True)
            
            logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataques, (t_fin-t_ini))

with tab2:
    st.header("Historial Organizado por Fecha")
    if os.path.exists("historial.csv"):
        h = pd.read_csv("historial.csv")
        h['Fecha_Dt'] = pd.to_datetime(h['Fecha'])
        
        # Agrupamos por día de la semana
        for fecha, grupo in h.groupby(h['Fecha_Dt'].dt.date):
            dia_nombre = pd.to_datetime(fecha).strftime('%A %d de %B')
            st.markdown(f"#### 🗓️ {dia_nombre.upper()}")
            
            # Estilo: si hay ataques, la fila se ve rojiza
            def destacar(val):
                return 'background-color: #f8d7da' if val > 0 else 'background-color: #d4edda'
            
            st.dataframe(grupo.drop(columns=['Fecha_Dt']).style.applymap(destacar, subset=['Ataques_Detectados']), use_container_width=True)
            st.divider()
