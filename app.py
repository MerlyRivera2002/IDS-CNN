import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logic 

st.set_page_config(page_title="Sistema de Detección de Intrusos (IDS)", layout="wide")

# --- 1. CONTROL DE ACCESO ---
if 'perfil' not in st.session_state: st.session_state.perfil = None

st.sidebar.title("🛡️ Gestión de Acceso")
if st.session_state.perfil is None:
    u = st.sidebar.text_input("Nombre de Usuario")
    p = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Iniciar Sesión"):
        if u == "admin" and p == "tesis2026": 
            st.session_state.perfil = "Administrador"
            st.rerun()
    st.stop()

# --- 2. RECURSOS DEL SISTEMA ---
@st.cache_resource
def load_assets():
    m = tf.keras.models.load_model("modelo_cnn.keras")
    s = joblib.load("scaler.pkl")
    f = joblib.load("features.pkl")
    return m, s, f

model, scaler, features_list = load_assets()

# --- 3. INTERFAZ PRINCIPAL ---
tab1, tab2 = st.tabs(["🚀 MONITOREO Y EVALUACIÓN", "📊 ANÁLISIS DE VULNERABILIDADES"])

with tab1:
    st.header("Monitoreo de Tráfico de Red en Tiempo Real")
    archivo = st.file_uploader("Cargar flujo de datos (Formato CSV)", type=["csv"])
    
    if archivo:
        if st.button("▶️ EJECUTAR PROCESAMIENTO"):
            # Contenedores dinámicos
            col_metricas, col_grafico = st.columns([1, 2])
            espacio_tabla = st.empty()
            
            with st.status("Analizando paquetes de red mediante CNN...", expanded=True) as status:
                t_ini = time.time()
                df_raw = pd.read_csv(archivo, nrows=1000)
                df_raw.columns = df_raw.columns.str.strip() # Limpia espacios en nombres
                df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
                
                # Seleccionar columnas existentes para la tabla (Evita el KeyError)
                cols_visuales = [c for c in ['Source Port', 'Destination Port', 'Protocol'] if c in df_clean.columns]
                if not cols_visuales: # Si no encuentra esas, toma las primeras 3
                    cols_visuales = df_clean.columns[:3].tolist()

                X = scaler.transform(df_clean[features_list]).reshape(-1, len(features_list), 1)
                
                preds, normal, ataque = [], 0, 0
                paso = 20
                for i in range(0, len(X), paso):
                    bloque_X = X[i:i+paso]
                    res = (model.predict(bloque_X, verbose=0) > 0.5).astype(int).flatten()
                    
                    for r in res:
                        preds.append(r)
                        if r == 1: ataque += 1
                        else: normal += 1
                    
                    # 1. Conteo y 2. Gráfico (Juntos arriba)
                    with col_metricas.container():
                        st.metric("Eventos Normales", normal)
                        st.metric("Intrusiones", ataque)
                    
                    with col_grafico.container():
                        fig_pie = px.pie(values=[normal, ataque], names=['Normal', 'Ataque'], 
                                        color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                        fig_pie.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=True)
                        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_realtime_{i}")

                    # 3. Tabla (Corriendo abajo)
                    with espacio_tabla.container():
                        st.write("**Flujo de Datos Recientes:**")
                        temp_df = df_clean.iloc[max(0, i-10):i+paso].copy()
                        temp_df['Clasificación'] = ["ANOMALÍA" if p == 1 else "BENIGNO" for p in preds[max(0, i-10):i+paso]]
                        st.dataframe(temp_df[cols_visuales + ['Clasificación']], use_container_width=True)
                    
                    time.sleep(0.2) # Velocidad ajustada para que se note el movimiento

                t_fin = time.time()
                status.update(label="✅ Análisis de Flujo Completado", state="complete")

            # --- SECCIÓN DE MÉTRICAS (SOLO AL TERMINAR) ---
            st.divider()
            st.header("📈 Evaluación Estadística del Modelo")
            
            if 'Label' in df_clean.columns:
                y_real = df_clean['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
                
                # Cálculos académicos
                acc = accuracy_score(y_real, preds)
                prec = precision_score(y_real, preds, zero_division=0)
                rec = recall_score(y_real, preds, zero_division=0)
                f1 = f1_score(y_real, preds, zero_division=0)
                
                c_matriz, c_grafico_met = st.columns([2, 3])
                
                with c_matriz:
                    st.write("**Matriz de Confusión**")
                    cm, _ = logic.generar_metricas_detalladas(y_real, preds)
                    fig_cm = px.imshow(cm, text_auto=True, x=['Pred. Benigno', 'Pred. Ataque'], 
                                      y=['Real Benigno', 'Real Ataque'], color_continuous_scale='Blues')
                    st.plotly_chart(fig_cm, use_container_width=True)

                with c_grafico_met:
                    st.write("**Indicadores de Rendimiento**")
                    metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    valores = [acc, prec, rec, f1]
                    
                    fig_bar = go.Figure([go.Bar(x=metricas_nombres, y=valores, marker_color=['#1abc9c','#3498db','#9b59b6','#34495e'], text=[f"{v:.4f}" for v in valores], textposition='auto')])
                    fig_bar.update_layout(yaxis=dict(range=[0, 1.1]), height=350)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.table(pd.DataFrame({"Métrica Académica": metricas_nombres, "Resultado": valores}))
            
            logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataque, (t_fin-t_ini))

# --- PESTAÑA 2: LENGUAJE ACADÉMICO ---
with tab2:
    st.header("Análisis Histórico de Vulnerabilidades")
    if os.path.exists("historial.csv"):
        df_h = pd.read_csv("historial.csv")
        st.subheader("Frecuencia de Incidentes por Puerto")
        # Aquí puedes llamar a logic para un análisis real de puertos
        st.info("Esta sección consolida la información de auditorías previas para identificar patrones de ataque.")
        st.dataframe(df_h, use_container_width=True)
