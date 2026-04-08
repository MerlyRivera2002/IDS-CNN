import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import logic 

st.set_page_config(page_title="IDS Tesis 2026", layout="wide", page_icon="🛡️")

# --- LOGIN (Tu lógica original) ---
if 'perfil' not in st.session_state: st.session_state.perfil = None

st.sidebar.title("🔐 Control de Acceso")
if st.session_state.perfil is None:
    u = st.sidebar.text_input("Usuario")
    p = st.sidebar.text_input("Clave", type="password")
    if st.sidebar.button("Ingresar"):
        if u == "admin" and p == "tesis2026": 
            st.session_state.perfil = "Administrador"
            st.rerun()
        elif u == "viewer" and p == "visita2026":
            st.session_state.perfil = "Visualizador"
            st.rerun()
        else: st.sidebar.error("Credenciales incorrectas")
    st.stop()
else:
    st.sidebar.success(f"Conectado como: {st.session_state.perfil}")
    st.sidebar.divider()
    st.sidebar.subheader("📅 Simulación de Tiempo")
    fecha_simulada = st.sidebar.date_input("Fecha del Escaneo", value=pd.to_datetime("2026-04-01"))
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.clear(); st.rerun()

@st.cache_resource
def load_assets():
    return tf.keras.models.load_model("modelo_cnn.keras"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

model, scaler, features_list = load_assets()

tab1, tab2 = st.tabs(["🚀 MONITOREO (Solo Admin)", "📊 BITÁCORA Y REPORTES"])

# ----------------------------------------- PESTAÑA 1 (TUYA) ---------------------------------------------------
with tab1:
    if st.session_state.perfil == "Administrador":
        st.header("🛡️ Monitor de Tráfico en Tiempo Real")
        archivo = st.file_uploader("Subir dataset para simulación", type=["csv"], key="uploader_sim")
        
        if archivo:
            if st.button("🚀 INICIAR MONITOREO"):
                col_izq, col_der = st.columns([1, 1])
                with col_izq: espacio_pastel = st.empty()
                with col_der: espacio_metricas = st.empty()
                
                st.divider()
                st.subheader("🛰️ Registro de Actividad")
                espacio_tabla = st.empty()
                
                df_raw = pd.read_csv(archivo)
                df_raw.columns = df_raw.columns.str.strip()
                df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
                
                preds_totales = []
                t_inicio = time.time()
                
                for i in range(0, len(df_clean), 15): 
                    chunk = df_clean.iloc[i : i + 15]
                    X_chunk = scaler.transform(chunk[features_list]).reshape(-1, len(features_list), 1)
                    chunk_preds = (model.predict(X_chunk, verbose=0) > 0.5).astype(int).flatten()
                    preds_totales.extend(chunk_preds)
                    
                    ataques = sum(preds_totales)
                    normales = len(preds_totales) - ataques
                    
                    fig_pie = px.pie(values=[normales, ataques], names=['Seguro', 'Amenaza'], 
                                   color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.6)
                    fig_pie.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
                    espacio_pastel.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{i}")
                    
                    with espacio_metricas.container():
                        st.metric("CONEXIONES TOTALES", f"{len(preds_totales)}")
                        st.metric("INTRUSIONES DETECTADAS", f"{ataques}", delta=f"+{chunk_preds.sum()}", delta_color="inverse")

                    with espacio_tabla.container():
                        vista = chunk.copy()
                        vista['Estado'] = ["🚨 ATAQUE" if p == 1 else "✅ NORMAL" for p in chunk_preds]
                        def sugerir_amenaza(row):
                            if "NORMAL" in row['Estado']: return "Tráfico Seguro"
                            p = row['Destination Port']
                            if p in [80, 443]: return "Ataque Web (HTTP/S)"
                            if p == 22: return "Fuerza Bruta (SSH)"
                            if p == 21: return "Acceso FTP"
                            return "Escaneo / Port Scan"
                        vista['Diagnóstico'] = vista.apply(sugerir_amenaza, axis=1)
                        st.table(vista[['Destination Port', 'Estado', 'Diagnóstico']])
                    time.sleep(0.08)

                st.success("✅ Simulación finalizada.")
                st.divider()

                if col_label := next((c for c in df_clean.columns if c.lower() == 'label'), None):
                    y_true = df_clean[col_label].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1)[:len(preds_totales)]
                    st.subheader("📊 Evaluación Final")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("Matriz de Confusión")
                        st.plotly_chart(px.imshow(confusion_matrix(y_true, preds_totales), text_auto=True, color_continuous_scale='Reds'), use_container_width=True)
                    with c2:
                        acc = accuracy_score(y_true, preds_totales)
                        st.metric("ACCURACY DEL MODELO", f"{acc:.2%}")

                # LLAMADA AL LOGIC
                p_top = df_clean.iloc[:len(preds_totales)]['Destination Port'].mode()[0]
                logic.guardar_en_historial("historial.csv", archivo.name, len(preds_totales), ataques, (time.time()-t_inicio), fecha_simulada, p_top)
    else:
        st.warning("🔒 Acceso solo Admin.")

# ----------------------------------------- PESTAÑA 2 (ANÁLISIS) -----------------------------------------------
with tab2:
    st.header("📊 Inteligencia de Amenazas y Reportes Históricos")
    df_h = logic.obtener_metricas_resumen("historial.csv")
    
    if df_h is not None and not df_h.empty:
        # --- 1. GRÁFICO DE BARRAS: ATAQUES POR DÍA ---
        st.subheader("📅 Tendencia de Intrusiones por Fecha")
        fig_barras = px.bar(
            df_h, 
            x='Fecha', 
            y='Ataques_Detectados', 
            color='Puerto_Critico',
            title="Ataques Totales por Día (Color por Puerto más atacado)",
            text_auto=True
        )
        st.plotly_chart(fig_barras, use_container_width=True)

        # --- 2. TABLA DE BITÁCORA ---
        st.divider()
        st.subheader("📋 Historial Detallado")
        st.dataframe(df_h, use_container_width=True)
        
        if st.button("🗑️ Resetear Base de Datos"):
            os.remove("historial.csv"); st.rerun()
    else:
        st.info("No hay datos registrados aún. Realiza una simulación en la Pestaña 1.")
