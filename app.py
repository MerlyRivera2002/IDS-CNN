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
                    fig_pie.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
                    espacio_pastel.plotly_chart(fig_pie, use_container_width=True, key=f"p_{i}")
                    
                    with espacio_metricas.container():
                        st.metric("CONEXIONES TOTALES", f"{len(preds_totales)}")
                        st.metric("INTRUSIONES", f"{ataques}", delta=f"+{chunk_preds.sum()}", delta_color="inverse")

                    with espacio_tabla.container():
                        vista = chunk.copy()
                        vista['Estado'] = ["🚨 ATAQUE" if p == 1 else "✅ NORMAL" for p in chunk_preds]
                        def sugerir_amenaza(row):
                            if "NORMAL" in row['Estado']: return "Tráfico Seguro"
                            p = row['Destination Port']
                            if p in [80, 443]: return "Ataque Web"
                            if p == 22: return "Fuerza Bruta SSH"
                            return "Port Scan"
                        vista['Diagnóstico'] = vista.apply(sugerir_amenaza, axis=1)
                        st.table(vista[['Destination Port', 'Estado', 'Diagnóstico']])
                    time.sleep(0.05)

                st.success("✅ Simulación finalizada.")
                
                # Evaluación Final
                col_label = next((c for c in df_clean.columns if c.lower() == 'label'), None)
                acc = 0 # Valor por defecto
                if col_label:
                    y_true = df_clean[col_label].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1)[:len(preds_totales)]
                    acc = accuracy_score(y_true, preds_totales)
                    # (Aquí irían los gráficos de matriz que ya tenías)
                    st.write(f"**Accuracy Final:** {acc:.4f}")

                p_top = logic.obtener_puerto_top(df_clean.iloc[:len(preds_totales)], preds_totales)
                # LLAMADA CORREGIDA A LOGIC
                logic.guardar_en_historial("historial.csv", archivo.name, len(preds_totales), ataques, (time.time()-t_inicio), fecha_simulada, p_top, acc)
    else:
        st.warning("🔒 Solo Administradores.")

# ----------------------------------------- PESTAÑA 2 -----------------------------------------------------------
with tab2:
    st.header("📊 Inteligencia de Amenazas y Reportes")
    
    if os.path.exists("historial.csv"):
        try:
            df_h = pd.read_csv("historial.csv")
            # Forzar columnas si faltan por archivos viejos
            if 'Accuracy' not in df_h.columns: df_h['Accuracy'] = 0.0
            
            df_h['Fecha_Dt'] = pd.to_datetime(df_h['Fecha'])
            df_h = df_h.sort_values('Fecha_Dt')

            # 1. Gráfico de Tendencia (Líneas con Puntos)
            st.subheader("📈 Tendencia Diaria de Ataques")
            fig1 = px.line(df_h, x='Fecha', y='Ataques_Detectados', markers=True, text='Ataques_Detectados')
            fig1.update_traces(line_color='#1f77b4', marker=dict(symbol='square', size=10))
            st.plotly_chart(fig1, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("🔑 Puertos Críticos")
                fig2 = px.line(df_h, x='Fecha', y='Ataques_Detectados', color='Puerto_Critico', markers=True)
                st.plotly_chart(fig2, use_container_width=True)
            with col_b:
                st.subheader("🎯 Estabilidad de la IA")
                fig3 = px.line(df_h, x='Fecha', y='Accuracy', markers=True)
                fig3.update_traces(line_color='#2ecc71')
                st.plotly_chart(fig3, use_container_width=True)

            st.divider()
            st.dataframe(df_h, use_container_width=True)
            st.download_button("📥 Descargar Reporte CSV", df_h.to_csv(index=False), "reporte.csv")
            
            if st.button("🔥 Borrar Historial (Limpiar Todo)"):
                os.remove("historial.csv"); st.rerun()

        except Exception as e:
            st.error("Error al leer el historial. Es probable que el formato sea antiguo.")
            if st.button("Resetear Historial Corrupto"):
                os.remove("historial.csv"); st.rerun()
    else:
        st.info("No hay datos. Realiza una simulación en la Pestaña 1.")
