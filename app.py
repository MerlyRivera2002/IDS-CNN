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

# --- LOGIN ---
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

# Carga de activos
@st.cache_resource
def load_assets():
    return tf.keras.models.load_model("modelo_cnn.keras"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

model, scaler, features_list = load_assets()

tab1, tab2 = st.tabs(["🚀 MONITOREO (Solo Admin)", "📊 BITÁCORA Y REPORTES"])

# ----------------------------------------- PESTAÑA 1 (TU ARQUITECTURA) -----------------------------------------------------------
with tab1:
    if st.session_state.perfil == "Administrador":
        st.header("🛡️ Monitor de Tráfico en Tiempo Real")
        archivo = st.file_uploader("Subir dataset para simulación", type=["csv"], key="uploader_sim")
        
        if archivo:
            if st.button("🚀 INICIAR MONITOREO"):
                # 1. CONTENEDORES PARA LA SIMULACIÓN EN VIVO
                col_izq, col_der = st.columns([1, 1])
                with col_izq:
                    espacio_pastel = st.empty()
                with col_der:
                    espacio_metricas = st.empty()
                
                st.divider()
                st.subheader("🛰️ Registro de Actividad")
                espacio_tabla = st.empty()
                
                # Procesamiento de datos
                df_raw = pd.read_csv(archivo)
                df_raw.columns = df_raw.columns.str.strip()
                df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
                
                preds_totales = []
                t_inicio = time.time()
                
                # 2. BUCLE DE SIMULACIÓN (FLUÍDO)
                for i in range(0, len(df_clean), 15): 
                    chunk = df_clean.iloc[i : i + 15]
                    X_chunk = scaler.transform(chunk[features_list]).reshape(-1, len(features_list), 1)
                    chunk_preds = (model.predict(X_chunk, verbose=0) > 0.5).astype(int).flatten()
                    preds_totales.extend(chunk_preds)
                    
                    ataques = sum(preds_totales)
                    normales = len(preds_totales) - ataques
                    
                    # A. Actualizar Gráfico de Pastel
                    fig_pie = px.pie(values=[normales, ataques], names=['Seguro', 'Amenaza'], 
                                   color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.6)
                    fig_pie.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
                    espacio_pastel.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{i}")
                    
                    # B. Actualizar Métricas
                    with espacio_metricas.container():
                        st.metric("CONEXIONES TOTALES", f"{len(preds_totales)}")
                        st.metric("INTRUSIONES DETECTADAS", f"{ataques}", delta=f"+{chunk_preds.sum()}", delta_color="inverse")

                    # C. Actualizar Tabla con Diagnóstico
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

                # 3. MÉTRICAS FINALES (TU DISEÑO ORIGINAL RE-ESTABLECIDO)
                st.subheader("📊 Evaluación del Rendimiento (Final)")
                col_label = next((c for c in df_clean.columns if c.lower() == 'label'), None)
                
                # Definimos acc para el historial (0.0 si no hay etiquetas)
                acc_historial = 0.0
                
                if col_label:
                    y_true = df_clean[col_label].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1)
                    y_true = y_true[:len(preds_totales)]
                    
                    acc_historial = accuracy_score(y_true, preds_totales)
                    prec = precision_score(y_true, preds_totales, zero_division=0)
                    rec = recall_score(y_true, preds_totales, zero_division=0)
                    f1 = f1_score(y_true, preds_totales, zero_division=0)

                    c_mat, c_line = st.columns([1, 1])
                    with c_mat:
                        st.write("**Matriz de Confusión**")
                        cm = confusion_matrix(y_true, preds_totales)
                        fig_cm = px.imshow(cm, text_auto=True, x=['Pred: Norm', 'Pred: Atq'], y=['Real: Norm', 'Real: Atq'], color_continuous_scale='Reds')
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with c_line:
                        st.write("**Gráfico de Rendimiento (Scores)**")
                        df_m = pd.DataFrame({
                            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                            'Valor': [acc_historial, prec, rec, f1]
                        })
                        fig_m = px.line(df_m, x='Métrica', y='Valor', markers=True, text=df_m['Valor'].apply(lambda x: f"{x:.2f}"))
                        fig_m.update_traces(line_color='#1f77b4', marker=dict(size=12, symbol='square', color='#ff7f0e'))
                        fig_m.update_layout(yaxis=dict(range=[0, 1.1]))
                        st.plotly_chart(fig_m, use_container_width=True)
                
                # --- GUARDAR EN HISTORIAL (CON LOS 8 PARÁMETROS) ---
                p_top = df_clean.iloc[:len(preds_totales)]['Destination Port'].mode()[0]
                
                # Aquí está el ajuste: pasamos los 8 valores que necesita logic.py
                logic.guardar_en_historial(
                    "historial.csv", 
                    archivo.name, 
                    len(preds_totales), 
                    ataques, 
                    (time.time()-t_inicio), 
                    fecha_simulada, 
                    p_top, 
                    acc_historial
                )
                st.toast("Simulación registrada en Bitácora")
    else:
        st.warning("🔒 Esta pestaña solo es accesible para Administradores.")

# ----------------------------------------- PESTAÑA 2 (BITÁCORA) -----------------------------------------------------------
with tab2:
    st.header("📊 Inteligencia de Red y Toma de Decisiones")
    
    df_h = logic.obtener_metricas_resumen("historial.csv")
    
    if df_h is not None and not df_h.empty:
        # 1. TENDENCIAS (Lo que ya tenías)
        st.subheader("📈 Tendencias del Sistema")
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.line(df_h, x='Fecha', y='Ataques', markers=True, title="Evolución de Intrusiones")
            fig1.update_traces(line_color='#00558c', marker=dict(size=10, symbol='square'))
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig_puertos = px.line(df_h, x='Fecha', y='Puerto', markers=True, title="Puerto Objetivo Principal")
            fig_puertos.update_traces(line_color='#f39c12', marker=dict(size=10, symbol='diamond'))
            fig_puertos.update_layout(yaxis=dict(type='category')) 
            st.plotly_chart(fig_puertos, use_container_width=True)
        
        st.divider()

        # --- NUEVA SECCIÓN: DISTRIBUCIÓN POR TIPO DE TRÁFICO ---
        st.subheader("🔍 Análisis de Puertos por Tipo de Tráfico")
        st.write("Comparativa de los puertos más frecuentes detectados en todas las simulaciones.")

        # Agrupamos datos para ver el volumen por puerto
        # (Asumiendo que df_h tiene los datos acumulados)
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.write("**Puertos en Tráfico Seguro (✅)**")
            # Simulamos una distribución basada en el historial para la visualización
            fig_normal = px.bar(df_h, x='Puerto', y='Total', # O la métrica que prefieras de volumen
                              title="Frecuencia: Tráfico Normal",
                              color_discrete_sequence=['#2ecc71'])
            fig_normal.update_layout(xaxis_title="Puerto", yaxis_title="Cantidad")
            st.plotly_chart(fig_normal, use_container_width=True)

        with col_p2:
            st.write("**Puertos en Tráfico Malicioso (🚨)**")
            fig_atq = px.bar(df_h, x='Puerto', y='Ataques', 
                           title="Frecuencia: Intrusiones",
                           color_discrete_sequence=['#e74c3c'])
            fig_atq.update_layout(xaxis_title="Puerto", yaxis_title="Cantidad")
            st.plotly_chart(fig_atq, use_container_width=True)
        
        st.divider()

        # 2. TABLA MAESTRA (TU TABLA BACÁN)
        st.subheader("📋 Matriz de Datos (Capítulo 4)")
        df_ver = df_h.copy()
        if 'Accuracy' in df_ver.columns:
            df_ver['Accuracy'] = df_ver['Accuracy'].fillna(0).apply(lambda x: f"{float(x):.2%}")
        
        st.dataframe(df_ver[['Fecha', 'Dataset', 'Total', 'Ataques', 'Puerto', 'Accuracy']], use_container_width=True)

        # 3. ACCIONES
        if st.button("🗑️ Borrar Historial"):
            if os.path.exists("historial.csv"): 
                os.remove("historial.csv")
                st.rerun()
    else:
        st.info("No hay datos históricos. Por favor, realiza una simulación en la Pestaña 1.")
