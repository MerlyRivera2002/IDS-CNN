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
# ----------------------------------------- PESTAÑA 1 (CORREGIDA) -----------------------------------------------------------
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
                
                # 2. BUCLE DE SIMULACIÓN
                for i in range(0, len(df_clean), 15): 
                    chunk = df_clean.iloc[i : i + 15]
                    X_chunk = scaler.transform(chunk[features_list]).reshape(-1, len(features_list), 1)
                    chunk_preds = (model.predict(X_chunk, verbose=0) > 0.5).astype(int).flatten()
                    preds_totales.extend(chunk_preds)
                    
                    ataques = sum(preds_totales)
                    normales = len(preds_totales) - ataques
                    
                    # Gráficos (Tu arquitectura intacta)
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
                        st.table(vista[['Destination Port', 'Estado']].tail(5)) # Simplificado para velocidad
                    
                    time.sleep(0.01)

                # --- ESTO ES LO QUE ESTABA MAL: EL GUARDADO DEBE IR AQUÍ ---
                st.success("✅ Simulación finalizada.")
                
                # Definimos acc con valor por defecto SIEMPRE
                acc_para_guardar = 0.0 
                
                col_label = next((c for c in df_clean.columns if c.lower() == 'label'), None)
                if col_label:
                    y_true = df_clean[col_label].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1)
                    y_true = y_true[:len(preds_totales)]
                    acc_para_guardar = accuracy_score(y_true, preds_totales)
                    # Aquí podrías poner tus gráficas de matriz de confusión si quieres...
                
                p_top = df_clean.iloc[:len(preds_totales)]['Destination Port'].mode()[0]
                
                # GUARDAR (Con los 8 argumentos exactos que pide tu logic.py)
                logic.guardar_en_historial(
                    "historial.csv", 
                    archivo.name, 
                    len(preds_totales), 
                    ataques, 
                    (time.time()-t_inicio), 
                    fecha_simulada, 
                    p_top, 
                    acc_para_guardar
                )
                st.balloons() # Para avisarte que sí guardó
    else:
        st.warning("🔒 Esta pestaña solo es accesible para Administradores.")

# 3. MÉTRICAS FINALES (TU DISEÑO ORIGINAL)
                st.subheader("📊 Evaluación del Rendimiento (Final)")
                col_label = next((c for c in df_clean.columns if c.lower() == 'label'), None)
                
                # --- ESTO ASEGURA QUE SIEMPRE HAYA UN VALOR PARA GUARDAR ---
                acc_final = 0.0 
                
                if col_label:
                    y_true = df_clean[col_label].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1)
                    y_true = y_true[:len(preds_totales)]
                    
                    acc_final = accuracy_score(y_true, preds_totales)
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
                            'Valor': [acc_final, prec, rec, f1]
                        })
                        fig_m = px.line(df_m, x='Métrica', y='Valor', markers=True, text=df_m['Valor'].apply(lambda x: f"{x:.2f}"))
                        # Mantenemos tus colores y símbolos originales
                        fig_m.update_traces(line_color='#1f77b4', marker=dict(size=12, symbol='square', color='#ff7f0e'))
                        fig_m.update_layout(yaxis=dict(range=[0, 1.1]))
                        st.plotly_chart(fig_m, use_container_width=True)
                
                # --- GUARDAR EN HISTORIAL (AQUÍ ESTÁ LA CONEXIÓN CORRECTA) ---
                p_top = df_clean.iloc[:len(preds_totales)]['Destination Port'].mode()[0]
                
                # Pasamos los 8 argumentos que logic.py necesita
                logic.guardar_en_historial(
                    "historial.csv", 
                    archivo.name, 
                    len(preds_totales), 
                    ataques, 
                    (time.time()-t_inicio), 
                    fecha_simulada, 
                    p_top, 
                    acc_final # <--- EL DATO QUE FALTABA
                )
                st.success("✅ Simulación finalizada y datos registrados en Bitácora.")

# ----------------------------------------- PESTAÑA 2 (BITÁCORA) -----------------------------------------------------------
with tab2:
    st.header("📊 Inteligencia de Red y Toma de Decisiones")
    
    df_h = logic.obtener_metricas_resumen("historial.csv")
    
    if df_h is not None and not df_h.empty:
        # Aseguramos columnas para la tabla
        if 'Normales' not in df_h.columns:
            df_h['Normales'] = df_h['Total'] - df_h['Ataques']
            
        # 1. GRÁFICA DE TENDENCIA
        st.subheader("📈 Tendencia de Ataques")
        fig1 = px.line(df_h, x='Fecha', y='Ataques', markers=True)
        fig1.update_traces(line_color='#00558c', marker=dict(size=10, symbol='square'))
        st.plotly_chart(fig1, use_container_width=True)
            
        st.divider()

        # 2. TABLA MAESTRA
        st.subheader("📋 Matriz de Datos (Capítulo 4)")
        df_ver = df_h.copy()
        if 'Accuracy' in df_ver.columns:
            df_ver['Accuracy'] = df_ver['Accuracy'].apply(lambda x: f"{float(x):.2%}")
        
        st.dataframe(df_ver[['Fecha', 'Dataset', 'Total', 'Normales', 'Ataques', 'Puerto', 'Accuracy']], use_container_width=True)

        # 3. ACCIONES
        if st.button("🗑️ Borrar Historial"):
            if os.path.exists("historial.csv"): os.remove("historial.csv")
            st.rerun()
    else:
        st.info("No hay datos históricos. Por favor, realiza una simulación en la Pestaña 1.")
