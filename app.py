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

# ----------------------------------------- PESTAÑA 1 -----------------------------------------------------------
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

                # 3. MÉTRICAS FINALES
                st.subheader("📊 Evaluación del Rendimiento (Final)")
                col_label = next((c for c in df_clean.columns if c.lower() == 'label'), None)
                
                if col_label:
                    y_true = df_clean[col_label].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1)
                    y_true = y_true[:len(preds_totales)]
                    
                    acc = accuracy_score(y_true, preds_totales)
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
                            'Valor': [acc, prec, rec, f1]
                        })
                        fig_m = px.line(df_m, x='Métrica', y='Valor', markers=True, text=df_m['Valor'].apply(lambda x: f"{x:.2f}"))
                        fig_m.update_traces(line_color='#1f77b4', marker=dict(size=12, symbol='square', color='#ff7f0e'))
                        fig_m.update_layout(yaxis=dict(range=[0, 1.1]))
                        st.plotly_chart(fig_m, use_container_width=True)
                
                # GUARDAR EN HISTORIAL
                p_top = df_clean.iloc[:len(preds_totales)]['Destination Port'].mode()[0]
                logic.guardar_en_historial("historial.csv", archivo.name, len(preds_totales), ataques, (time.time()-t_inicio), fecha_simulada, p_top)
    else:
        st.warning("🔒 Esta pestaña solo es accesible para Administradores.")

# ----------------------------------------- PESTAÑA 2 -----------------------------------------------------------
with tab2:
    st.header("📊 Bitácora de Capturas")
    st.subheader("📈 Evolución Diaria de Capturas")

    if os.path.exists("historial.csv"):
        df_h = pd.read_csv("historial.csv")
        df_h['Fecha_Dt'] = pd.to_datetime(df_h['Fecha'])
        df_h = df_h.sort_values('Fecha_Dt')
        df_h['Dia_Eje'] = df_h['Fecha_Dt'].dt.strftime('%a %d')

        fig_evolucion = px.line(
            df_h, x='Dia_Eje', y='Ataques_Detectados',
            markers=True, text='Ataques_Detectados'
        )

        fig_evolucion.update_traces(
            mode='lines+markers', line_shape='linear',
            marker=dict(symbol='square', size=10, color='#1f77b4', line=dict(width=1, color='white')),
            line=dict(width=3, color='#1f77b4'),
            textposition="top center"
        )

        fig_evolucion.update_layout(
            xaxis=dict(showgrid=True, gridcolor='lightgrey', rangeslider=dict(visible=True), type='category'),
            yaxis=dict(showgrid=True, gridcolor='lightgrey', range=[0, df_h['Ataques_Detectados'].max() + 5]),
            plot_bgcolor='rgba(240,240,240,0.5)', height=500
        )

        st.plotly_chart(fig_evolucion, use_container_width=True)
        
        st.divider()
        st.subheader("📋 Registros Históricos")
        
        # Lista de columnas que QUEREMOS mostrar
        cols_deseadas = ['Fecha', 'Archivo', 'Total_Registros', 'Ataques_Detectados', 'Puerto_Critico']
        
        # Filtramos solo las que DE VERDAD existen en el archivo para evitar el KeyError
        cols_reales = [c for c in cols_deseadas if c in df_h.columns]
        
        if len(cols_reales) > 0:
            st.dataframe(df_h[cols_reales], use_container_width=True)
        else:
            # Si no encuentra ninguna, muestra todo el dataframe para que veas qué nombres tienen
            st.write("Mostrando todas las columnas disponibles:")
            st.dataframe(df_h, use_container_width=True)
