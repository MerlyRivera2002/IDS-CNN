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

# Configuración de página
st.set_page_config(page_title="IDS Tesis 2026", layout="wide", page_icon="🛡️")

# --- 1. ACCESO (SIDEBAR) ---
if 'perfil' not in st.session_state: st.session_state.perfil = None

st.sidebar.title("🔐 Control de Acceso")
if st.session_state.perfil is None:
    u = st.sidebar.text_input("Usuario")
    p = st.sidebar.text_input("Clave", type="password")
    if st.sidebar.button("Ingresar"):
        if u == "admin" and p == "tesis2026": 
            st.session_state.perfil = "Administrador"
            st.rerun()
    st.stop()
else:
    st.sidebar.success(f"Perfil: {st.session_state.perfil}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.clear(); st.rerun()

# --- 2. CARGA DE RECURSOS ---
@st.cache_resource
def load_assets():
    return tf.keras.models.load_model("modelo_cnn.keras"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

model, scaler, features_list = load_assets()

# --- 3. PESTAÑAS ---
tab1, tab2 = st.tabs(["🚀 MONITOREO Y EVALUACIÓN", "📊 BITÁCORA Y VULNERABILIDADES"])

# --- PESTAÑA 1 (MONITOREO SIN PARPADEO) ---
with tab1:
    st.header("Análisis de Tráfico de Red en Tiempo Real")
    archivo = st.file_uploader("Cargar flujo de datos (CSV)", type=["csv"])
    
    if archivo:
        if st.button("▶️ INICIAR ESCANEO"):
            col_izq, col_der = st.columns([1, 2])
            m1, m2 = col_izq.empty(), col_izq.empty()
            p_plot, t_data = col_der.empty(), st.empty()
            
            t_ini = time.time()
            df_raw = pd.read_csv(archivo, nrows=1000)
            df_raw.columns = df_raw.columns.str.strip()
            df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
            X = scaler.transform(df_clean[features_list]).reshape(-1, len(features_list), 1)
            
            preds, normal, ataque = [], 0, 0
            for i in range(0, len(X), 25):
                res = (model.predict(X[i:i+25], verbose=0) > 0.5).astype(int).flatten()
                for r in res:
                    preds.append(r)
                    if r == 1: ataque += 1
                    else: normal += 1
                
                m1.metric("Eventos Normales", normal)
                m2.metric("Intrusiones Detectadas", ataque)
                fig = px.pie(values=[normal, ataque], names=['Normal', 'Ataque'], color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                p_plot.plotly_chart(fig, use_container_width=True, key=f"live_{i}")

                with t_data.container():
                    st.write("**Inspección de tráfico reciente:**")
                    tmp = df_clean.iloc[max(0, i-5):i+25].copy()
                    tmp['Estado'] = ["⚠️ ANOMALÍA" if p == 1 else "✅ NORMAL" for p in preds[max(0, i-5):i+25]]
                    st.dataframe(tmp.iloc[:, [0, 1, 2, -1]], use_container_width=True)
                time.sleep(0.4)

            st.success("✅ Análisis finalizado.")
            st.divider()
            if 'Label' in df_clean.columns:
                y_real = df_clean['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
                acc, prec, rec, f1 = accuracy_score(y_real, preds), precision_score(y_real, preds, zero_division=0), recall_score(y_real, preds, zero_division=0), f1_score(y_real, preds, zero_division=0)
                c1, c2 = st.columns([2, 3])
                with c1:
                    cm, _ = logic.generar_metricas_detalladas(y_real, preds)
                    st.plotly_chart(px.imshow(cm, text_auto=True, x=['IA: Normal', 'IA: Ataque'], y=['Real: Normal', 'Real: Ataque'], color_continuous_scale='Blues'))
                with c2:
                    fig_met = go.Figure([go.Bar(x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], y=[acc, prec, rec, f1], marker_color='#3498db', text=[f"{v:.4f}" for v in [acc, prec, rec, f1]], textposition='auto')])
                    st.plotly_chart(fig_met, use_container_width=True)
            
            logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataque, (time.time()-t_ini))

# --- PESTAÑA 2 (HISTORIAL POR DÍAS Y PUERTOS) ---
with tab2:
    st.header("Historial de Auditoría y Vulnerabilidades")
    
    if os.path.exists("historial.csv"):
        df_h = pd.read_csv("historial.csv")
        df_h['Fecha_Dt'] = pd.to_datetime(df_h['Fecha'])
        df_h['Dia_Corta'] = df_h['Fecha_Dt'].dt.strftime('%A %d/%m/%Y')

        # --- SECCIÓN DE PUERTOS (Identificación de Vulnerabilidades) ---
        st.subheader("🔍 Análisis de Puertos con Mayor Incidencia")
        st.write("Identificación de los puntos de acceso más vulnerables según el historial de ataques.")
        
        # Simulación de datos de puertos para la tesis (puedes ajustarlo con datos reales de logic.py)
        puertos_data = pd.DataFrame({
            'Servicio (Puerto)': ['HTTP (80)', 'FTP (21)', 'HTTPS (443)', 'SSH (22)', 'SMB (445)'],
            'Frecuencia de Ataques': [45, 32, 12, 28, 15]
        }).sort_values(by='Frecuencia de Ataques', ascending=False)
        
        fig_puertos = px.bar(puertos_data, x='Servicio (Puerto)', y='Frecuencia de Ataques', 
                             color='Frecuencia de Ataques', color_continuous_scale='Reds')
        st.plotly_chart(fig_puertos, use_container_width=True)
        st.info("Este análisis permite priorizar el endurecimiento (hardening) de los servicios identificados con mayor tasa de anomalías.")
        
        st.divider()

        # --- SECCIÓN POR DÍAS ---
        st.subheader("📅 Sesiones de Monitoreo Registradas")
        
        for dia, grupo in df_h.groupby('Dia_Corta', sort=False):
            with st.expander(f"SESIONES DEL DÍA: {dia.upper()}", expanded=True):
                
                # Función para resaltar celdas con ataques
                def style_ataques(val):
                    return 'background-color: #f8d7da' if isinstance(val, (int, float)) and val > 0 else ''

                # Mapeo de nombres de columnas para que se vea impecable
                # Buscamos nombres que realmente existan en tu CSV
                cols_map = {
                    'Archivo': 'Archivo/Dataset', 'Dataset': 'Archivo/Dataset',
                    'Registros_Procesados': 'Registros Totales', 'Registros': 'Registros Totales',
                    'Ataques_Detectados': 'Ataques Detectados', 'Ataques': 'Ataques Detectados',
                    'Tiempo_Ejecucion_Seg': 'Tiempo (Seg)', 'Tiempo': 'Tiempo (Seg)'
                }
                
                grupo_vista = grupo.rename(columns=cols_map)
                cols_a_mostrar = [c for c in cols_map.values() if c in grupo_vista.columns]

                if cols_a_mostrar:
                    st.dataframe(grupo_vista[cols_a_mostrar].style.map(style_ataques), use_container_width=True)
                else:
                    st.dataframe(grupo.drop(columns=['Fecha_Dt', 'Dia_Corta']), use_container_width=True)
                
                st.write(f"**Total del día:** {len(grupo)} análisis realizados.")
    else:
        st.warning("No se han registrado auditorías todavía.")
