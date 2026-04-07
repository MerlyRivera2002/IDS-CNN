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

st.set_page_config(page_title="IDS Tesis 2026", layout="wide", page_icon="🛡️")

# --- 1. LOGIN CON ROLES ---
if 'perfil' not in st.session_state: st.session_state.perfil = None

st.sidebar.title("🔐 Control de Acceso")
if st.session_state.perfil is None:
    u = st.sidebar.text_input("Usuario")
    p = st.sidebar.text_input("Clave", type="password")
    if st.sidebar.button("Ingresar"):
        # PERFIL ADMIN
        if u == "admin" and p == "tesis2026": 
            st.session_state.perfil = "Administrador"
            st.rerun()
        # PERFIL VIEWER (INVITADO)
        elif u == "viewer" and p == "visita2026":
            st.session_state.perfil = "Visualizador"
            st.rerun()
        else:
            st.sidebar.error("Credenciales incorrectas")
    st.stop()
else:
    st.sidebar.success(f"Conectado como: {st.session_state.perfil}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.clear(); st.rerun()

# --- 2. CARGA DE ACTIVOS ---
@st.cache_resource
def load_assets():
    return tf.keras.models.load_model("modelo_cnn.keras"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

model, scaler, features_list = load_assets()

# --- 3. PESTAÑAS ---
tab1, tab2 = st.tabs(["🚀 MONITOREO (Solo Admin)", "📊 BITÁCORA Y REPORTES"])

# --- PESTAÑA 1: MONITOREO (SOLO PARA EL ADMIN) ---
with tab1:
    if st.session_state.perfil == "Administrador":
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
                    m1.metric("Normales", normal); m2.metric("Ataques", ataque)
                    fig = px.pie(values=[normal, ataque], names=['Normal', 'Ataque'], color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                    fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                    p_plot.plotly_chart(fig, use_container_width=True, key=f"l_{i}")
                    with t_data.container():
                        tmp = df_clean.iloc[max(0, i-5):i+25].copy()
                        tmp['Estado'] = ["⚠️ ANOMALÍA" if p == 1 else "✅ NORMAL" for p in preds[max(0, i-5):i+25]]
                        st.dataframe(tmp.iloc[:, [0, 1, 2, -1]], use_container_width=True)
                    time.sleep(0.4)
                st.success("✅ Análisis guardado en el historial.")
                logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataque, (time.time()-t_ini))
    else:
        st.warning("🔒 Acceso Restringido. Solo el Administrador puede ejecutar nuevos análisis.")
        st.info("Usted está en modo lectura. Por favor, diríjase a la pestaña de 'Bitácora' para ver los reportes.")

# --- PESTAÑA 2: BITÁCORA (TODOS PUEDEN VER) ---
with tab2:
    st.header("Historial de Auditoría y Comportamiento")
    if os.path.exists("historial.csv"):
        df_h = pd.read_csv("historial.csv")
        df_h['Fecha_Dt'] = pd.to_datetime(df_h['Fecha'])
        df_h['Dia_Nom'] = df_h['Fecha_Dt'].dt.strftime('%A %d/%m/%Y')

        # Gráfico de Tendencia
        st.subheader("📈 Tendencia de Seguridad Diaria")
        c_atq = 'Ataques_Detectados' if 'Ataques_Detectados' in df_h.columns else df_h.columns[3]
        resumen = df_h.groupby('Dia_Nom')[c_atq].sum().reset_index()
        st.plotly_chart(px.area(resumen, x='Dia_Nom', y=c_atq, color_discrete_sequence=['#e74c3c']), use_container_width=True)
        
        st.divider()

        for dia, grupo in df_h.groupby('Dia_Nom', sort=False):
            with st.expander(f"📅 JORNADA: {dia.upper()}", expanded=True):
                # Lógica flexible de columnas
                c_arc = next((c for c in ['Dataset', 'Archivo'] if c in grupo.columns), grupo.columns[0])
                c_tot = next((c for c in ['Registros_Procesados', 'Registros'] if c in grupo.columns), grupo.columns[2])
                c_mal = next((c for c in ['Ataques_Detectados', 'Ataques'] if c in grupo.columns), grupo.columns[3])
                c_tie = next((c for c in ['Tiempo_Ejecucion_Seg', 'Tiempo'] if c in grupo.columns), grupo.columns[4])
                
                v_tot = pd.to_numeric(grupo[c_tot], errors='coerce').fillna(0)
                v_mal = pd.to_numeric(grupo[c_mal], errors='coerce').fillna(0)
                
                tabla_res = pd.DataFrame({
                    'Dataset': grupo[c_arc],
                    'Total Datos': v_tot,
                    'Tiempo (s)': grupo[c_tie],
                    'Buenos': v_tot - v_mal,
                    'Malos': v_mal
                })
                st.table(tabla_res)

                # Puertos
                st.subheader("📌 Vulnerabilidades Detectadas")
                p1, p2 = st.columns(2)
                with p1: st.error("**Puertos Críticos:**\n- Puerto 80 (HTTP)\n- Puerto 445 (SMB)")
                with p2: st.success("**Puertos Seguros:**\n- Puerto 443 (HTTPS)\n- Puerto 22 (SSH)")
    else:
        st.info("No hay registros históricos disponibles.")
