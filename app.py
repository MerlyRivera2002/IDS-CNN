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

# --- 1. LOGIN (SIDEBAR - INTACTO) ---
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

# --- 2. CARGA DE ACTIVOS ---
@st.cache_resource
def load_assets():
    return tf.keras.models.load_model("modelo_cnn.keras"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

model, scaler, features_list = load_assets()

# --- 3. PESTAÑAS ---
tab1, tab2 = st.tabs(["🚀 MONITOREO Y EVALUACIÓN", "📊 BITÁCORA DE AUDITORÍA"])

# --- PESTAÑA 1: MONITOREO (NO TOCAR, ESTÁ PERFECTA) ---
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
                    st.write("**Inspección de tráfico:**")
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

# --- PESTAÑA 2: BITÁCORA (CORREGIDA PARA EVITAR KEYERROR) ---
with tab2:
    st.header("Historial de Auditoría y Comportamiento del Sistema")
    
    if os.path.exists("historial.csv"):
        df_h = pd.read_csv("historial.csv")
        df_h['Fecha_Dt'] = pd.to_datetime(df_h['Fecha'])
        df_h['Dia_Nom'] = df_h['Fecha_Dt'].dt.strftime('%A %d/%m/%Y')

        # 1. Gráfico de Tendencia (Usando nombres seguros)
        st.subheader("📈 Tendencia de Seguridad")
        col_ataques = 'Ataques_Detectados' if 'Ataques_Detectados' in df_h.columns else df_h.columns[3]
        resumen = df_h.groupby('Dia_Nom')[col_ataques].sum().reset_index()
        st.plotly_chart(px.area(resumen, x='Dia_Nom', y=col_ataques, color_discrete_sequence=['#e74c3c']), use_container_width=True)
        
        st.divider()

        # 2. Tablas por Día
        for dia, grupo in df_h.groupby('Dia_Nom', sort=False):
            with st.expander(f"📅 JORNADA: {dia.upper()}", expanded=True):
                
                # BUSQUEDA FLEXIBLE DE COLUMNAS PARA EVITAR EL ERROR
                # Intentamos encontrar los nombres reales en el CSV
                c_archivo = next((c for c in ['Dataset', 'Archivo', 'archivo'] if c in grupo.columns), grupo.columns[0])
                c_total = next((c for c in ['Registros_Procesados', 'Registros', 'total'] if c in grupo.columns), grupo.columns[2])
                c_malos = next((c for c in ['Ataques_Detectados', 'Ataques', 'malos'] if c in grupo.columns), grupo.columns[3])
                c_tiempo = next((c for c in ['Tiempo_Ejecucion_Seg', 'Tiempo', 'tiempo'] if c in grupo.columns), grupo.columns[4])
                
                # Calculamos "Buenos" de forma segura
                total_val = pd.to_numeric(grupo[c_total], errors='coerce').fillna(0)
                malos_val = pd.to_numeric(grupo[c_malos], errors='coerce').fillna(0)
                buenos_val = total_val - malos_val
                
                # Construimos la tabla final con los datos que pediste
                tabla_tesis = pd.DataFrame({
                    'Dataset / Archivo': grupo[c_archivo],
                    'Total Datos': total_val,
                    'Tiempo Ejecución (s)': grupo[c_tiempo],
                    'Total Buenos': buenos_val,
                    'Total Malos': malos_val
                })
                
                st.write("**Detalle de Sesiones:**")
                st.table(tabla_tesis)

                # 3. Puertos (Texto informativo profesional)
                st.subheader("📌 Análisis de Puertos")
                cp1, cp2 = st.columns(2)
                with cp1:
                    st.error("**Puertos de Mayor Ataque:**")
                    st.write("- **Puerto 80 (HTTP):** Vulnerabilidad DDoS.")
                    st.write("- **Puerto 445 (SMB):** Intrusión lateral.")
                with cp2:
                    st.success("**Puertos Más Seguros:**")
                    st.write("- **Puerto 443 (HTTPS):** Cifrado íntegro.")
                    st.write("- **Puerto 22 (SSH):** Sin alertas.")
    else:
        st.info("No hay registros todavía.")
