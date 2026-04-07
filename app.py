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

# Configuración de página profesional
st.set_page_config(page_title="IDS Tesis 2026", layout="wide", page_icon="🛡️")

# --- 1. ACCESO (SIDEBAR) ---
if 'perfil' not in st.session_state: st.session_state.perfil = None

st.sidebar.title("🔐 Control de Acceso")
if st.session_state.perfil is None:
    u = st.sidebar.text_input("Usuario")
    p = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Ingresar"):
        if u == "admin" and p == "tesis2026": 
            st.session_state.perfil = "Administrador"
            st.rerun()
    st.stop()
else:
    st.sidebar.success(f"Perfil: {st.session_state.perfil}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.clear()
        st.rerun()

# --- 2. RECURSOS ---
@st.cache_resource
def load_assets():
    return tf.keras.models.load_model("modelo_cnn.keras"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

model, scaler, features_list = load_assets()

# --- 3. INTERFAZ ---
tab1, tab2 = st.tabs(["🚀 MONITOREO Y EVALUACIÓN", "📊 BITÁCORA POR DÍAS"])

with tab1:
    st.header("Análisis de Tráfico de Red en Tiempo Real")
    archivo = st.file_uploader("Cargar flujo de datos (CSV)", type=["csv"])
    
    if archivo:
        if st.button("▶️ INICIAR ESCANEO"):
            # Creamos la estructura base UNA SOLA VEZ para evitar el parpadeo
            col_izq, col_der = st.columns([1, 2])
            
            # Contenedores vacíos que se actualizarán internamente
            m1 = col_izq.empty()
            m2 = col_izq.empty()
            p_plot = col_der.empty()
            t_data = st.empty()
            
            t_ini = time.time()
            df_raw = pd.read_csv(archivo, nrows=1000)
            df_raw.columns = df_raw.columns.str.strip()
            df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
            X = scaler.transform(df_clean[features_list]).reshape(-1, len(features_list), 1)
            
            preds, normal, ataque = [], 0, 0
            paso = 25 
            
            # --- BUCLE DE PROCESAMIENTO FLUIDO ---
            for i in range(0, len(X), paso):
                res = (model.predict(X[i:i+paso], verbose=0) > 0.5).astype(int).flatten()
                for r in res:
                    preds.append(r)
                    if r == 1: ataque += 1
                    else: normal += 1
                
                # Actualizamos solo el CONTENIDO de los contenedores
                m1.metric("Eventos Normales", normal)
                m2.metric("Intrusiones Detectadas", ataque)
                
                fig = px.pie(values=[normal, ataque], names=['Normal', 'Ataque'], 
                            color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                
                # key dinámica pero interna para evitar recarga de página
                p_plot.plotly_chart(fig, use_container_width=True, key=f"f_{i}")

                with t_data.container():
                    st.write("**Inspección de tráfico reciente:**")
                    temp = df_clean.iloc[max(0, i-5):i+paso].copy()
                    temp['Estado'] = ["⚠️ ANOMALÍA" if p == 1 else "✅ NORMAL" for p in preds[max(0, i-5):i+paso]]
                    st.dataframe(temp.iloc[:, [0, 1, 2, -1]], use_container_width=True)
                
                time.sleep(0.4) # Velocidad estable

            st.success("✅ Análisis finalizado. Los resultados se mantienen para su evaluación.")
            st.divider()
            
            # --- SECCIÓN DE MÉTRICAS (Al final) ---
            if 'Label' in df_clean.columns:
                y_real = df_clean['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
                acc, prec = accuracy_score(y_real, preds), precision_score(y_real, preds, zero_division=0)
                rec, f1 = recall_score(y_real, preds, zero_division=0), f1_score(y_real, preds, zero_division=0)
                
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.subheader("Matriz de Confusión")
                    cm, _ = logic.generar_metricas_detalladas(y_real, preds)
                    st.plotly_chart(px.imshow(cm, text_auto=True, x=['IA: Normal', 'IA: Ataque'], 
                                             y=['Real: Normal', 'Real: Ataque'], color_continuous_scale='Blues'))
                with c2:
                    st.subheader("Indicadores de Rendimiento")
                    fig_met = go.Figure([go.Bar(x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                                              y=[acc, prec, rec, f1], marker_color='#3498db', 
                                              text=[f"{v:.4f}" for v in [acc, prec, rec, f1]], textposition='auto')])
                    st.plotly_chart(fig_met, use_container_width=True)
                    st.table(pd.DataFrame({"Métrica": ['Accuracy', 'Precision', 'Recall', 'F1-Score'], "Valor": [acc, prec, rec, f1]}))

            logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataque, (time.time()-t_ini))

with tab2:
    st.header("Historial de Auditoría")
    if os.path.exists("historial.csv"):
        df_h = pd.read_csv("historial.csv")
        df_h['Fecha_Dt'] = pd.to_datetime(df_h['Fecha'])
        df_h['Dia_Formato'] = df_h['Fecha_Dt'].dt.strftime('%A %d/%m/%Y')

        for dia, grupo in df_h.groupby('Dia_Formato', sort=False):
            with st.expander(f"📅 REGISTROS DEL DÍA: {dia.upper()}", expanded=True):
                def estilo_fila(val):
                    return 'background-color: #f8d7da' if isinstance(val, (int, float)) and val > 0 else ''

                cols_validas = [c for c in ['Archivo', 'Dataset', 'Registros', 'Ataques', 'Tiempo', 'Ataques_Detectados', 'Registros_Procesados'] if c in grupo.columns]
                
                if cols_validas:
                    # Usamos .map para evitar el error de versiones nuevas de Pandas
                    st.dataframe(grupo[cols_validas].style.map(estilo_fila), use_container_width=True)
                else:
                    st.dataframe(grupo.drop(columns=['Fecha_Dt', 'Dia_Formato']), use_container_width=True)
                
                st.write(f"**Resumen:** {len(grupo)} auditorías completadas.")
    else:
        st.info("No hay registros en la bitácora.")
