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

# --- 1. LOGIN EN SIDEBAR (Lateral) ---
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

# --- 2. CARGA DE ASSETS ---
@st.cache_resource
def load_assets():
    return tf.keras.models.load_model("modelo_cnn.keras"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

model, scaler, features_list = load_assets()

# --- 3. PESTAÑAS ---
tab1, tab2 = st.tabs(["🚀 MONITOREO Y EVALUACIÓN", "📊 HISTORIAL DE AUDITORÍA"])

with tab1:
    st.header("Análisis de Tráfico de Red en Tiempo Real")
    archivo = st.file_uploader("Cargar archivo de tráfico (CSV)", type=["csv"])
    
    if archivo:
        if st.button("▶️ INICIAR ESCANEO"):
            # --- CREACIÓN DE ESPACIOS FIJOS (Para evitar multiplicaciones) ---
            col_metricas, col_grafico = st.columns([1, 2])
            
            # Objetos 'empty' para que Streamlit sepa que debe REEMPLAZAR, no añadir
            conteo_1 = col_metricas.empty()
            conteo_2 = col_metricas.empty()
            grafico_pie = col_grafico.empty()
            tabla_flujo = st.empty()
            
            t_ini = time.time()
            df_raw = pd.read_csv(archivo, nrows=1000)
            df_raw.columns = df_raw.columns.str.strip()
            df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
            X = scaler.transform(df_clean[features_list]).reshape(-1, len(features_list), 1)
            
            preds, normal, ataque = [], 0, 0
            paso = 25 
            
            for i in range(0, len(X), paso):
                res = (model.predict(X[i:i+paso], verbose=0) > 0.5).astype(int).flatten()
                for r in res:
                    preds.append(r)
                    if r == 1: ataque += 1
                    else: normal += 1
                
                # ACTUALIZACIÓN EN EL MISMO SITIO (Sin duplicar)
                conteo_1.metric("Eventos Normales", normal)
                conteo_2.metric("Intrusiones Detectadas", ataque)
                
                fig = px.pie(values=[normal, ataque], names=['Normal', 'Ataque'], 
                            color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0))
                grafico_pie.plotly_chart(fig, use_container_width=True, key=f"pie_id_{i}")

                with tabla_flujo.container():
                    st.write("**Inspección de flujo dinámico:**")
                    temp_df = df_clean.iloc[max(0, i-5):i+paso].copy()
                    temp_df['Resultado IA'] = ["⚠️ ANOMALÍA" if p == 1 else "✅ BENIGNO" for p in preds[max(0, i-5):i+paso]]
                    st.dataframe(temp_df.iloc[:, [0, 1, 2, -1]], use_container_width=True)
                
                time.sleep(0.4) # Velocidad controlada para defensa

            st.success("✅ Análisis finalizado. Resultados estadísticos generados debajo.")

            # --- SECCIÓN DE MÉTRICAS (Aparece una sola vez al final) ---
            st.divider()
            st.header("📈 Evaluación Estadística del Modelo")
            
            if 'Label' in df_clean.columns:
                y_real = df_clean['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
                acc, prec = accuracy_score(y_real, preds), precision_score(y_real, preds, zero_division=0)
                rec, f1 = recall_score(y_real, preds, zero_division=0), f1_score(y_real, preds, zero_division=0)
                
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.write("**Matriz de Confusión Académica**")
                    cm, _ = logic.generar_metricas_detalladas(y_real, preds)
                    st.plotly_chart(px.imshow(cm, text_auto=True, x=['Pred: Normal', 'Pred: Ataque'], 
                                             y=['Real: Normal', 'Real: Ataque'], color_continuous_scale='Blues'))

                with c2:
                    st.write("**Métricas de Rendimiento (Evaluación)**")
                    met_n = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    met_v = [acc, prec, rec, f1]
                    fig_bar = go.Figure([go.Bar(x=met_n, y=met_v, marker_color='#3498db', text=[f"{v:.4f}" for v in met_v], textposition='auto')])
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.table(pd.DataFrame({"Métrica": met_n, "Valor": met_v}))

            logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataque, (time.time()-t_ini))

with tab2:
    st.header("Bitácora de Auditoría")
    if os.path.exists("historial.csv"):
        st.dataframe(pd.read_csv("historial.csv"), use_container_width=True)
