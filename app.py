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

st.set_page_config(page_title="IDS Tesis 2026", layout="wide")

# --- 1. ACCESO ---
if 'perfil' not in st.session_state: st.session_state.perfil = None

if st.session_state.perfil is None:
    st.title("🛡️ Acceso al Sistema de Detección")
    u = st.text_input("Usuario")
    p = st.text_input("Clave", type="password")
    if st.button("Entrar"):
        if u == "admin" and p == "tesis2026": st.session_state.perfil = "Admin"; st.rerun()
    st.stop()

# --- 2. CARGA ---
@st.cache_resource
def load_assets():
    return tf.keras.models.load_model("modelo_cnn.keras"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

model, scaler, features_list = load_assets()

tab1, tab2 = st.tabs(["🚀 MONITOR & EVALUACIÓN", "📊 HISTORIAL"])

with tab1:
    st.header("Análisis de Tráfico en Tiempo Real")
    archivo = st.file_uploader("Subir CSV", type=["csv"])
    
    if archivo:
        if st.button("▶️ INICIAR"):
            # ESTA ES LA CLAVE: Contenedores vacíos para no repetir gráficos
            placeholder_monitor = st.empty() 
            placeholder_tabla = st.empty()
            
            t_ini = time.time()
            df_raw = pd.read_csv(archivo, nrows=1000)
            df_raw.columns = df_raw.columns.str.strip()
            df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
            X = scaler.transform(df_clean[features_list]).reshape(-1, len(features_list), 1)
            
            preds, normal, ataque = [], 0, 0
            paso = 40 # Procesamos de 40 en 40 para que se vea el flujo
            
            for i in range(0, len(X), paso):
                res = (model.predict(X[i:i+paso], verbose=0) > 0.5).astype(int).flatten()
                for r in res:
                    preds.append(r)
                    if r == 1: ataque += 1
                    else: normal += 1
                
                # REESCRIBIMOS EL MISMO ESPACIO (No se duplica)
                with placeholder_monitor.container():
                    c1, c2 = st.columns([1, 2])
                    c1.metric("Normales", normal)
                    c1.metric("Ataques", ataque)
                    
                    fig = px.pie(values=[normal, ataque], names=['Normal', 'Ataque'], 
                                color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                    fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                    c2.plotly_chart(fig, use_container_width=True)

                with placeholder_tabla.container():
                    st.write("🔍 Inspección de flujo actual:")
                    temp_df = df_clean.iloc[max(0, i-5):i+paso].copy()
                    temp_df['Estado'] = ["⚠️ ATAQUE" if p == 1 else "✅ NORMAL" for p in preds[max(0, i-5):i+paso]]
                    # Mostramos solo columnas clave para que no se vea amontonado
                    st.dataframe(temp_df.iloc[:, [0,1,2,-1]], use_container_width=True)
                
                time.sleep(0.1)

            # --- AL TERMINAR: BORRAMOS EL MONITOR Y PONEMOS LAS MÉTRICAS ---
            placeholder_monitor.empty()
            placeholder_tabla.empty()
            st.success("✅ Simulación finalizada. Generando reporte académico...")

            # Métricas de Tesis
            if 'Label' in df_clean.columns:
                y_real = df_clean['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
                acc, prec, rec, f1 = accuracy_score(y_real, preds), precision_score(y_real, preds, zero_division=0), recall_score(y_real, preds, zero_division=0), f1_score(y_real, preds, zero_division=0)
                
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("Matriz de Confusión")
                    cm, _ = logic.generar_metricas_detalladas(y_real, preds)
                    st.plotly_chart(px.imshow(cm, text_auto=True, x=['IA: Normal', 'IA: Ataque'], y=['Real: Normal', 'Real: Ataque'], color_continuous_scale='Blues'))
                
                with col_right:
                    st.subheader("Indicadores de Rendimiento")
                    fig_met = go.Figure([go.Bar(x=['Accuracy', 'Precision', 'Recall', 'F1'], y=[acc, prec, rec, f1], marker_color='#3498db')])
                    st.plotly_chart(fig_met, use_container_width=True)
                    st.table(pd.DataFrame({"Indicador": ['Accuracy', 'Precision', 'Recall', 'F1'], "Valor": [acc, prec, rec, f1]}))

            logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataque, (time.time()-t_ini))

with tab2:
    st.header("Auditoría Histórica")
    if os.path.exists("historial.csv"):
        st.dataframe(pd.read_csv("historial.csv"), use_container_width=True)
