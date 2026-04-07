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

# --- 1. LOGIN EN SIDEBAR ---
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
            # Contenedores que se quedan fijos
            conteo_espacio = st.columns([1, 2])
            tabla_espacio = st.empty()
            
            t_ini = time.time()
            df_raw = pd.read_csv(archivo, nrows=1000)
            df_raw.columns = df_raw.columns.str.strip()
            df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
            X = scaler.transform(df_clean[features_list]).reshape(-1, len(features_list), 1)
            
            preds, normal, ataque = [], 0, 0
            paso = 25 # Bloques pequeños para mejor visualización
            
            for i in range(0, len(X), paso):
                res = (model.predict(X[i:i+paso], verbose=0) > 0.5).astype(int).flatten()
                for r in res:
                    preds.append(r)
                    if r == 1: ataque += 1
                    else: normal += 1
                
                # Actualización del Monitor (No se borra al final)
                with conteo_espacio[0].container():
                    st.metric("Normales", normal)
                    st.metric("Ataques", ataque)
                
                with conteo_espacio[1].container():
                    fig_pie = px.pie(values=[normal, ataque], names=['Normal', 'Ataque'], 
                                    color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                    fig_pie.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_pie, use_container_width=True, key=f"realtime_pie_{i}")

                with tabla_espacio.container():
                    st.write("**Flujo de paquetes procesados:**")
                    temp_df = df_clean.iloc[max(0, i-10):i+paso].copy()
                    temp_df['Clasificación'] = ["ANOMALÍA" if p == 1 else "BENIGNO" for p in preds[max(0, i-10):i+paso]]
                    st.dataframe(temp_df.iloc[:, [0, 1, 2, -1]], use_container_width=True)
                
                time.sleep(0.3) # Velocidad moderada para defensa de tesis

            st.success("✅ Análisis de flujo completo. Resultados consolidados debajo.")

            # --- SECCIÓN DE MÉTRICAS (APARECE DEBAJO DE LO ANTERIOR) ---
            st.divider()
            st.header("📈 Evaluación Estadística Final")
            
            if 'Label' in df_clean.columns:
                y_real = df_clean['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
                acc = accuracy_score(y_real, preds)
                prec = precision_score(y_real, preds, zero_division=0)
                rec = recall_score(y_real, preds, zero_division=0)
                f1 = f1_score(y_real, preds, zero_division=0)
                
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.subheader("Matriz de Confusión")
                    cm, _ = logic.generar_metricas_detalladas(y_real, preds)
                    fig_cm = px.imshow(cm, text_auto=True, x=['IA: Benigno', 'IA: Ataque'], 
                                      y=['Real: Benigno', 'Real: Ataque'], color_continuous_scale='Blues')
                    st.plotly_chart(fig_cm, use_container_width=True)

                with c2:
                    st.subheader("Indicadores Académicos")
                    nombres = ['Exactitud', 'Precisión', 'Sensibilidad', 'F1-Score']
                    valores = [acc, prec, rec, f1]
                    fig_met = go.Figure([go.Bar(x=nombres, y=valores, marker_color='#3498db', text=[f"{v:.4f}" for v in valores], textposition='auto')])
                    fig_met.update_layout(yaxis=dict(range=[0, 1.1]), height=350)
                    st.plotly_chart(fig_met, use_container_width=True)
                    st.table(pd.DataFrame({"Métrica": nombres, "Valor": valores}))

            logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataque, (time.time()-t_ini))

with tab2:
    st.header("Historial de Auditorías")
    if os.path.exists("historial.csv"):
        st.dataframe(pd.read_csv("historial.csv"), use_container_width=True)
