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
        st.session_state.clear()
        st.rerun()

# --- 2. CARGA DE ACTIVOS ---
@st.cache_resource
def load_assets():
    m = tf.keras.models.load_model("modelo_cnn.keras")
    s = joblib.load("scaler.pkl")
    f = joblib.load("features.pkl")
    return m, s, f

model, scaler, features_list = load_assets()

# --- 3. PESTAÑAS ---
tab1, tab2 = st.tabs(["🚀 MONITOREO Y EVALUACIÓN", "📊 BITÁCORA POR DÍAS"])

# --- PESTAÑA 1: MONITOREO EN VIVO ---
with tab1:
    st.header("Análisis de Tráfico de Red en Tiempo Real")
    archivo = st.file_uploader("Cargar flujo de datos (CSV)", type=["csv"])
    
    if archivo:
        if st.button("▶️ INICIAR ESCANEO"):
            # Espacios fijos para evitar duplicidad de gráficos
            col_izq, col_der = st.columns([1, 2])
            cont_m1 = col_izq.empty()
            cont_m2 = col_izq.empty()
            cont_pie = col_der.empty()
            cont_tabla = st.empty()
            
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
                
                # Actualización de métricas y gráfico (Sobre escribe, no duplica)
                cont_m1.metric("Eventos Normales", normal)
                cont_m2.metric("Intrusiones Detectadas", ataque)
                
                fig = px.pie(values=[normal, ataque], names=['Normal', 'Ataque'], 
                            color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0), showlegend=True)
                cont_pie.plotly_chart(fig, use_container_width=True, key=f"p1_live_{i}")

                with cont_tabla.container():
                    st.write("**Flujo de paquetes procesados:**")
                    temp = df_clean.iloc[max(0, i-5):i+paso].copy()
                    temp['Clasificación'] = ["⚠️ ANOMALÍA" if p == 1 else "✅ NORMAL" for p in preds[max(0, i-5):i+paso]]
                    st.dataframe(temp.iloc[:, [0, 1, 2, -1]], use_container_width=True)
                
                time.sleep(0.4) # Velocidad pausada para exposición

            st.success("✅ Análisis finalizado. Resultados estadísticos debajo.")

            # --- SECCIÓN DE MÉTRICAS (Fijas al final) ---
            st.divider()
            st.header("📈 Evaluación Estadística del Modelo")
            
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

# --- PESTAÑA 2: BITÁCORA ORGANIZADA ---
with tab2:
    st.header("Historial de Auditoría por Fecha")
    
    if os.path.exists("historial.csv"):
        df_h = pd.read_csv("historial.csv")
        df_h['Fecha_Dt'] = pd.to_datetime(df_h['Fecha'])
        df_h['Dia_Corta'] = df_h['Fecha_Dt'].dt.strftime('%A %d/%m/%Y')

        # Agrupamos por día
        for dia, grupo in df_h.groupby('Dia_Corta', sort=False):
            with st.expander(f"📅 SESIONES DEL DÍA: {dia.upper()}", expanded=True):
                
                # Función para resaltar ataques
                def color_ataques(val):
                    color = '#f8d7da' if isinstance(val, (int, float)) and val > 0 else ''
                    return f'background-color: {color}'

                # Selección dinámica de columnas para evitar el KeyError
                # Esto busca los nombres que realmente existan en tu CSV
                cols_posibles = ['Archivo', 'Dataset', 'Registros_Procesados', 'Ataques_Detectados', 'Tiempo_Ejecucion_Seg', 'Ataques', 'Registros', 'Tiempo']
                cols_a_mostrar = [c for c in cols_posibles if c in grupo.columns]

                if cols_a_mostrar:
                    st.dataframe(grupo[cols_a_mostrar].style.applymap(color_ataques), use_container_width=True)
                else:
                    # Si no encuentra ninguna, muestra la tabla original sin la columna de fecha técnica
                    st.dataframe(grupo.drop(columns=['Fecha_Dt', 'Dia_Corta']), use_container_width=True)
                
                st.write(f"**Resumen:** Se ejecutaron {len(grupo)} auditorías este día.")
    else:
        st.info("No se han registrado análisis todavía.")
