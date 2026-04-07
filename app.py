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

# --- LOGIN (TU CÓDIGO INTACTO) ---
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
    # --- NUEVO: SELECTOR DE FECHA PARA TESIS ---
    st.sidebar.divider()
    st.sidebar.subheader("📅 Simulación de Tiempo")
    fecha_simulada = st.sidebar.date_input("Fecha del Escaneo", value=pd.to_datetime("2026-04-01"))
    
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.clear(); st.rerun()

# Carga de archivos
@st.cache_resource
def load_assets():
    return tf.keras.models.load_model("modelo_cnn.keras"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

model, scaler, features_list = load_assets()

tab1, tab2 = st.tabs(["🚀 MONITOREO (Solo Admin)", "📊 BITÁCORA Y REPORTES"])

# PESTAÑA 1: MONITOREO (CON TU ANIMACIÓN)
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
                df_raw = pd.read_csv(archivo)
                df_raw.columns = df_raw.columns.str.strip()
                df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
                
                X = scaler.transform(df_clean[features_list]).reshape(-1, len(features_list), 1)
                preds, normal, ataque = [], 0, 0
                
                # TU BUCLE DE ANIMACIÓN INTACTO
                for i in range(0, len(X), 25):
                    res = (model.predict(X[i:i+25], verbose=0) > 0.5).astype(int).flatten()
                    for r in res:
                        preds.append(r)
                        if r == 1: ataque += 1
                        else: normal += 1
                    
                    m1.metric("Eventos Normales", normal)
                    m2.metric("Intrusiones Detectadas", ataque)
                    
                    fig = px.pie(values=[normal, ataque], names=['Normal', 'Ataque'], 
                                 color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                    fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                    p_plot.plotly_chart(fig, use_container_width=True, key=f"l_{i}")
                    
                    with t_data.container():
                        tmp = df_clean.iloc[max(0, i-5):i+25].copy()
                        tmp['Estado'] = ["⚠️ ATAQUE" if p == 1 else "✅ NORMAL" for p in preds[max(0, i-5):i+25]]
                        st.dataframe(tmp.iloc[:, [0, 1, 2, -1]], use_container_width=True)
                    time.sleep(0.05)

                # CÁLCULO DE PUERTO TOP PARA EL HISTORIAL
                p_top = logic.obtener_puerto_top(df_clean, preds)
                logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataque, (time.time()-t_ini), fecha_simulada, p_top)

                st.success(f"✅ Análisis finalizado. Datos guardados para el día {fecha_simulada}")
                st.divider()

                # SECCIÓN DE MÉTRICAS (MATRIZ Y BARRAS)
                col_label = next((c for c in df_clean.columns if c.lower() == 'label'), None)
                if col_label:
                    y_true = df_clean[col_label].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1)
                    acc, prec = accuracy_score(y_true, preds), precision_score(y_true, preds, zero_division=0)
                    rec, f1 = recall_score(y_true, preds, zero_division=0), f1_score(y_true, preds, zero_division=0)

                    c_mat, c_met = st.columns([2, 3])
                    with c_mat:
                        st.write("**Matriz de Confusión**")
                        cm, _ = logic.generar_metricas_detalladas(y_true, preds)
                        st.plotly_chart(px.imshow(cm, text_auto=True, x=['Pred: Normal', 'Pred: Ataque'], y=['Real: Normal', 'Real: Ataque'], color_continuous_scale='Blues'), use_container_width=True)
                    with c_met:
                        st.write("**Métricas de Precisión**")
                        fig_bar = go.Figure([go.Bar(x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], y=[acc, prec, rec, f1], marker_color='#3498db', text=[f"{v*100:.1f}%" for v in [acc, prec, rec, f1]], textposition='auto')])
                        st.plotly_chart(fig_bar, use_container_width=True)
    else: st.warning("🔒 Acceso Restringido. Use el perfil Administrador.")

# --- EN LA PESTAÑA 2: GRÁFICA ESTILO CAPTURAS ---
st.subheader("📈 Evolución Diaria de Capturas")

if os.path.exists("historial.csv"):
    df_h = pd.read_csv("historial.csv")
    df_h['Fecha_Dt'] = pd.to_datetime(df_h['Fecha'])
    df_h = df_h.sort_values('Fecha_Dt')
    
    # Creamos un nombre corto para el eje X (Ej: "Lun 01")
    df_h['Dia_Eje'] = df_h['Fecha_Dt'].dt.strftime('%a %d')

    # Creamos la gráfica
    fig_evolucion = px.line(
        df_h, 
        x='Dia_Eje', 
        y='Ataques_Detectados',
        markers=True,
        text='Ataques_Detectados' # Pone el número encima del punto
    )

    # --- AQUÍ ESTÁ LA MAGIA PARA QUE SE PAREZCA A TU FOTO ---
    fig_evolucion.update_traces(
        mode='lines+markers',
        line_shape='linear',
        marker=dict(
            symbol='square', # Puntos cuadrados como en tu imagen
            size=10, 
            color='#1f77b4', # Azul clásico
            line=dict(width=1, color='white')
        ),
        line=dict(width=3, color='#1f77b4'),
        textposition="top center"
    )

    # Configuración de la cuadrícula y el SCROLLBAR
    fig_evolucion.update_layout(
        xaxis=dict(
            showgrid=True, 
            gridcolor='lightgrey',
            # Esto activa el scrollbar abajo para "correr al lado"
            rangeslider=dict(visible=True), 
            type='category'
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgrey',
            range=[0, df_h['Ataques_Detectados'].max() + 2] # Para que no pegue al techo
        ),
        plot_bgcolor='rgba(240,240,240,0.5)', # Fondo gris suave como el de tu foto
        height=500
    )

    st.plotly_chart(fig_evolucion, use_container_width=True)
else:
    st.info("No hay datos para mostrar la evolución.")
