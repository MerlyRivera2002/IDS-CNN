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

# --- ---------------------------------------------DENTRO DE TU PESTAÑA 1 -------------------------------------------------------------
with tab1:
    st.header("🛡️ Monitor de Tráfico en Tiempo Real")
    archivo = st.file_uploader("Subir dataset para simulación", type=["csv"])
    
    if archivo:
        if st.button("🚀 INICIAR MONITOREO"):
            # 1. CREAMOS LOS CONTENEDORES VACÍOS (Para evitar el parpadeo)
            col_izq, col_der = st.columns([1, 1])
            with col_izq:
                espacio_pastel = st.empty()
            with col_der:
                espacio_metricas = st.empty()
            
            st.write("---")
            st.subheader("🛰️ Registro de Actividad")
            espacio_tabla = st.empty()
            
            # Procesamiento inicial
            df_raw = pd.read_csv(archivo)
            # ... (Aquí va tu limpieza de datos y preparación) ...
            
            preds_totales = []
            
            # 2. EL BUCLE DE SIMULACIÓN (Velocidad controlada)
            # Procesamos de 10 en 10 para que sea más fluido
            for i in range(0, len(df_clean), 10):
                chunk = df_clean.iloc[i : i + 10]
                # Inferencia de la IA
                X_chunk = scaler.transform(chunk[features_list]).reshape(-1, len(features_list), 1)
                chunk_preds = (model.predict(X_chunk, verbose=0) > 0.5).astype(int).flatten()
                preds_totales.extend(chunk_preds)
                
                # CÁLCULOS PARA VISUALIZACIÓN
                ataques = sum(preds_totales)
                normales = len(preds_totales) - ataques
                
                # A. Actualizar Gráfico de Pastel (Sin parpadeo)
                fig_pie = px.pie(values=[normales, ataques], names=['Seguro', 'Amenaza'], 
                               color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.6)
                fig_pie.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
                espacio_pastel.plotly_chart(fig_pie, use_container_width=True, key=f"p_{i}")
                
                # B. Actualizar Métricas (Estilo Dashboard Pro)
                with espacio_metricas.container():
                    st.metric("CONEXIONES TOTALES", len(preds_totales))
                    st.metric("INTRUSIONES DETECTADAS", ataques, delta=f"+{chunk_preds.sum()}", delta_color="inverse")

                # C. Actualizar Tabla (Con la columna de "Naturaleza")
                with espacio_tabla.container():
                    # Tomamos los últimos 15 para la vista rápida
                    vista = chunk.copy()
                    vista['Estado'] = ["🚨 ATAQUE" if p == 1 else "✅ NORMAL" for p in chunk_preds]
                    
                    # AQUÍ ESTÁ TU COLUMNA DE NATURALEZA
                    def sugerir_amenaza(row):
                        if "NORMAL" in row['Estado']: return "Tráfico Seguro"
                        p = row['Destination Port']
                        if p in [80, 443]: return "Ataque Web (HTTP/S)"
                        if p == 22: return "Fuerza Bruta (SSH)"
                        if p == 21: return "Acceso FTP no aut."
                        return "Escaneo / Port Scan"
                    
                    vista['Diagnóstico'] = vista.apply(sugerir_amenaza, axis=1)
                    
                    # Mostramos solo columnas clave para que no se vea ancho
                    st.table(vista[['Destination Port', 'Estado', 'Diagnóstico']])

                # 3. EL "PASO" (Control de velocidad)
                # 0.1 es una velocidad natural. 0.05 es más rápido pero visible.
                time.sleep(0.1) 

            st.success("✅ Simulación finalizada. Resultados guardados en el historial.")

# --------------------------------------------- EN LA PESTAÑA 2: GRÁFICA ESTILO CAPTURAS -----------------------------------------------
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
