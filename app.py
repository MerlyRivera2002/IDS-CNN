import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import logic
import os
import time
import plotly.express as px

# Configuración de página
st.set_page_config(page_title="IDS Tesis 2026", layout="wide")
st.title("🛡️ Sistema de Monitoreo de Red - Tesis 2026")

# Cargar IA
@st.cache_resource
def cargar_recursos():
    modelo = load_model('modelo_cnn.keras')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return modelo, scaler, features

model, scaler, features_list = cargar_recursos()

# --- SIDEBAR: CALENDARIO DE SIMULACIÓN ---
st.sidebar.header("📅 Simulación Temporal")
fecha_sel = st.sidebar.date_input("Fecha del Escaneo", value=pd.to_datetime("2026-04-01"))
st.sidebar.info("Cambia la fecha para simular el comportamiento de diferentes días de Abril.")

tab1, tab2 = st.tabs(["🔍 Detección Actual", "📈 Historial y Tendencias"])

# --- PESTAÑA 1: ESCANEO ---
with tab1:
    archivo = st.file_uploader("Subir tráfico de red (CSV)", type=["csv"])
    
    if archivo:
        df = pd.read_csv(archivo)
        
        if st.button("🔴 Iniciar Análisis Crítico"):
            with st.spinner("Analizando patrones de ataque..."):
                t_ini = time.time()
                
                # Procesar datos
                df_input = df[features_list]
                df_scaled = scaler.transform(df_input)
                
                # Predicción
                preds_prob = model.predict(df_scaled)
                preds = (preds_prob > 0.5).astype(int).flatten()
                
                num_ataques = int(preds.sum())
                t_fin = time.time() - t_ini
                
                # Detectar puerto más afectado
                p_critico = logic.obtener_puerto_critico(df, preds)
                
                # Guardar datos
                logic.guardar_en_historial("data_history.csv", archivo.name, len(df), num_ataques, t_fin, fecha_sel, p_critico)
                
                # Métricas visuales
                c1, c2, c3 = st.columns(3)
                c1.metric("Alertas de Ataque", num_ataques)
                c2.metric("Puerto en Riesgo", p_critico)
                c3.metric("Tiempo de Respuesta", f"{round(t_fin, 2)}s")
                
                st.success(f"Datos registrados para el día: {fecha_sel}")

# --- PESTAÑA 2: TENDENCIAS (CAPÍTULO 4) ---
with tab2:
    st.header("📊 Análisis Evolutivo de Seguridad")
    
    hist_file = "data_history.csv"
    if os.path.exists(hist_file) and os.path.getsize(hist_file) > 0:
        df_h = pd.read_csv(hist_file)
        df_h['Fecha'] = pd.to_datetime(df_h['Fecha'])
        df_h = df_h.sort_values("Fecha") # Ordenar cronológicamente
        
        # Gráfica de Líneas con Puntos (Scatter + Line)
        fig = px.line(df_h, x="Fecha", y="Ataques", markers=True,
                     title="Tendencia de Incidentes por Día",
                     labels={"Ataques": "Cantidad de Ataques", "Fecha": "Línea de Tiempo"})
        
        # Personalización de estilo
        fig.update_traces(line_color='#FF4B4B', marker=dict(size=12, symbol='diamond'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla detallada
        st.subheader("📋 Auditoría de Datos")
        st.dataframe(df_h, use_container_width=True)
        
        # Botón de descarga para el análisis final
        csv_data = df_h.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Datos para Capítulo 4", data=csv_data, file_name="auditoria_abril.csv")
        
    else:
        st.warning("No hay datos registrados aún. Realiza un escaneo para empezar a graficar.")
