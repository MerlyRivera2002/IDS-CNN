import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import logic
import os
import time
import plotly.express as px

# Configuración inicial
st.set_page_config(page_title="IDS Tesis 2026", layout="wide")
st.title("🛡️ Sistema de Detección de Intrusos (IDS)")

# Cargar Modelo y Scaler
@st.cache_resource
def cargar_recursos():
    modelo = load_model('modelo_cnn.keras')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return modelo, scaler, features

model, scaler, features_list = cargar_recursos()

# --- SIDEBAR: SIMULACIÓN ---
st.sidebar.header("⚙️ Configuración de Simulación")
fecha_simulada = st.sidebar.date_input("Selecciona Fecha del Análisis", value=pd.to_datetime("2026-04-01"))
st.sidebar.info("Usa este calendario para simular diferentes días de Abril para tu Tesis.")

tab1, tab2 = st.tabs(["🔍 Detección en Tiempo Real", "📊 Tendencias y Reportes"])

# --- PESTAÑA 1: DETECCIÓN ---
with tab1:
    archivo = st.file_uploader("Sube tu dataset (CSV)", type=["csv"])
    
    if archivo:
        df = pd.read_csv(archivo)
        st.write(f"Archivo cargado: {archivo.name} ({len(df)} filas)")
        
        if st.button("🚀 Iniciar Escaneo Crítico"):
            t_ini = time.time()
            
            # Preprocesamiento
            df_input = df[features_list]
            df_scaled = scaler.transform(df_input)
            
            # Predicción
            preds_prob = model.predict(df_scaled)
            preds = (preds_prob > 0.5).astype(int).flatten()
            
            num_ataques = int(preds.sum())
            t_fin = time.time() - t_ini
            
            # Identificar Puerto Crítico
            puerto_top = logic.obtener_puerto_critico(df, preds)
            
            # Guardar en data_history.csv
            logic.guardar_en_historial("data_history.csv", archivo.name, len(df), num_ataques, t_fin, fecha_simulada, puerto_top)
            
            # Mostrar Resultados Rápidos
            c1, c2, c3 = st.columns(3)
            c1.metric("Ataques Detectados", num_ataques, delta_color="inverse")
            c2.metric("Puerto más Atacado", puerto_top)
            c3.metric("Tiempo", f"{round(t_fin, 2)}s")
            
            st.success(f"Análisis del {fecha_simulada} guardado en el historial.")

# --- PESTAÑA 2: TENDENCIAS (PARA EL CAP 4) ---
with tab2:
    st.header("📈 Análisis de Comportamiento Histórico")
    
    if os.path.exists("data_history.csv"):
        df_h = pd.read_csv("data_history.csv")
        df_h['Fecha'] = pd.to_datetime(df_h['Fecha'])
        df_h = df_h.sort_values("Fecha") # Ordenar por fecha para la línea
        
        # 1. Gráfica de Líneas con Puntos (Lo que pidió el Profe)
        fig_linea = px.line(df_h, x="Fecha", y="Ataques_Detectados", 
                            markers=True, 
                            title="Tendencia Diaria de Incidentes",
                            labels={"Ataques_Detectados": "Núm. de Ataques", "Fecha": "Día Simulado"})
        
        # Estilo de puntos
        fig_linea.update_traces(line_color='#ef4444', marker=dict(size=10, symbol='circle'))
        st.plotly_chart(fig_linea, use_container_width=True)
        
        # 2. Tabla de Datos para el Cap 4
        st.subheader("📋 Resumen de Auditoría")
        st.dataframe(df_h, use_container_width=True)
        
        # Botón para descargar reporte
        csv_report = df_h.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Reporte para Tesis", data=csv_report, file_name="reporte_abril_tesis.csv")
        
    else:
        st.warning("Aún no hay datos. Realiza un escaneo en la Pestaña 1.")
