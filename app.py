import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import matplotlib.pyplot as plt
import plotly.express as px # <--- ESTO HARÁ QUE SE VEA DIFERENTE
import logic 

# Configuración de estilo
st.set_page_config(page_title="IDS PRO 2026", layout="wide", page_icon="🛡️")

# --- 1. LOGIN (Simplificado para que no estorbe) ---
if 'perfil' not in st.session_state: st.session_state.perfil = None
if 'datos_analizados' not in st.session_state: st.session_state.datos_analizados = None

if st.session_state.perfil is None:
    st.title("🔐 Acceso al IDS")
    u = st.text_input("Usuario")
    p = st.text_input("Clave", type="password")
    if st.button("Ingresar"):
        if u == "admin" and p == "tesis2026": 
            st.session_state.perfil = "Administrador"
            st.rerun()
    st.stop()

# --- 2. CARGA DE MODELO ---
@st.cache_resource
def load_assets():
    m = tf.keras.models.load_model("modelo_cnn.keras")
    s = joblib.load("scaler.pkl")
    f = joblib.load("features.pkl")
    return m, s, f

model, scaler, features_list = load_assets()

# --- 3. NUEVA ESTRUCTURA DE PESTAÑAS ---
tab1, tab2 = st.tabs(["🚀 MONITOR & MÉTRICAS", "📅 BITÁCORA HISTÓRICA"])

with tab1:
    st.header("🛡️ Análisis de Tráfico en Tiempo Real")
    archivo = st.file_uploader("Cargar Dataset (CSV)", type=["csv"])
    
    if archivo:
        # AQUÍ CAMBIAS LA CANTIDAD DE DATOS (2000 por ahora)
        limite = 2000 
        
        if st.button("▶️ INICIAR ESCANEO"):
            # Contenedor para el monitor animado
            espacio_monitor = st.empty()
            
            with st.status("Procesando...", expanded=True) as status:
                t_ini = time.time()
                df = pd.read_csv(archivo, nrows=limite)
                df.columns = df.columns.str.strip()
                df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
                X = scaler.transform(df_clean[features_list]).reshape(-1, len(features_list), 1)
                
                # Predicción rápida en bloques
                probabilidades = model.predict(X, verbose=0)
                preds = (probabilidades > 0.5).astype(int).flatten()
                
                t_fin = time.time()
                status.update(label="✅ Análisis Completo", state="complete")

            # --- RESULTADOS VISUALES (ESTO ES LO QUE CAMBIA) ---
            st.divider()
            st.subheader("📊 Métricas del Análisis Actual")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            ataques = int(np.sum(preds))
            normales = len(preds) - ataques
            
            with col1:
                # GRÁFICO DE DONA (Más moderno que el de pastel)
                fig_dona = px.pie(names=["Normal", "Ataque"], values=[normales, ataques], 
                                 hole=0.5, color_discrete_sequence=["#00CC96", "#EF553B"])
                st.plotly_chart(fig_dona, use_container_width=True)
            
            with col2:
                # GRÁFICO DE PUERTOS (Barras horizontales)
                st.write("**Puertos más afectados**")
                top_p = logic.analizar_puertos(df_clean, preds)
                if top_p is not None:
                    st.bar_chart(top_p)
            
            with col3:
                # MATRIZ DE CONFUSIÓN (Mapa de calor)
                if 'Label' in df_clean.columns:
                    st.write("**Matriz de Confusión**")
                    y_real = df_clean['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
                    cm, _ = logic.generar_metricas_detalladas(y_real, preds)
                    fig_hm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                      x=['Normal', 'Ataque'], y=['Real N', 'Real A'])
                    st.plotly_chart(fig_hm, use_container_width=True)

            # TABLA COMPLETA AL FINAL
            st.write("---")
            st.subheader("🔍 Detalle de los Datos Analizados")
            df_clean['Resultado_IA'] = ["⚠️ ATAQUE" if p == 1 else "✅ NORMAL" for p in preds]
            st.dataframe(df_clean, use_container_width=True)
            
            # Guardar para el historial
            logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataques, (t_fin-t_ini))

with tab2:
    st.header("📅 Historial de Escaneos")
    if os.path.exists("historial.csv"):
        h = pd.read_csv("historial.csv")
        h['Fecha'] = pd.to_datetime(h['Fecha'])
        
        # SEPARACIÓN POR DÍAS CON COLORES
        dias = h['Fecha'].dt.strftime('%A %d de %B').unique()
        
        for dia in dias:
            st.markdown(f"### 🗓️ {dia}")
            # Filtramos los datos de ese día específico
            datos_dia = h[h['Fecha'].dt.strftime('%A %d de %B') == dia]
            
            # Aplicamos colores a la tabla
            def color_ataque(val):
                color = 'background-color: #ffb3b3' if val > 0 else 'background-color: #b3ffb3'
                return color
            
            st.dataframe(datos_dia.style.applymap(color_ataque, subset=['Ataques_Detectados']), use_container_width=True)
            st.write("") 
    else:
        st.info("No hay registros previos.")
