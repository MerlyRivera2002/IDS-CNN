import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import matplotlib.pyplot as plt
import logic  # Importamos tu archivo de lógica

# Configuración de la página
st.set_page_config(page_title="IDS Tesis 2026", layout="wide", page_icon="🛡️")

# --- 1. SISTEMA DE PERFILES Y SESIÓN ---
if 'perfil' not in st.session_state:
    st.session_state.perfil = None
if 'ultimas_preds' not in st.session_state:
    st.session_state.ultimas_preds = None

st.sidebar.title("🔐 Acceso al Sistema")

if st.session_state.perfil is None:
    user = st.sidebar.text_input("Usuario")
    pw = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Ingresar"):
        if user == "admin" and pw == "tesis2026":
            st.session_state.perfil = "Administrador"
            st.rerun()
        elif user == "viewer" and pw == "consulta":
            st.session_state.perfil = "Usuario"
            st.rerun()
        else:
            st.sidebar.error("Credenciales incorrectas")
    st.stop()
else:
    st.sidebar.success(f"Conectado como: {st.session_state.perfil}")
    if st.sidebar.button("Cerrar Sesión"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- 2. CARGA DE ASSETS ---
@st.cache_resource
def load_assets():
    try:
        m = tf.keras.models.load_model("modelo_cnn.keras")
        s = joblib.load("scaler.pkl")
        f = joblib.load("features.pkl")
        return m, s, f
    except Exception as e:
        st.error(f"Error cargando archivos base: {e}")
        st.stop()

model, scaler, features_list = load_assets()

# --- 3. DISEÑO DE PESTAÑAS ---
tab1, tab2, tab3 = st.tabs(["🚀 Monitor en Vivo", "📊 Historial Diario", "🎯 Evaluación de Desempeño"])

# --- PESTAÑA 1: MONITOR ---
with tab1:
    st.title("🛡️ Monitor de Intrusiones en Tiempo Real")
    
    if st.session_state.perfil == "Administrador":
        archivo = st.file_uploader("📂 Subir tráfico de red (CSV)", type=["csv"], key="uploader_id")
        
        if archivo:
            st.info(f"Archivo detectado: **{archivo.name}**")
            
            # Botón de acción
            if st.button("🚀 Iniciar Análisis de Seguridad"):
                with st.status("Ejecutando proceso...", expanded=True) as status:
                    t_inicio = time.time()
                    
                    # Carga y Limpieza
                    status.write("Leyendo datos del tráfico...")
                    df_raw = pd.read_csv(archivo, nrows=5000)
                    df_raw.columns = df_raw.columns.str.strip()
                    df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    # Transformación
                    status.write("Preparando datos para la CNN...")
                    X_scaled = scaler.transform(df_clean[features_list])
                    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                    
                    # Predicción con barra de progreso
                    status.write("Clasificando tráfico con Inteligencia Artificial...")
                    bar = st.progress(0)
                    preds = []
                    normal, ataque = 0, 0
                    total = len(X_scaled)
                    
                    for i in range(total):
                        prob = model.predict(X_scaled[i:i+1], verbose=0)[0][0]
                        res = 1 if prob > 0.5 else 0
                        preds.append(res)
                        if res == 0: normal += 1
                        else: ataque += 1
                        
                        if i % 100 == 0: # Actualiza la barra cada 100 registros
                            bar.progress((i + 1) / total)
                    
                    t_fin = time.time()
                    status.update(label="✅ Análisis Finalizado", state="complete", expanded=False)
                
                # --- RESULTADOS ---
                t_total, t_reg = logic.calcular_eficiencia(t_inicio, t_fin, total)
                logic.guardar_en_historial("historial.csv", archivo.name, total, ataque, t_total)
                
                st.divider()
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Analizados", total)
                col_m2.metric("Normal", normal, delta_color="normal")
                col_m3.metric("Ataques", ataque, delta="- Detected", delta_color="inverse")
                
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots()
                    ax.pie([normal, ataque], labels=["Normal", "Ataque"], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
                    st.pyplot(fig)
                
                with c2:
                    st.success(f"Velocidad: {t_total} segundos totales.")
                    st.info(f"Eficiencia: {t_reg} seg/registro.")

                # Guardar en sesión para otras pestañas
                st.session_state.ultimas_preds = preds
                st.session_state.ultimo_df = df_clean
    else:
        st.warning("Acceso restringido: Solo el Administrador puede procesar nuevos archivos.")

# --- PESTAÑA 2: HISTORIAL ---
with tab2:
    st.title("📅 Reporte Histórico de Actividad")
    if os.path.exists("historial.csv"):
        df_h = pd.read_csv("historial.csv")
        st.dataframe(df_h, use_container_width=True)
        
        st.write("### Tendencia de Ataques por Sesión")
        st.line_chart(df_h.set_index("Fecha")["Ataques_Detectados"])
    else:
        st.info("No hay registros previos. Realiza un escaneo para empezar el historial.")

# --- PESTAÑA 3: EVALUACIÓN (CAPÍTULO 4) ---
with tab3:
    st.title("🎯 Evaluación de Desempeño y Métricas")
    if st.session_state.ultimas_preds is not None:
        df_ev = st.session_state.ultimo_df
        
        if 'Label' in df_ev.columns:
            y_real = df_ev['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
            cm, reporte = logic.generar_metricas_detalladas(y_real, st.session_state.ultimas_preds)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("### Matriz de Confusión")
                st.write(cm)
            with col_b:
                st.write("### Reporte de Precisión")
                st.json(reporte)
        else:
            st.error("⚠️ El archivo cargado no contiene la columna 'Label' (Etiqueta Real). No se puede generar la Matriz de Confusión.")
    else:
        st.info("💡 Primero realiza un escaneo en la pestaña 'Monitor' para ver los resultados estadísticos aquí.")
