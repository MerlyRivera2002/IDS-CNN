import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import matplotlib.pyplot as plt
import logic 

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

with tab1:
    st.title("🛡️ Monitor de Intrusiones en Tiempo Real")
    
    if st.session_state.perfil == "Administrador":
        archivo = st.file_uploader("📂 Subir tráfico de red (CSV)", type=["csv"], key="uploader_id")
        
        if archivo:
            st.info(f"Archivo detectado: **{archivo.name}**")
            
            # --- CONFIGURACIÓN DE CANTIDAD DE DATOS ---
            # CAMBIA AQUÍ: Cambia el 2000 por el número de filas que quieras analizar
            filas_a_analizar = 2000 
            
            if st.button("🚀 Iniciar Simulación"):
                # Contenedores para la simulación visual
                metrica_ataque = st.empty()
                grafico_espacio = st.empty()
                
                with st.status("Simulando tráfico de red...", expanded=True) as status:
                    t_inicio = time.time()
                    
                    # Carga y Limpieza
                    df_raw = pd.read_csv(archivo, nrows=filas_a_analizar)
                    df_raw.columns = df_raw.columns.str.strip()
                    df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    # Transformación
                    X_scaled = scaler.transform(df_clean[features_list])
                    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                    
                    total = len(X_scaled)
                    preds = []
                    normal, ataque = 0, 0
                    
                    # SIMULACIÓN POR BLOQUES (Para que se vea real y rápido)
                    batch_size = 20 # Procesa de 20 en 20
                    for i in range(0, total, batch_size):
                        # Predicción del bloque
                        bloque = X_scaled[i : i + batch_size]
                        probabilidades = model.predict(bloque, verbose=0)
                        resultados = (probabilidades > 0.5).astype(int).flatten()
                        
                        # Actualizar contadores
                        for res in resultados:
                            preds.append(res)
                            if res == 0: normal += 1
                            else: ataque += 1
                        
                        # Actualización visual en tiempo real
                        with grafico_espacio.container():
                            c1, c2 = st.columns(2)
                            with c1:
                                fig, ax = plt.subplots(figsize=(4, 3))
                                ax.pie([normal, ataque], labels=["Normal", "Ataque"], 
                                       autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
                                st.pyplot(fig)
                                plt.close()
                            with c2:
                                st.metric("Registros Procesados", len(preds))
                                st.metric("Ataques Detectados", ataque)
                        
                        # Pequeña pausa para el efecto visual (puedes bajarla a 0.01 si quieres más veloz)
                        time.sleep(0.05) 

                    t_fin = time.time()
                    status.update(label="✅ Simulación Finalizada", state="complete", expanded=False)
                
                # --- GUARDADO Y RESULTADOS FINALES ---
                t_total, t_reg = logic.calcular_eficiencia(t_inicio, t_fin, total)
                logic.guardar_en_historial("historial.csv", archivo.name, total, ataque, t_total)
                
                st.success(f"Análisis completado: {total} registros en {t_total:.2f} segundos.")
                st.session_state.ultimas_preds = preds
                st.session_state.ultimo_df = df_clean
    else:
        st.warning("Acceso restringido: Solo el Administrador puede procesar nuevos archivos.")

# --- PESTAÑAS 2 Y 3 (Se mantienen igual) ---
with tab2:
    st.title("📅 Reporte Histórico de Actividad")
    if os.path.exists("historial.csv"):
        df_h = pd.read_csv("historial.csv")
        st.dataframe(df_h, use_container_width=True)
        st.write("### Tendencia de Ataques por Sesión")
        st.line_chart(df_h.set_index("Fecha")["Ataques_Detectados"])

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
