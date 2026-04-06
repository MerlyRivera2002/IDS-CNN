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

# --- 1. SESIÓN Y LOGIN ---
if 'perfil' not in st.session_state: st.session_state.perfil = None
if 'ultimas_preds' not in st.session_state: st.session_state.ultimas_preds = None

st.sidebar.title("🔐 Acceso")
if st.session_state.perfil is None:
    u = st.sidebar.text_input("Usuario")
    p = st.sidebar.text_input("Clave", type="password")
    if st.sidebar.button("Entrar"):
        if u == "admin" and p == "tesis2026": st.session_state.perfil = "Administrador"; st.rerun()
        elif u == "viewer" and p == "consulta": st.session_state.perfil = "Usuario"; st.rerun()
    st.stop()
else:
    st.sidebar.success(f"Perfil: {st.session_state.perfil}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.clear(); st.rerun()

# --- 2. CARGA DE MODELO ---
@st.cache_resource
def load_assets():
    m = tf.keras.models.load_model("modelo_cnn.keras")
    s = joblib.load("scaler.pkl")
    f = joblib.load("features.pkl")
    return m, s, f

model, scaler, features_list = load_assets()

# --- 3. PESTAÑAS ---
tab1, tab2 = st.tabs(["🚀 Monitor & Análisis", "📊 Dashboard Histórico"])

# --- PESTAÑA 1: MONITOR, GRÁFICOS Y TABLA ---
with tab1:
    st.title("🛡️ Sistema de Detección e Informe")
    if st.session_state.perfil == "Administrador":
        archivo = st.file_uploader("Subir CSV para analizar", type=["csv"])
        if archivo:
            # CONFIGURACIÓN: Cambia aquí la cantidad de datos
            filas_a_analizar = 2000 
            
            if st.button("▶️ Iniciar Escaneo Real"):
                grafico_espacio = st.empty()
                with st.status("Procesando tráfico...", expanded=True) as status:
                    t_inicio = time.time()
                    df_raw = pd.read_csv(archivo, nrows=filas_a_analizar)
                    df_raw.columns = df_raw.columns.str.strip()
                    df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
                    X_scaled = scaler.transform(df_clean[features_list]).reshape(-1, len(features_list), 1)
                    
                    preds, normal, ataque = [], 0, 0
                    for i in range(0, len(X_scaled), 50):
                        res = (model.predict(X_scaled[i:i+50], verbose=0) > 0.5).astype(int).flatten()
                        for r in res:
                            preds.append(r)
                            if r == 1: ataque += 1
                            else: normal += 1
                        
                        with grafico_espacio.container():
                            c1, c2 = st.columns(2)
                            fig, ax = plt.subplots(figsize=(5, 3))
                            ax.pie([normal, ataque], labels=["Normal", "Ataque"], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
                            c1.pyplot(fig); plt.close()
                            c2.metric("Ataques Detectados", ataque)
                            c2.metric("Total Procesado", len(preds))
                    
                    t_fin = time.time()
                    status.update(label="✅ Análisis Completo", state="complete")

                # --- SECCIÓN DE MÉTRICAS (ABAJO DEL MONITOR) ---
                st.divider()
                st.subheader("📊 Métricas de Eficiencia y Puertos")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Clasificación por Puerto (Top Atacados)**")
                    top_p = logic.analizar_puertos(df_clean, preds)
                    if top_p is not None: st.bar_chart(top_p)
                    else: st.info("No hay ataques suficientes para mostrar puertos.")

                with col2:
                    t_total, t_reg = logic.calcular_eficiencia(t_inicio, t_fin, len(preds))
                    st.write(f"⏱️ **Tiempo Total:** {t_total:.2f} segundos")
                    st.write(f"⚡ **Velocidad:** {t_reg:.6f} seg/registro")
                    
                    if 'Label' in df_clean.columns:
                        st.write("**🎯 Matriz de Confusión**")
                        y_real = df_clean['Label'].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
                        cm, rep = logic.generar_metricas_detalladas(y_real, preds)
                        st.table(cm) # Muestra la matriz de forma simple

                # --- TABLA DE DATOS ANALIZADOS ---
                st.divider()
                st.subheader("📝 Tabla de Datos Analizados")
                df_final = df_clean.copy()
                df_final['Resultado_IA'] = ["ATAQUE" if p == 1 else "NORMAL" for p in preds]
                st.dataframe(df_final, use_container_width=True)

                # Guardar en historial
                logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataque, t_total)
                st.session_state.ultimas_preds = preds
    else:
        st.warning("Acceso solo para Administradores.")

# --- PESTAÑA 2: DASHBOARD HISTÓRICO ---
with tab2:
    st.title("📅 Reportes por Día")
    if os.path.exists("historial.csv"):
        df_hist = pd.read_csv("historial.csv")
        
        # Resaltado por color (Diferencia visual entre filas)
        def resaltar_ataques(val):
            color = '#ffcccc' if val > 0 else '#ccffcc'
            return f'background-color: {color}'
        
        st.write("### Registro de Actividad")
        st.dataframe(df_hist.style.applymap(resaltar_ataques, subset=['Ataques_Detectados']), use_container_width=True)
        
        st.write("### Comportamiento Semanal")
        st.bar_chart(df_hist.set_index("Fecha")["Ataques_Detectados"])
    else:
        st.info("Aún no hay datos guardados.")
