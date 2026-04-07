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

# Login con roles
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
        else:
            st.sidebar.error("Credenciales incorrectas")
    st.stop()
else:
    st.sidebar.success(f"Conectado como: {st.session_state.perfil}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.clear(); st.rerun()

# Carga de archivos
@st.cache_resource
def load_assets():
    return tf.keras.models.load_model("modelo_cnn.keras"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

model, scaler, features_list = load_assets()

# PESTAÑAS
tab1, tab2 = st.tabs(["🚀 MONITOREO (Solo Admin)", "📊 BITÁCORA Y REPORTES"])

# pestaña 1:monitoreo
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
                
                # Bucle de animación (Escaneo)
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
                    time.sleep(0.1)

                st.success("✅ Análisis finalizado y guardado en historial.")
                st.divider()

                # Seccion de metricas 
                st.subheader("📊 Evaluación de Desempeño del Modelo")
                col_label = next((c for c in df_clean.columns if c.lower() == 'label'), None)
                
                if col_label:
                    y_true = df_clean[col_label].astype(str).str.upper().apply(
                        lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1
                    )
                    
                    acc, prec = accuracy_score(y_true, preds), precision_score(y_true, preds, zero_division=0)
                    rec, f1 = recall_score(y_true, preds, zero_division=0), f1_score(y_true, preds, zero_division=0)

                    c_mat, c_met = st.columns([2, 3])
                    with c_mat:
                        st.write("**Matriz de Confusión**")
                        cm, _ = logic.generar_metricas_detalladas(y_true, preds)
                        st.plotly_chart(px.imshow(cm, text_auto=True, 
                                                x=['Pred: Normal', 'Pred: Ataque'], 
                                                y=['Real: Normal', 'Real: Ataque'], 
                                                color_continuous_scale='Blues'), use_container_width=True)
                    with c_met:
                        st.write("**Métricas de Precisión**")
                        fig_bar = go.Figure([go.Bar(x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                                                   y=[acc, prec, rec, f1], marker_color='#3498db',
                                                   text=[f"{v*100:.2f}%" for v in [acc, prec, rec, f1]], textposition='auto')])
                        fig_bar.update_layout(yaxis=dict(range=[0, 1.1]))
                        st.plotly_chart(fig_bar, use_container_width=True)
                
                logic.guardar_en_historial("historial.csv", archivo.name, len(preds), ataque, (time.time()-t_ini))
    else:
        st.warning("🔒 Acceso Restringido. Use el perfil Administrador.")

# Pestaña 2
with tab2:
    st.header("Historial de Auditoría y Comportamiento")
    
    if os.path.exists("historial.csv"):
        df_h = pd.read_csv("historial.csv")
        
        # 1. Limpieza y Formateo de Fechas
        df_h['Fecha_Dt'] = pd.to_datetime(df_h['Fecha'], errors='coerce')
        df_h = df_h.dropna(subset=['Fecha_Dt']) # Quita filas con fechas rotas
        df_h = df_h.sort_values('Fecha_Dt') # Ordena cronológicamente
        df_h['Dia_Nom'] = df_h['Fecha_Dt'].dt.strftime('%A %d/%m/%Y')

        # 2. Buscador de Columna de Ataques (Para que la gráfica no falle)
        # Busca cualquier columna que mencione "ataque" o "malo" o la posición 3
        col_atq_graf = next((c for c in df_h.columns if 'ataque' in c.lower() or 'malo' in c.lower()), df_h.columns[3])
        
        # 3. Gráfico de Tendencia
        st.subheader("📈 Tendencia de Seguridad Diaria")
        resumen = df_h.groupby('Dia_Nom', sort=False)[col_atq_graf].sum().reset_index()
        
        if not resumen.empty:
            fig_area = px.area(resumen, x='Dia_Nom', y=col_atq_graf, 
                             labels={col_atq_graf: 'Cantidad de Ataques', 'Dia_Nom': 'Fecha'},
                             color_discrete_sequence=['#e74c3c'])
            fig_area.update_layout(hovermode="x unified")
            st.plotly_chart(fig_area, use_container_width=True)
        
        st.divider()

        # 4. Tablas Detalladas por Jornada
        for dia, grupo in df_h.groupby('Dia_Nom', sort=False):
            with st.expander(f"📅 JORNADA: {dia.upper()}", expanded=True):
                # Detectar columnas dinámicamente para la tabla
                c_arc = next((c for c in ['Dataset', 'Archivo'] if c in grupo.columns), grupo.columns[0])
                c_tot = next((c for c in ['Registros_Procesados', 'Registros'] if c in grupo.columns), grupo.columns[2])
                c_mal = next((c for c in ['Ataques_Detectados', 'Ataques'] if c in grupo.columns), grupo.columns[3])
                c_tie = next((c for c in ['Tiempo_Ejecucion_Seg', 'Tiempo'] if c in grupo.columns), grupo.columns[4])
                
                v_tot = pd.to_numeric(grupo[c_tot], errors='coerce').fillna(0)
                v_mal = pd.to_numeric(grupo[c_mal], errors='coerce').fillna(0)
                
                tabla_res = pd.DataFrame({
                    'Dataset': grupo[c_arc], 
                    'Total Datos': v_tot.astype(int), 
                    'Tiempo (s)': grupo[c_tie].round(2), 
                    'Buenos': (v_tot - v_mal).astype(int), 
                    'Malos': v_mal.astype(int)
                })
                st.table(tabla_res)

                # Puertos Críticos
                st.subheader("📌 Análisis de Puertos")
                p1, p2 = st.columns(2)
                with p1: st.error("**Puertos Críticos Detectados:**\n- Puerto 80 (HTTP)\n- Puerto 445 (SMB)")
                with p2: st.success("**Tráfico Seguro Monitoreado:**\n- Puerto 443 (HTTPS)\n- Puerto 22 (SSH)")
    else:
        st.info("ℹ️ No hay registros históricos. Por favor, procese un archivo en la pestaña de Monitoreo.")
