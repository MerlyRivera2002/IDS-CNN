import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import logic

st.set_page_config(page_title="IDS Tesis 2026", layout="wide", page_icon="🛡️")

# --- LOGIN (igual que antes) ---
if 'perfil' not in st.session_state:
    st.session_state.perfil = None

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
    st.sidebar.divider()
    st.sidebar.subheader("📅 Simulación de Tiempo")
    fecha_simulada = st.sidebar.date_input("Fecha del Escaneo", value=pd.to_datetime("2026-04-01"))
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.clear()
        st.rerun()

# Carga de activos
@st.cache_resource
def load_assets():
    return (tf.keras.models.load_model("modelo_cnn.keras"),
            joblib.load("scaler.pkl"),
            joblib.load("features.pkl"))

model, scaler, features_list = load_assets()

# Crear las 3 pestañas
tab1, tab2, tab3 = st.tabs(["🚀 SIMULACIÓN EN VIVO", "📈 ANÁLISIS Y TENDENCIAS", "📋 REPORTES Y DESCARGA"])

# =====================================================================
# PESTAÑA 1: SIMULACIÓN (mejorada con barra de progreso y gráficos)
# =====================================================================
with tab1:
    if st.session_state.perfil == "Administrador":
        st.header("🛡️ Monitor de Tráfico en Tiempo Real")
        
        # Controles
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
        with col_ctrl1:
            archivo = st.file_uploader("Subir dataset CSV (CIC-IDS2017 o similar)", type=["csv"], key="uploader_sim")
        with col_ctrl2:
            velocidad = st.slider("Velocidad (segundos/lote)", min_value=0.02, max_value=0.5, value=0.08, step=0.02)
        with col_ctrl3:
            if st.button("🔄 Limpiar resultados", use_container_width=True):
                if 'simulacion_activa' in st.session_state:
                    del st.session_state.simulacion_activa
                st.rerun()
        
        # Contenedores dinámicos (siempre presentes)
        col_izq, col_der = st.columns([1, 1])
        with col_izq:
            espacio_pastel = st.empty()
            espacio_evolucion = st.empty()
        with col_der:
            espacio_metricas = st.empty()
            espacio_puertos = st.empty()
        
        st.divider()
        st.subheader("🛰️ Registro de Actividad (último lote)")
        espacio_tabla = st.empty()
        
        # Estado inicial
        if 'simulacion_activa' not in st.session_state:
            st.session_state.simulacion_activa = False
            espacio_pastel.info("👈 Esperando inicio de simulación")
            espacio_evolucion.info("📈 Gráfico de evolución")
            espacio_metricas.info("📊 Métricas en tiempo real")
            espacio_puertos.info("🔍 Top puertos atacados")
            espacio_tabla.info("📋 Los registros aparecerán aquí")
        
        if archivo:
            if st.button("🚀 INICIAR MONITOREO", type="primary", use_container_width=True):
                with st.spinner("Preparando datos..."):
                    df_raw = pd.read_csv(archivo)
                    df_raw.columns = df_raw.columns.str.strip()
                    df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
                    total_registros = len(df_clean)
                    
                    preds_totales = []
                    tiempos = []
                    ataques_acumulados = []
                    t_inicio = time.time()
                    
                    progress_bar = st.progress(0, text="Simulación en curso...")
                    
                    for idx, inicio in enumerate(range(0, total_registros, 15)):
                        chunk = df_clean.iloc[inicio: inicio+15]
                        X_chunk = scaler.transform(chunk[features_list]).reshape(-1, len(features_list), 1)
                        chunk_preds = (model.predict(X_chunk, verbose=0) > 0.5).astype(int).flatten()
                        preds_totales.extend(chunk_preds)
                        
                        ataques = sum(preds_totales)
                        normales = len(preds_totales) - ataques
                        ataques_acumulados.append(ataques)
                        tiempos.append(len(preds_totales))
                        
                        progress = (inicio + len(chunk)) / total_registros
                        progress_bar.progress(min(progress, 1.0), text=f"Procesando {int(progress*100)}%")
                        
                        # Pastel
                        fig_pie = px.pie(values=[normales, ataques], names=['Seguro', 'Amenaza'],
                                         color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.6,
                                         title=f"Tráfico actual (Total: {len(preds_totales)})")
                        fig_pie.update_layout(height=280)
                        espacio_pastel.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{idx}")
                        
                        # Evolución
                        df_evol = pd.DataFrame({'Registros': tiempos, 'Ataques': ataques_acumulados})
                        fig_evol = px.line(df_evol, x='Registros', y='Ataques', title='Evolución de intrusiones',
                                           labels={'Ataques': 'Ataques detectados'})
                        fig_evol.update_traces(line_color='#e74c3c', fill='tozeroy')
                        espacio_evolucion.plotly_chart(fig_evol, use_container_width=True, key=f"evol_{idx}")
                        
                        # Métricas dinámicas
                        with espacio_metricas.container():
                            st.metric("📊 Conexiones totales", f"{len(preds_totales)}")
                            st.metric("🚨 Intrusiones detectadas", f"{ataques}", delta=f"+{chunk_preds.sum()}", delta_color="inverse")
                            elapsed = time.time() - t_inicio
                            if elapsed > 0:
                                st.metric("⚡ Registros/seg", f"{len(preds_totales)/elapsed:.1f}")
                            # Si hay etiqueta, mostrar precisión acumulada
                            col_label = next((c for c in df_clean.columns if c.lower() == 'label'), None)
                            if col_label:
                                y_true_parcial = df_clean[col_label].iloc[:len(preds_totales)].astype(str).str.upper().apply(
                                    lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1)
                                acc_parcial = accuracy_score(y_true_parcial, preds_totales) if len(preds_totales) > 0 else 0
                                st.metric("🎯 Precisión acumulada", f"{acc_parcial:.2%}")
                        
                        # Top puertos
                        with espacio_puertos.container():
                            st.write("**🔍 Top 5 puertos objetivo (último lote)**")
                            puertos_chunk = chunk['Destination Port'].value_counts().head(5).reset_index()
                            puertos_chunk.columns = ['Puerto', 'Frecuencia']
                            fig_puertos = px.bar(puertos_chunk, x='Puerto', y='Frecuencia', color='Frecuencia',
                                                 color_continuous_scale='Reds')
                            fig_puertos.update_layout(height=200)
                            st.plotly_chart(fig_puertos, use_container_width=True, key=f"puertos_{idx}")
                        
                        # Tabla de diagnóstico
                        with espacio_tabla.container():
                            vista = chunk.copy()
                            vista['Estado'] = ["🚨 ATAQUE" if p == 1 else "✅ NORMAL" for p in chunk_preds]
                            def sugerir_amenaza(row):
                                if row['Estado'] == "✅ NORMAL":
                                    return "Tráfico seguro"
                                p = row['Destination Port']
                                if p in [80,443]: return "Ataque web (HTTP/S)"
                                if p == 22: return "Fuerza bruta SSH"
                                if p == 21: return "Acceso FTP no autorizado"
                                return "Escaneo / puerto sospechoso"
                            vista['Diagnóstico'] = vista.apply(sugerir_amenaza, axis=1)
                            st.table(vista[['Destination Port', 'Estado', 'Diagnóstico']])
                        
                        time.sleep(velocidad)
                    
                    progress_bar.empty()
                    st.success("✅ Simulación finalizada.")
                    st.balloons()
                    
                    # Resumen ejecutivo
                    st.subheader("📊 Resumen ejecutivo")
                    elapsed_total = time.time() - t_inicio
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                    with col_r1: st.metric("Total registros", f"{len(preds_totales)}")
                    with col_r2: st.metric("Ataques detectados", f"{ataques}", delta=f"{ataques/len(preds_totales):.1%}")
                    with col_r3: st.metric("Tiempo simulación", f"{elapsed_total:.2f} s")
                    with col_r4: st.metric("Eficiencia", f"{len(preds_totales)/elapsed_total:.1f} reg/s")
                    
                    # Evaluación final con etiquetas
                    col_label = next((c for c in df_clean.columns if c.lower() == 'label'), None)
                    if col_label:
                        y_true = df_clean[col_label].astype(str).str.upper().apply(
                            lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1)[:len(preds_totales)]
                        acc = accuracy_score(y_true, preds_totales)
                        prec = precision_score(y_true, preds_totales, zero_division=0)
                        rec = recall_score(y_true, preds_totales, zero_division=0)
                        f1 = f1_score(y_true, preds_totales, zero_division=0)
                        
                        st.write("**Métricas de rendimiento del modelo**")
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1: st.metric("Accuracy", f"{acc:.2%}")
                        with col_m2: st.metric("Precision", f"{prec:.2%}")
                        with col_m3: st.metric("Recall", f"{rec:.2%}")
                        with col_m4: st.metric("F1-Score", f"{f1:.2%}")
                        
                        # Guardar en historial
                        p_top = df_clean.iloc[:len(preds_totales)]['Destination Port'].mode()[0]
                        logic.guardar_en_historial(
                            "historial.csv", archivo.name, len(preds_totales), ataques,
                            elapsed_total, fecha_simulada, p_top, acc,
                            precision=prec, recall=rec, f1=f1
                        )
                    else:
                        p_top = df_clean.iloc[:len(preds_totales)]['Destination Port'].mode()[0]
                        logic.guardar_en_historial(
                            "historial.csv", archivo.name, len(preds_totales), ataques,
                            elapsed_total, fecha_simulada, p_top, 0.0
                        )
                    st.toast("Simulación registrada en Bitácora")
        else:
            # No hay archivo, ya se muestran los mensajes iniciales
            pass
    else:
        st.warning("🔒 Esta pestaña solo es accesible para Administradores.")

# =====================================================================
# PESTAÑA 2: ANÁLISIS Y TENDENCIAS
# =====================================================================
with tab2:
    st.header("📈 Análisis histórico y tendencias")
    df_h = logic.obtener_metricas_resumen("historial.csv")
    
    if df_h is not None and not df_h.empty:
        # KPIs
        st.subheader("📌 Resumen global")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total simulaciones", len(df_h))
        with col2: st.metric("Total ataques detectados", f"{df_h['Ataques'].sum():,}")
        with col3:
            avg_acc = df_h['Accuracy'].mean() if 'Accuracy' in df_h else 0
            st.metric("Precisión promedio", f"{avg_acc:.2%}" if pd.notna(avg_acc) else "N/A")
        with col4:
            puerto_top = df_h.loc[df_h['Ataques'].idxmax(), 'Puerto'] if not df_h.empty else "N/A"
            st.metric("Puerto más atacado", puerto_top)
        
        st.divider()
        
        # Gráficas de tendencia
        st.subheader("📈 Evolución temporal")
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.line(df_h, x='Fecha', y='Ataques', markers=True,
                           title="Evolución de intrusiones detectadas",
                           labels={'Ataques': 'Número de ataques'})
            fig1.update_traces(line_color='#e74c3c')
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            # Top 5 puertos históricos
            puertos_hist = df_h['Puerto'].value_counts().head(5).reset_index()
            puertos_hist.columns = ['Puerto', 'Frecuencia']
            fig_puertos = px.bar(puertos_hist, x='Puerto', y='Frecuencia',
                                 title="Top 5 puertos más frecuentes",
                                 color='Frecuencia', color_continuous_scale='Reds')
            st.plotly_chart(fig_puertos, use_container_width=True)
        
        st.divider()
        
        # Tabla completa con todas las métricas
        st.subheader("📋 Registro detallado de todas las simulaciones")
        # Seleccionar y formatear columnas
        columnas = ['Fecha', 'Hora', 'Dataset', 'Total', 'Normales', 'Ataques',
                    'Accuracy', 'Precision', 'Recall', 'F1', 'Puerto', 'Tiempo (s)']
        for col in columnas:
            if col not in df_h.columns:
                df_h[col] = np.nan
        df_display = df_h.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
        st.dataframe(df_display[columnas], use_container_width=True, height=400)
    else:
        st.info("💡 No hay datos históricos. Realiza una simulación en la Pestaña 1.")

# =====================================================================
# PESTAÑA 3: REPORTES Y DESCARGA
# =====================================================================
with tab3:
    st.header("📋 Reportes personalizados y exportación")
    df_h = logic.obtener_metricas_resumen("historial.csv")
    
    if df_h is not None and not df_h.empty:
        st.subheader("📅 Filtrar por rango de fechas")
        min_fecha = df_h['Fecha'].min().date()
        max_fecha = df_h['Fecha'].max().date()
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fecha_ini = st.date_input("Desde", value=min_fecha, min_value=min_fecha, max_value=max_fecha, key="rep_ini")
        with col_f2:
            fecha_fin = st.date_input("Hasta", value=max_fecha, min_value=min_fecha, max_value=max_fecha, key="rep_fin")
        
        mask = (df_h['Fecha'].dt.date >= fecha_ini) & (df_h['Fecha'].dt.date <= fecha_fin)
        df_filtrado = df_h.loc[mask].copy()
        
        if df_filtrado.empty:
            st.warning("No hay registros en ese rango.")
        else:
            st.success(f"✅ {len(df_filtrado)} registros encontrados.")
            
            # Mostrar tabla filtrada
            st.subheader("📋 Datos del reporte")
            columnas = ['Fecha', 'Hora', 'Dataset', 'Total', 'Ataques', 'Accuracy', 'Precision', 'Recall', 'F1', 'Puerto']
            df_show = df_filtrado[columnas].copy()
            for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
                if col in df_show.columns:
                    df_show[col] = df_show[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
            st.dataframe(df_show, use_container_width=True)
            
            # Gráfico resumen del reporte
            st.subheader("📊 Resumen gráfico del reporte")
            if not df_filtrado.empty:
                fig_report = px.bar(df_filtrado, x='Fecha', y='Ataques', title='Ataques por fecha en el periodo seleccionado',
                                    color='Ataques', color_continuous_scale='Reds')
                st.plotly_chart(fig_report, use_container_width=True)
            
            # Botón de descarga
            csv_buffer = io.StringIO()
            df_filtrado.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📎 Descargar reporte (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"reporte_{fecha_ini}_{fecha_fin}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.divider()
        st.subheader("📥 Exportar historial completo")
        csv_total = io.StringIO()
        df_h.to_csv(csv_total, index=False)
        st.download_button(
            label="📎 Descargar todo el historial",
            data=csv_total.getvalue(),
            file_name="historial_completo.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Opción para borrar historial
        if st.button("🗑️ Borrar todo el historial", type="secondary"):
            if os.path.exists("historial.csv"):
                os.remove("historial.csv")
                st.success("Historial eliminado. Recarga la página.")
                st.rerun()
    else:
        st.info("💡 No hay datos históricos. Realiza una simulación primero.")
