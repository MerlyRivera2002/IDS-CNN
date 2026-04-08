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

# --- LOGIN ---
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

tab1, tab2 = st.tabs(["🚀 MONITOREO (Solo Admin)", "📊 BITÁCORA Y REPORTES"])

# ----------------------------------------- PESTAÑA 1 -----------------------------------------------------------
with tab1:
    if st.session_state.perfil == "Administrador":
        st.header("🛡️ Monitor de Tráfico en Tiempo Real")
        archivo = st.file_uploader("Subir dataset para simulación", type=["csv"], key="uploader_sim")

        if archivo:
            if st.button("🚀 INICIAR MONITOREO"):
                # Contenedores para la simulación en vivo
                col_izq, col_der = st.columns([1, 1])
                with col_izq:
                    espacio_pastel = st.empty()
                with col_der:
                    espacio_metricas = st.empty()

                st.divider()
                st.subheader("🛰️ Registro de Actividad")
                espacio_tabla = st.empty()

                # Procesamiento de datos
                df_raw = pd.read_csv(archivo)
                df_raw.columns = df_raw.columns.str.strip()
                df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()

                preds_totales = []
                t_inicio = time.time()

                # Bucle de simulación
                for i in range(0, len(df_clean), 15):
                    chunk = df_clean.iloc[i: i + 15]
                    X_chunk = scaler.transform(chunk[features_list]).reshape(-1, len(features_list), 1)
                    chunk_preds = (model.predict(X_chunk, verbose=0) > 0.5).astype(int).flatten()
                    preds_totales.extend(chunk_preds)

                    ataques = sum(preds_totales)
                    normales = len(preds_totales) - ataques

                    # Gráfico de pastel
                    fig_pie = px.pie(values=[normales, ataques],
                                     names=['Seguro', 'Amenaza'],
                                     color_discrete_sequence=['#2ecc71', '#e74c3c'],
                                     hole=0.6)
                    fig_pie.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
                    espacio_pastel.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{i}")

                    # Métricas
                    with espacio_metricas.container():
                        st.metric("CONEXIONES TOTALES", f"{len(preds_totales)}")
                        st.metric("INTRUSIONES DETECTADAS", f"{ataques}",
                                  delta=f"+{chunk_preds.sum()}", delta_color="inverse")

                    # Tabla de diagnóstico
                    with espacio_tabla.container():
                        vista = chunk.copy()
                        vista['Estado'] = ["🚨 ATAQUE" if p == 1 else "✅ NORMAL" for p in chunk_preds]

                        def sugerir_amenaza(row):
                            if "NORMAL" in row['Estado']:
                                return "Tráfico Seguro"
                            p = row['Destination Port']
                            if p in [80, 443]:
                                return "Ataque Web (HTTP/S)"
                            if p == 22:
                                return "Fuerza Bruta (SSH)"
                            if p == 21:
                                return "Acceso FTP"
                            return "Escaneo / Port Scan"

                        vista['Diagnóstico'] = vista.apply(sugerir_amenaza, axis=1)
                        st.table(vista[['Destination Port', 'Estado', 'Diagnóstico']])

                    time.sleep(0.08)

                st.success("✅ Simulación finalizada.")
                st.divider()

                # --- Evaluación final ---
                st.subheader("📊 Evaluación del Rendimiento (Final)")
                col_label = next((c for c in df_clean.columns if c.lower() == 'label'), None)

                acc_historial = 0.0
                prec, rec, f1 = None, None, None

                if col_label:
                    y_true = df_clean[col_label].astype(str).str.upper().apply(
                        lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1
                    )
                    y_true = y_true[:len(preds_totales)]

                    acc_historial = accuracy_score(y_true, preds_totales)
                    prec = precision_score(y_true, preds_totales, zero_division=0)
                    rec = recall_score(y_true, preds_totales, zero_division=0)
                    f1 = f1_score(y_true, preds_totales, zero_division=0)

                    c_mat, c_line = st.columns([1, 1])
                    with c_mat:
                        st.write("**Matriz de Confusión**")
                        cm = confusion_matrix(y_true, preds_totales)
                        fig_cm = px.imshow(cm, text_auto=True,
                                           x=['Pred: Norm', 'Pred: Atq'],
                                           y=['Real: Norm', 'Real: Atq'],
                                           color_continuous_scale='Reds')
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with c_line:
                        st.write("**Gráfico de Rendimiento (Scores)**")
                        df_m = pd.DataFrame({
                            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                            'Valor': [acc_historial, prec, rec, f1]
                        })
                        fig_m = px.line(df_m, x='Métrica', y='Valor', markers=True,
                                        text=df_m['Valor'].apply(lambda x: f"{x:.2f}"))
                        fig_m.update_traces(line_color='#1f77b4',
                                            marker=dict(size=12, symbol='square', color='#ff7f0e'))
                        fig_m.update_layout(yaxis=dict(range=[0, 1.1]))
                        st.plotly_chart(fig_m, use_container_width=True)

                # --- Guardar en historial ---
                p_top = df_clean.iloc[:len(preds_totales)]['Destination Port'].mode()[0]
                if col_label:
                    logic.guardar_en_historial(
                        "historial.csv",
                        archivo.name,
                        len(preds_totales),
                        ataques,
                        (time.time() - t_inicio),
                        fecha_simulada,
                        p_top,
                        acc_historial,
                        precision=prec,
                        recall=rec,
                        f1=f1
                    )
                else:
                    logic.guardar_en_historial(
                        "historial.csv",
                        archivo.name,
                        len(preds_totales),
                        ataques,
                        (time.time() - t_inicio),
                        fecha_simulada,
                        p_top,
                        0.0
                    )
                st.toast("Simulación registrada en Bitácora")
    else:
        st.warning("🔒 Esta pestaña solo es accesible para Administradores.")

# ----------------------------------------- PESTAÑA 2 -----------------------------------------------------------
with tab2:
    st.header("📊 Inteligencia de Red y Toma de Decisiones")

    df_h = logic.obtener_metricas_resumen("historial.csv")

    if df_h is not None and not df_h.empty:
        # --- KPIs ---
        st.subheader("📌 Resumen Global")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Simulaciones", len(df_h))
        with col2:
            st.metric("Total Ataques Detectados", f"{df_h['Ataques'].sum():,}")
        with col3:
            avg_acc = df_h['Accuracy'].mean() if 'Accuracy' in df_h else 0
            st.metric("Precisión Promedio", f"{avg_acc:.2%}" if pd.notna(avg_acc) else "N/A")
        with col4:
            puerto_mas_atacado = df_h.loc[df_h['Ataques'].idxmax(), 'Puerto'] if not df_h.empty else "N/A"
            st.metric("Puerto Más Atacado", puerto_mas_atacado)

        st.divider()

        # --- Gráficas de tendencia ---
        st.subheader("📈 Evolución Temporal")
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.line(df_h, x='Fecha', y='Ataques', markers=True,
                           title="Evolución de Intrusiones Detectadas",
                           labels={'Ataques': 'Número de Ataques', 'Fecha': 'Fecha'})
            fig1.update_traces(line_color='#e74c3c', marker=dict(size=8))
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            puertos_counts = df_h['Puerto'].value_counts().reset_index()
            puertos_counts.columns = ['Puerto', 'Frecuencia']
            fig_puertos = px.bar(puertos_counts.head(5), x='Puerto', y='Frecuencia',
                                 title="Top 5 Puertos Más Frecuentes",
                                 color='Frecuencia', color_continuous_scale='Reds')
            st.plotly_chart(fig_puertos, use_container_width=True)

        st.divider()

        # --- Tabla detallada con todas las métricas ---
        st.subheader("📋 Registro Detallado de Simulaciones")
        columnas_mostrar = ['Fecha', 'Dataset', 'Total', 'Normales', 'Ataques',
                            'Accuracy', 'Precision', 'Recall', 'F1', 'Puerto', 'Tiempo (s)']
        # Aseguramos que existan las columnas
        for col in columnas_mostrar:
            if col not in df_h.columns:
                df_h[col] = np.nan
        df_display = df_h.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
        st.dataframe(df_display[columnas_mostrar], use_container_width=True, height=400)

        st.divider()

        # --- REPORTES CON FILTRO POR FECHA ---
        st.subheader("📥 Exportar Reporte por Rango de Fechas")

        # Obtener fechas mínima y máxima del historial
        min_fecha = df_h['Fecha'].min().date()
        max_fecha = df_h['Fecha'].max().date()

        col_fecha1, col_fecha2 = st.columns(2)
        with col_fecha1:
            fecha_inicio = st.date_input("Fecha de inicio", value=min_fecha, min_value=min_fecha, max_value=max_fecha)
        with col_fecha2:
            fecha_fin = st.date_input("Fecha de fin", value=max_fecha, min_value=min_fecha, max_value=max_fecha)

        # Filtrar dataframe por rango de fechas
        mask = (df_h['Fecha'].dt.date >= fecha_inicio) & (df_h['Fecha'].dt.date <= fecha_fin)
        df_filtrado = df_h.loc[mask].copy()

        if df_filtrado.empty:
            st.warning("No hay registros en el rango de fechas seleccionado.")
        else:
            st.success(f"Se encontraron {len(df_filtrado)} registros entre {fecha_inicio} y {fecha_fin}.")
            # Botón de descarga para el filtro
            csv_buffer_filtrado = io.StringIO()
            df_filtrado.to_csv(csv_buffer_filtrado, index=False)
            st.download_button(
                label=f"📎 Descargar Reporte ({fecha_inicio} a {fecha_fin})",
                data=csv_buffer_filtrado.getvalue(),
                file_name=f"reporte_{fecha_inicio}_{fecha_fin}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.divider()

        # --- DESCARGA GENERAL (todo el historial) ---
        st.subheader("📥 Exportar Reporte General (Completo)")
        csv_buffer_general = io.StringIO()
        df_h.to_csv(csv_buffer_general, index=False)
        st.download_button(
            label="📎 Descargar Historial Completo",
            data=csv_buffer_general.getvalue(),
            file_name="reporte_completo_historial.csv",
            mime="text/csv",
            use_container_width=True
        )

        # --- Botón borrar historial ---
        if st.button("🗑️ Borrar Todo el Historial", type="secondary"):
            if os.path.exists("historial.csv"):
                os.remove("historial.csv")
                st.success("Historial eliminado. Recarga la página para ver los cambios.")
                st.rerun()
    else:
        st.info("💡 No hay datos históricos. Por favor, realiza una simulación en la Pestaña 1 (Monitor).")
