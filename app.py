import os
import io
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)
import logic

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IDS · CNN Tesis 2026",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# TEMA VISUAL — INYECCIÓN CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Fondo general ── */
html, body, [data-testid="stApp"] {
    background: #080f1a !important;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1929 !important;
    border-right: 1px solid #1a3a5c;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] input {
    background: #0a1622 !important;
    border: 1px solid #1e4a7a !important;
    color: #e0eaf5 !important;
    border-radius: 6px;
}
[data-testid="stSidebar"] .stButton > button {
    background: #0a2540 !important;
    border: 1px solid #1e6fbf !important;
    color: #7ec8f7 !important;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #0f3560 !important;
    border-color: #00c8ff !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #0d1929;
    border-radius: 10px 10px 0 0;
    padding: 6px 8px 0;
    border-bottom: 1px solid #1a3a5c;
    gap: 4px;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #6a8eaa !important;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    border-radius: 7px 7px 0 0;
    padding: 8px 20px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    transition: all .2s;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #7ec8f7 !important;
    background: #0a1f38;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #00c8ff !important;
    background: #0a1f38;
    border-bottom: 2px solid #00c8ff;
}
[data-testid="stTabsContent"] {
    background: #0d1929;
    border-radius: 0 0 10px 10px;
    padding: 1.5rem;
    border: 1px solid #1a3a5c;
    border-top: none;
}

/* ── Métricas ── */
[data-testid="stMetric"] {
    background: #0a1f38;
    border: 1px solid #1a4a7a;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #7ea8c8 !important; font-size: 12px !important; font-family: 'Space Mono', monospace; text-transform: uppercase; letter-spacing: .06em; }
[data-testid="stMetricValue"] { color: #00c8ff !important; font-size: 26px !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] { color: #ff5555 !important; font-size: 13px !important; }

/* ── Headers ── */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #e0eaf5 !important; }
h1 { font-size: 22px !important; letter-spacing: .04em; }
h2 { font-size: 17px !important; color: #7ec8f7 !important; }
h3 { font-size: 14px !important; color: #6a9ec0 !important; }

/* ── Texto general ── */
p, span, label, div { color: #c8d8e8; }

/* ── Dataframe / tablas ── */
[data-testid="stDataFrame"], iframe {
    border: 1px solid #1a3a5c !important;
    border-radius: 8px !important;
    background: #080f1a !important;
}
.stDataFrame td, .stDataFrame th {
    font-size: 13px !important;
    color: #c8d8e8 !important;
    background: #0a1622 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #1e4a7a !important;
    border-radius: 10px;
    background: #080f1a !important;
    padding: 12px;
}

/* ── Slider ── */
[data-testid="stSlider"] .stSlider > div { background: #00c8ff; }

/* ── Botones principales ── */
.stButton > button {
    background: linear-gradient(135deg, #0a4080, #0d5fa0) !important;
    color: #e0eaf5 !important;
    border: 1px solid #1e6fbf !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: .04em;
    transition: all .2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0d5fa0, #0a7acf) !important;
    border-color: #00c8ff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,200,255,.2);
}
[data-testid="stButton-primary"] > button {
    background: linear-gradient(135deg, #00829a, #00c8ff) !important;
    color: #060e18 !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0060a0, #00c8ff) !important;
}

/* ── Alerts ── */
[data-testid="stSuccess"] { background: #0a2818 !important; border-left: 3px solid #00c853 !important; color: #8affa0 !important; }
[data-testid="stWarning"] { background: #1e1500 !important; border-left: 3px solid #ffaa00 !important; color: #ffd060 !important; }
[data-testid="stError"]   { background: #1e0808 !important; border-left: 3px solid #ff3333 !important; color: #ff8080 !important; }
[data-testid="stInfo"]    { background: #081828 !important; border-left: 3px solid #0080c0 !important; color: #7ec8f7 !important; }

/* ── Divider ── */
hr { border-color: #1a3a5c !important; }

/* ── Date input ── */
[data-testid="stDateInput"] input {
    background: #0a1622 !important;
    border: 1px solid #1e4a7a !important;
    color: #e0eaf5 !important;
    border-radius: 6px;
}

/* ── Select box ── */
[data-testid="stSelectbox"] > div {
    background: #0a1622 !important;
    border: 1px solid #1e4a7a !important;
    border-radius: 6px;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: #0a1f38 !important;
    border: 1px solid #1e6fbf !important;
    color: #7ec8f7 !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PALETA PLOTLY — TEMA OSCURO COHERENTE
# ═══════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#080f1a",
    font=dict(family="DM Sans", color="#c8d8e8", size=13),
    margin=dict(t=40, b=30, l=10, r=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c8d8e8")),
    xaxis=dict(gridcolor="#142035", linecolor="#1a3a5c", tickcolor="#1a3a5c", tickfont=dict(color="#7ea8c8")),
    yaxis=dict(gridcolor="#142035", linecolor="#1a3a5c", tickcolor="#1a3a5c", tickfont=dict(color="#7ea8c8")),
    title_font=dict(family="Space Mono", color="#e0eaf5", size=14),
)
COLOR_ATAQUE  = "#ff4f5e"
COLOR_NORMAL  = "#00c8ff"
COLOR_ACCENT  = "#7ec8f7"
COLOR_WARN    = "#ffaa00"
COLOR_SUCCESS = "#00c853"

def apply_theme(fig, height=320):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig


# ═══════════════════════════════════════════════════════════════
# LOGIN
# ═══════════════════════════════════════════════════════════════
if "perfil" not in st.session_state:
    st.session_state.perfil = None

st.sidebar.markdown("### 🔐 Control de Acceso")
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
    st.sidebar.success(f"✔ {st.session_state.perfil}")
    st.sidebar.divider()
    st.sidebar.markdown("**📅 Fecha de simulación**")
    fecha_simulada = st.sidebar.date_input("", value=pd.to_datetime("2026-04-01"), label_visibility="collapsed")
    st.sidebar.divider()
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# CARGA DE MODELO
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_assets():
    return (
        tf.keras.models.load_model("modelo_cnn.keras"),
        joblib.load("scaler.pkl"),
        joblib.load("features.pkl"),
    )

model, scaler, features_list = load_assets()

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#0a1f38,#0d2d52);border:1px solid #1a4a7a;border-radius:12px;padding:18px 24px;margin-bottom:1.4rem;display:flex;align-items:center;gap:16px;">
  <div style="font-size:36px;">🛡️</div>
  <div>
    <div style="font-family:'Space Mono',monospace;font-size:20px;color:#e0eaf5;font-weight:700;letter-spacing:.05em;">SISTEMA DE DETECCIÓN DE INTRUSIONES</div>
    <div style="font-family:'DM Sans',sans-serif;color:#7ea8c8;font-size:13px;margin-top:3px;">Red Neuronal Convolucional 1D · Clasificación Binaria · CICIDS2017</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🚀  SIMULACIÓN EN VIVO",
    "📈  ANÁLISIS Y TENDENCIAS",
    "📋  MOVIMIENTOS Y REPORTES",
])

# ╔══════════════════════════════════════════════════════════════╗
# ║                     PESTAÑA 1                               ║
# ╚══════════════════════════════════════════════════════════════╝
with tab1:
    if st.session_state.perfil == "Administrador":
        st.header("Monitor de Tráfico en Tiempo Real")

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([3, 1.5, 1])
        with col_ctrl1:
            archivo = st.file_uploader("Subir dataset CSV", type=["csv"], key="uploader_sim")
        with col_ctrl2:
            velocidad = st.slider("Seg / lote", 0.02, 0.5, 0.08, 0.02)
        with col_ctrl3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Limpiar", use_container_width=True):
                st.session_state.pop("simulacion_activa", None)
                st.rerun()

        col_izq, col_der = st.columns(2)
        with col_izq:
            esp_pastel   = st.empty()
            esp_evolucion = st.empty()
        with col_der:
            esp_metricas = st.empty()
            esp_puertos  = st.empty()

        st.divider()
        st.subheader("Registro de actividad — último lote")
        esp_tabla = st.empty()

        if "simulacion_activa" not in st.session_state:
            st.session_state.simulacion_activa = False
            for e in [esp_pastel, esp_evolucion, esp_metricas, esp_puertos, esp_tabla]:
                e.info("Esperando inicio de simulación…")

        if archivo:
            if st.button("🚀 INICIAR MONITOREO", type="primary", use_container_width=True):
                with st.spinner("Preparando datos…"):
                    df_raw   = pd.read_csv(archivo)
                    df_raw.columns = df_raw.columns.str.strip()
                    df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()
                    total    = len(df_clean)

                    preds_totales = []
                    tiempos, ataques_acum = [], []
                    t0 = time.time()
                    pbar = st.progress(0, text="Simulación en curso…")

                    for idx, inicio in enumerate(range(0, total, 15)):
                        chunk = df_clean.iloc[inicio:inicio+15]
                        X_chunk = scaler.transform(
                            chunk[features_list]
                        ).reshape(-1, len(features_list), 1)
                        chunk_preds = (
                            model.predict(X_chunk, verbose=0) > 0.35
                        ).astype(int).flatten()
                        preds_totales.extend(chunk_preds)

                        ataques = int(sum(preds_totales))
                        normales = len(preds_totales) - ataques
                        ataques_acum.append(ataques)
                        tiempos.append(len(preds_totales))

                        prog = (inicio + len(chunk)) / total
                        pbar.progress(min(prog, 1.0), text=f"Procesando {int(prog*100)}%")

                        # ── Pastel ──
                        fig_pie = go.Figure(go.Pie(
                            values=[normales, ataques],
                            labels=["Normal", "Ataque"],
                            hole=0.62,
                            marker=dict(colors=[COLOR_NORMAL, COLOR_ATAQUE],
                                        line=dict(color="#080f1a", width=3)),
                            textfont=dict(size=13, color="#e0eaf5"),
                        ))
                        fig_pie.update_layout(
                            **PLOTLY_LAYOUT, height=280,
                            title=f"Tráfico — {len(preds_totales)} registros",
                            showlegend=True,
                            legend=dict(orientation="h", y=-0.05,
                                        font=dict(color="#c8d8e8", size=12)),
                            annotations=[dict(
                                text=f"<b>{ataques}</b><br><span style='font-size:10px'>ataques</span>",
                                x=0.5, y=0.5, font=dict(size=16, color=COLOR_ATAQUE),
                                showarrow=False
                            )]
                        )
                        esp_pastel.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{idx}")

                        # ── Evolución ──
                        fig_evol = go.Figure()
                        fig_evol.add_trace(go.Scatter(
                            x=tiempos, y=ataques_acum, mode="lines",
                            fill="tozeroy",
                            line=dict(color=COLOR_ATAQUE, width=3),
                            fillcolor="rgba(255,79,94,.15)",
                            name="Ataques acumulados"
                        ))
                        fig_evol.update_layout(**PLOTLY_LAYOUT, height=240,
                                               title="Evolución de intrusiones",
                                               xaxis_title="Registros procesados",
                                               yaxis_title="Ataques")
                        esp_evolucion.plotly_chart(fig_evol, use_container_width=True, key=f"evol_{idx}")

                        # ── Métricas dinámicas ──
                        with esp_metricas.container():
                            m1, m2 = st.columns(2)
                            m1.metric("Conexiones", f"{len(preds_totales):,}")
                            m2.metric("Intrusiones", f"{ataques:,}", delta=f"+{int(chunk_preds.sum())}")
                            elapsed = time.time() - t0
                            m3, m4 = st.columns(2)
                            m3.metric("Reg/seg", f"{len(preds_totales)/max(elapsed,0.01):.1f}")
                            col_label = next((c for c in df_clean.columns if c.lower() == "label"), None)
                            if col_label:
                                y_true_p = df_clean[col_label].iloc[:len(preds_totales)].astype(str).str.upper().apply(
                                    lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1)
                                acc_p = accuracy_score(y_true_p, preds_totales)
                                m4.metric("Accuracy parcial", f"{acc_p:.2%}")

                        # ── Puertos ──
                        with esp_puertos.container():
                            puertos = chunk["Destination Port"].value_counts().head(5).reset_index()
                            puertos.columns = ["Puerto", "Freq"]
                            fig_p = go.Figure(go.Bar(
                                x=puertos["Puerto"].astype(str),
                                y=puertos["Freq"],
                                marker=dict(color=puertos["Freq"],
                                            colorscale=[[0,"#0a3060"],[1,COLOR_ATAQUE]],
                                            line=dict(width=0)),
                                text=puertos["Freq"], textposition="outside",
                                textfont=dict(color="#c8d8e8", size=12),
                            ))
                            fig_p.update_layout(**PLOTLY_LAYOUT, height=210,
                                                title="Top 5 puertos — lote actual",
                                                xaxis_title="Puerto destino",
                                                yaxis_title="Paquetes")
                            st.plotly_chart(fig_p, use_container_width=True, key=f"puertos_{idx}")

                        # ── Tabla ──
                        with esp_tabla.container():
                            vista = chunk[["Destination Port"] + [c for c in chunk.columns if c != "Destination Port"][:4]].copy()
                            vista.insert(0, "Estado", ["🚨 ATAQUE" if p == 1 else "✅ NORMAL" for p in chunk_preds])
                            def diagnose(row):
                                if "NORMAL" in row["Estado"]:
                                    return "Tráfico seguro"
                                p = row["Destination Port"]
                                if p in [80, 443]: return "Ataque web HTTP/S"
                                if p == 22: return "Fuerza bruta SSH"
                                if p == 21: return "Acceso FTP no autorizado"
                                return "Escaneo / puerto sospechoso"
                            vista["Diagnóstico"] = vista.apply(diagnose, axis=1)
                            # Tabla con scroll completo
                            st.dataframe(
                                vista,
                                use_container_width=True,
                                height=400,
                                hide_index=True,
                            )

                        time.sleep(velocidad)

                    pbar.empty()
                    st.success("✅ Simulación finalizada.")
                    st.balloons()

                    # ── Resumen ──
                    st.subheader("Resumen ejecutivo")
                    elapsed_total = time.time() - t0
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total registros",    f"{len(preds_totales):,}")
                    c2.metric("Ataques detectados", f"{ataques:,}",
                              delta=f"{ataques/len(preds_totales):.1%}")
                    c3.metric("Tiempo simulación",  f"{elapsed_total:.2f} s")
                    c4.metric("Eficiencia",         f"{len(preds_totales)/elapsed_total:.1f} reg/s")

                    col_label = next((c for c in df_clean.columns if c.lower() == "label"), None)
                    if col_label:
                        y_true = df_clean[col_label].astype(str).str.upper().apply(
                            lambda x: 0 if "BENIGN" in x or "NORMAL" in x else 1
                        )[:len(preds_totales)]
                        acc  = accuracy_score(y_true,  preds_totales)
                        prec = precision_score(y_true, preds_totales, zero_division=0)
                        rec  = recall_score(y_true,    preds_totales, zero_division=0)
                        f1   = f1_score(y_true,        preds_totales, zero_division=0)

                        # Gráfico métricas
                        st.subheader("Evaluación del modelo CNN")
                        df_met = pd.DataFrame({
                            "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score"],
                            "Valor":   [acc*100, prec*100, rec*100, f1*100],
                        })
                        fig_met = go.Figure()
                        colors_met = [COLOR_NORMAL, COLOR_SUCCESS, COLOR_WARN, COLOR_ATAQUE]
                        for i, row in df_met.iterrows():
                            fig_met.add_trace(go.Bar(
                                name=row["Métrica"],
                                x=[row["Métrica"]],
                                y=[row["Valor"]],
                                marker_color=colors_met[i],
                                text=[f"{row['Valor']:.2f}%"],
                                textposition="outside",
                                textfont=dict(size=14, color="#e0eaf5"),
                                width=0.5,
                            ))
                        fig_met.update_layout(
                            **PLOTLY_LAYOUT, height=360,
                            title="Métricas de clasificación binaria",
                            showlegend=False,
                            yaxis=dict(range=[0, 105], gridcolor="#142035",
                                       ticksuffix="%", tickfont=dict(color="#7ea8c8")),
                            bargap=0.35,
                        )
                        st.plotly_chart(fig_met, use_container_width=True)

                        # Matriz de confusión
                        st.subheader("Matriz de Confusión")
                        cm = confusion_matrix(y_true, preds_totales)
                        fig_cm = go.Figure(go.Heatmap(
                            z=cm,
                            x=["Pred: Normal", "Pred: Ataque"],
                            y=["Real: Normal", "Real: Ataque"],
                            colorscale=[[0, "#0a1f38"], [0.5, "#0060a0"], [1, COLOR_ATAQUE]],
                            text=cm, texttemplate="<b>%{text}</b>",
                            textfont=dict(size=22, color="#ffffff"),
                            showscale=True,
                            colorbar=dict(tickfont=dict(color="#c8d8e8")),
                        ))
                        fig_cm.update_layout(**PLOTLY_LAYOUT, height=340,
                                             title="Matriz de Confusión — Normal vs Ataque",
                                             xaxis=dict(title="Predicción", side="bottom",
                                                        tickfont=dict(size=13, color="#7ea8c8")),
                                             yaxis=dict(title="Valor Real",
                                                        tickfont=dict(size=13, color="#7ea8c8")))
                        st.plotly_chart(fig_cm, use_container_width=True)

                        p_top = df_clean.iloc[:len(preds_totales)]["Destination Port"].mode()[0]
                        logic.guardar_en_historial(
                            "historial.csv", archivo.name, len(preds_totales), ataques,
                            elapsed_total, fecha_simulada, p_top, acc,
                            precision=prec, recall=rec, f1=f1,
                        )
                    else:
                        p_top = df_clean.iloc[:len(preds_totales)]["Destination Port"].mode()[0]
                        logic.guardar_en_historial(
                            "historial.csv", archivo.name, len(preds_totales), ataques,
                            elapsed_total, fecha_simulada, p_top, 0.0,
                        )
                    st.toast("Simulación guardada en bitácora ✔")
    else:
        st.warning("🔒 Solo Administradores pueden ejecutar simulaciones.")


# ╔══════════════════════════════════════════════════════════════╗
# ║                     PESTAÑA 2                               ║
# ╚══════════════════════════════════════════════════════════════╝
with tab2:
    st.header("Análisis Histórico y Tendencias")

    archivo_local = "historial.csv"

    if not os.path.exists(archivo_local):
        st.warning("Sin datos históricos. Ejecuta una simulación primero.")
        st.stop()

    try:
        df_h = pd.read_csv(archivo_local)
    except Exception as e:
        st.error(f"Error al leer historial: {e}")
        if st.button("🗑️ Borrar archivo corrupto"):
            os.remove(archivo_local)
            st.rerun()
        st.stop()

    if df_h.empty:
        st.warning("Historial vacío. Ejecuta una simulación.")
        st.stop()

    df_h["Fecha"] = pd.to_datetime(df_h["Fecha"], errors="coerce")
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        if col in df_h.columns:
            df_h[col] = pd.to_numeric(df_h[col], errors="coerce")
    df_h = df_h.dropna(subset=["Fecha"]).sort_values("Fecha")

    # ── KPIs ──
    st.subheader("Resumen global")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total simulaciones",      len(df_h))
    k2.metric("Total ataques detectados", f"{df_h['Ataques'].sum():,}")
    k3.metric("Accuracy promedio",        f"{df_h['Accuracy'].mean():.2%}")
    k4.metric("Puerto más atacado",       df_h.loc[df_h["Ataques"].idxmax(), "Puerto"])

    st.divider()

    # ── Evolución temporal de intrusiones ──
    st.subheader("Evolución temporal de intrusiones")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_h["Fecha"], y=df_h["Ataques"],
        mode="lines+markers",
        line=dict(color=COLOR_ATAQUE, width=3),
        marker=dict(size=9, color=COLOR_ATAQUE,
                    line=dict(color="#080f1a", width=2)),
        fill="tozeroy", fillcolor="rgba(255,79,94,.12)",
        name="Ataques"
    ))
    fig_line.update_layout(**PLOTLY_LAYOUT, height=320,
                           title="Ataques detectados a lo largo del tiempo",
                           xaxis_title="Fecha", yaxis_title="Cantidad de ataques",
                           xaxis=dict(tickformat="%d %b %Y", tickangle=-30,
                                      gridcolor="#142035", tickfont=dict(color="#7ea8c8")))
    st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # ── Puertos más atacados — TREEMAP (más "perrón") ──
    st.subheader("Distribución de puertos atacados")
    puertos_df = (
        df_h["Puerto"].value_counts().reset_index()
    )
    puertos_df.columns = ["Puerto", "Frecuencia"]
    puertos_df["Puerto"] = puertos_df["Puerto"].astype(str)

    fig_tree = go.Figure(go.Treemap(
        labels=puertos_df["Puerto"],
        parents=[""] * len(puertos_df),
        values=puertos_df["Frecuencia"],
        textinfo="label+value+percent root",
        textfont=dict(size=14, color="#ffffff"),
        marker=dict(
            colorscale=[[0, "#0a2040"], [0.4, "#0060a0"], [1, COLOR_ATAQUE]],
            colorbar=dict(tickfont=dict(color="#c8d8e8")),
            line=dict(width=2, color="#080f1a"),
        ),
        hovertemplate="<b>%{label}</b><br>Frecuencia: %{value}<extra></extra>",
    ))
    fig_tree.update_layout(**PLOTLY_LAYOUT, height=380,
                           title="Puertos más atacados (tamaño = frecuencia)")
    st.plotly_chart(fig_tree, use_container_width=True)

    st.divider()

    # ── Métricas globales interactivas ──
    st.subheader("Métricas globales del modelo")

    acc_g  = df_h["Accuracy"].mean()
    prec_g = df_h["Precision"].mean()
    rec_g  = df_h["Recall"].mean()
    f1_g   = df_h["F1"].mean()

    if "metrica_activa" not in st.session_state:
        st.session_state.metrica_activa = None

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Accuracy",  f"{acc_g:.2%}")
        if st.button("Ver evolución", key="btn_acc"): st.session_state.metrica_activa = "Accuracy"
    with m2:
        st.metric("Precision", f"{prec_g:.2%}")
        if st.button("Ver evolución", key="btn_prec"): st.session_state.metrica_activa = "Precision"
    with m3:
        st.metric("Recall",    f"{rec_g:.2%}")
        if st.button("Ver evolución", key="btn_rec"): st.session_state.metrica_activa = "Recall"
    with m4:
        st.metric("F1-Score",  f"{f1_g:.2%}")
        if st.button("Ver evolución", key="btn_f1"): st.session_state.metrica_activa = "F1"

    if st.session_state.metrica_activa:
        met = st.session_state.metrica_activa
        colors_map = {"Accuracy": COLOR_NORMAL, "Precision": COLOR_SUCCESS,
                      "Recall": COLOR_WARN, "F1": COLOR_ATAQUE}
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(
            x=df_h["Fecha"], y=df_h[met],
            mode="lines+markers",
            line=dict(color=colors_map[met], width=3),
            marker=dict(size=9, color=colors_map[met],
                        line=dict(color="#080f1a", width=2)),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(colors_map[met].lstrip('#')[i:i+2],16)) for i in (0,2,4))},.12)",
        ))
        fig_ev.update_layout(
            **PLOTLY_LAYOUT, height=300,
            title=f"Evolución de {met}",
            yaxis=dict(tickformat=".0%", gridcolor="#142035",
                       tickfont=dict(color="#7ea8c8")),
            xaxis=dict(tickformat="%d %b %Y", tickangle=-30,
                       gridcolor="#142035", tickfont=dict(color="#7ea8c8")),
        )
        st.plotly_chart(fig_ev, use_container_width=True)

    st.divider()

    # ── MATRIZ DE CONFUSIÓN ACUMULADA ──
    st.subheader("Matriz de Confusión acumulada (todas las simulaciones)")

    cols_cm = ["TN", "FP", "FN", "TP"]
    if all(c in df_h.columns for c in cols_cm):
        tn = int(df_h["TN"].sum())
        fp = int(df_h["FP"].sum())
        fn = int(df_h["FN"].sum())
        tp = int(df_h["TP"].sum())
    else:
        # Estimación a partir de métricas si no hay columnas individuales
        total_s   = df_h["Total"].sum() if "Total" in df_h.columns else 1000
        ataques_s = df_h["Ataques"].sum() if "Ataques" in df_h.columns else 100
        normales_s = total_s - ataques_s
        tp = int(ataques_s * rec_g)
        fn = int(ataques_s - tp)
        fp = int(tp / max(prec_g, 0.0001) - tp)
        tn = int(normales_s - fp)

    cm_vals = [[tn, fp], [fn, tp]]
    fig_cm_hist = go.Figure(go.Heatmap(
        z=cm_vals,
        x=["Pred: Normal", "Pred: Ataque"],
        y=["Real: Normal", "Real: Ataque"],
        colorscale=[[0, "#0a1f38"], [0.5, "#0060a0"], [1, COLOR_ATAQUE]],
        text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=20, color="#ffffff"),
        showscale=True,
        colorbar=dict(tickfont=dict(color="#c8d8e8")),
    ))
    fig_cm_hist.update_layout(
        **PLOTLY_LAYOUT, height=360,
        title="Matriz de Confusión acumulada — Normal vs Ataque",
        xaxis=dict(title="Predicción", tickfont=dict(size=13, color="#7ea8c8")),
        yaxis=dict(title="Valor Real", tickfont=dict(size=13, color="#7ea8c8")),
    )
    st.plotly_chart(fig_cm_hist, use_container_width=True)

    # Métricas derivadas de la CM
    if (tp + fn) > 0 and (tp + fp) > 0:
        recall_calc = tp / (tp + fn)
        prec_calc   = tp / (tp + fp)
        st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:6px;">
          <div style="background:#0a1f38;border:1px solid #1a4a7a;border-radius:8px;padding:10px 18px;">
            <div style="font-size:11px;color:#7ea8c8;font-family:'Space Mono',monospace;text-transform:uppercase;">Verdaderos Negativos</div>
            <div style="font-size:22px;color:#00c8ff;font-weight:600;">{tn:,}</div>
          </div>
          <div style="background:#0a1f38;border:1px solid #1a4a7a;border-radius:8px;padding:10px 18px;">
            <div style="font-size:11px;color:#7ea8c8;font-family:'Space Mono',monospace;text-transform:uppercase;">Falsos Positivos</div>
            <div style="font-size:22px;color:#ffaa00;font-weight:600;">{fp:,}</div>
          </div>
          <div style="background:#0a1f38;border:1px solid #1a4a7a;border-radius:8px;padding:10px 18px;">
            <div style="font-size:11px;color:#7ea8c8;font-family:'Space Mono',monospace;text-transform:uppercase;">Falsos Negativos</div>
            <div style="font-size:22px;color:#ff4f5e;font-weight:600;">{fn:,}</div>
          </div>
          <div style="background:#0a1f38;border:1px solid #1a4a7a;border-radius:8px;padding:10px 18px;">
            <div style="font-size:11px;color:#7ea8c8;font-family:'Space Mono',monospace;text-transform:uppercase;">Verdaderos Positivos</div>
            <div style="font-size:22px;color:#00c853;font-weight:600;">{tp:,}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Tabla detallada ──
    st.subheader("Registro detallado de simulaciones")
    columnas = ["Fecha", "Hora", "Dataset", "Total", "Normales", "Ataques",
                "Accuracy", "Precision", "Recall", "F1", "Puerto", "Tiempo (s)"]
    for col in columnas:
        if col not in df_h.columns:
            df_h[col] = np.nan
    df_disp = df_h.copy()
    df_disp["Fecha"] = df_disp["Fecha"].dt.date
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "—")
    st.dataframe(df_disp[columnas], use_container_width=True, height=400, hide_index=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                     PESTAÑA 3                               ║
# ╚══════════════════════════════════════════════════════════════╝
with tab3:
    st.header("Movimientos y Reportes")
    df_h3 = logic.obtener_metricas_resumen("historial.csv")

    if df_h3 is None or df_h3.empty:
        st.info("💡 Sin datos históricos. Realiza una simulación primero.")
        st.stop()

    # ── Tarjetas resumen ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0a1f38,#0d2d52);border:1px solid #1a4a7a;border-radius:10px;padding:14px 20px;margin-bottom:1.2rem;">
      <span style="font-family:'Space Mono',monospace;color:#7ea8c8;font-size:12px;text-transform:uppercase;letter-spacing:.08em;">Resumen del historial completo</span>
    </div>
    """, unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Simulaciones totales",    len(df_h3))
    s2.metric("Registros analizados",    f"{df_h3['Total'].sum():,}" if "Total" in df_h3 else "—")
    s3.metric("Total ataques detectados", f"{df_h3['Ataques'].sum():,}" if "Ataques" in df_h3 else "—")
    s4.metric("F1 promedio",             f"{df_h3['F1'].mean():.2%}" if "F1" in df_h3 else "—")

    st.divider()

    # ── Listado de movimientos ──
    st.subheader("Listado de movimientos")
    columnas_mov = ["Fecha", "Hora", "Dataset", "Total", "Normales", "Ataques",
                    "Accuracy", "Precision", "Recall", "F1", "Puerto", "Tiempo (s)"]
    col_exist = [c for c in columnas_mov if c in df_h3.columns]
    df_mov = df_h3[col_exist].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        if col in df_mov.columns:
            df_mov[col] = df_mov[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "—")
    st.dataframe(df_mov, use_container_width=True, height=350, hide_index=True)

    st.divider()

    # ── Descarga por rango de fechas ──
    st.subheader("Reporte por rango de fechas")
    min_f = df_h3["Fecha"].min().date()
    max_f = df_h3["Fecha"].max().date()

    fa, fb = st.columns(2)
    with fa: f_ini = st.date_input("Desde", value=min_f, min_value=min_f, max_value=max_f, key="rep_ini")
    with fb: f_fin = st.date_input("Hasta", value=max_f, min_value=min_f, max_value=max_f, key="rep_fin")

    mask = (df_h3["Fecha"].dt.date >= f_ini) & (df_h3["Fecha"].dt.date <= f_fin)
    df_filt = df_h3.loc[mask].copy()

    if df_filt.empty:
        st.warning("Sin registros en ese rango de fechas.")
    else:
        st.success(f"✅ {len(df_filt)} registros entre {f_ini} y {f_fin}")
        df_prev = df_filt[col_exist].copy()
        for col in ["Accuracy", "Precision", "Recall", "F1"]:
            if col in df_prev.columns:
                df_prev[col] = df_prev[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "—")
        st.dataframe(df_prev, use_container_width=True, hide_index=True)

        buf = io.StringIO()
        df_filt.to_csv(buf, index=False)
        st.download_button("📎 Descargar reporte CSV",
                           data=buf.getvalue(),
                           file_name=f"reporte_{f_ini}_{f_fin}.csv",
                           mime="text/csv",
                           use_container_width=True)

    st.divider()

    # ── Exportar todo ──
    st.subheader("Exportar historial completo")
    buf_all = io.StringIO()
    df_h3.to_csv(buf_all, index=False)
    st.download_button("📎 Descargar todas las simulaciones",
                       data=buf_all.getvalue(),
                       file_name="historial_completo.csv",
                       mime="text/csv",
                       use_container_width=True)

    st.divider()

    # ── Zona peligrosa ──
    with st.expander("⚠️ Zona peligrosa"):
        if st.button("Borrar todo el historial", type="secondary"):
            if os.path.exists("historial.csv"):
                os.remove("historial.csv")
                st.success("Historial eliminado.")
                st.rerun()
