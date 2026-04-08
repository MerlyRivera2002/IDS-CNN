import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# --- FUNCIÓN 1: EL CRONÓMETRO (INTACTA) ---
def calcular_eficiencia(inicio, fin, total_registros):
    tiempo_total = fin - inicio
    tiempo_por_registro = tiempo_total / total_registros if total_registros > 0 else 0
    return round(tiempo_total, 4), round(tiempo_por_registro, 6)

# --- FUNCIÓN 2: LA MATRIZ DE CONFUSIÓN (INTACTA) ---
def generar_metricas_detalladas(y_real, y_pred):
    cm = confusion_matrix(y_real, y_pred)
    reporte = classification_report(y_real, y_pred, output_dict=True, zero_division=0)
    return cm, reporte

# --- FUNCIÓN 3: EL HISTORIAL (Ahora guarda el Accuracy para tu gráfica de puntitos) ---
def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo, fecha_simulada, puerto_top, accuracy_val):
    nuevo_registro = pd.DataFrame([{
        "Fecha": str(fecha_simulada), 
        "Dataset": nombre_dataset,
        "Registros_Procesados": total,
        "Ataques_Detectados": ataques,
        "Puerto_Critico": puerto_top,
        "Tiempo_Ejecucion_Seg": tiempo,
        "Accuracy": round(accuracy_val, 4) # <-- NUEVO: Para la tendencia de IA
    }])
    
    if not os.path.isfile(archivo_hist) or os.path.getsize(archivo_hist) == 0:
        nuevo_registro.to_csv(archivo_hist, index=False)
    else:
        nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)

# --- FUNCIÓN 4: ANÁLISIS DE PUERTOS REAL ---
def obtener_puerto_top(df, y_pred):
    col_puerto = next((c for c in df.columns if 'port' in c.lower()), None)
    if col_puerto:
        # Solo miramos los registros que la IA dijo que eran ataque
        df_temp = df.copy()
        ataques = df_temp[np.array(y_pred) == 1]
        if not ataques.empty:
            return ataques[col_puerto].mode()[0]
    return "N/A"
