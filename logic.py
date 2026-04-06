import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report
import os

# --- FUNCIÓN 1: EL CRONÓMETRO (Eficiencia) ---
def calcular_eficiencia(inicio, fin, total_registros):
    tiempo_total = fin - inicio
    tiempo_por_registro = tiempo_total / total_registros if total_registros > 0 else 0
    return round(tiempo_total, 4), round(tiempo_por_registro, 6)

# --- FUNCIÓN 2: LA MATRIZ DE CONFUSIÓN (Efectividad) ---
def generar_metricas_detalladas(y_real, y_pred):
    # Genera la matriz: [VP, FP, FN, VN]
    cm = confusion_matrix(y_real, y_pred)
    reporte = classification_report(y_real, y_pred, output_dict=True)
    return cm, reporte

# --- FUNCIÓN 3: EL HISTORIAL (Para el Dashboard por día) ---
def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo):
    fecha = time.strftime("%Y-%m-%d %H:%M:%S")
    # Si el archivo no existe, le ponemos encabezados
    nuevo_registro = pd.DataFrame([{
        "Fecha": fecha,
        "Dataset": nombre_dataset,
        "Total_Registros": total,
        "Ataques_Detectados": ataques,
        "Tiempo_Segundos": tiempo
    }])
    
    if not os.path.isfile(archivo_hist):
        nuevo_registro.to_csv(archivo_hist, index=False)
    else:
        nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)

# --- FUNCIÓN 4: ANÁLISIS DE PUERTOS (Lo que pidió el profe) ---
def analizar_puertos(df, y_pred):
    # Solo nos interesan los registros que el modelo marcó como ATAQUE (1)
    df_resultados = df.copy()
    df_resultados['Prediccion'] = y_pred
    ataques = df_resultados[df_resultados['Prediccion'] == 1]
    
    if 'Destination Port' in ataques.columns:
        top_puertos = ataques['Destination Port'].value_counts().head(5)
        return top_puertos
    return None
