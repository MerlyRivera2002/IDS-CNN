import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report
import os

# --- FUNCIÓN 1: EL CRONÓMETRO ---
def calcular_eficiencia(inicio, fin, total_registros):
    tiempo_total = fin - inicio
    tiempo_por_registro = tiempo_total / total_registros if total_registros > 0 else 0
    return round(tiempo_total, 4), round(tiempo_por_registro, 6)

# --- FUNCIÓN 2: LA MATRIZ DE CONFUSIÓN ---
def generar_metricas_detalladas(y_real, y_pred):
    cm = confusion_matrix(y_real, y_pred)
    # Si y_real o y_pred solo tienen una clase, classification_report puede fallar, 
    # por eso usamos zero_division=0
    reporte = classification_report(y_real, y_pred, output_dict=True, zero_division=0)
    return cm, reporte

# --- FUNCIÓN 3: EL HISTORIAL (Nombres de columnas sincronizados con app.py) ---
def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo):
    fecha = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # IMPORTANTE: Estos nombres deben ser IGUALES a los que busca el app.py
    nuevo_registro = pd.DataFrame([{
        "Fecha": fecha,
        "Dataset": nombre_dataset,
        "Registros_Procesados": total,  # Cambiado para coincidir con app.py
        "Ataques_Detectados": ataques,
        "Tiempo_Ejecucion_Seg": tiempo   # Cambiado para coincidir con app.py
    }])
    
    if not os.path.isfile(archivo_hist):
        nuevo_registro.to_csv(archivo_hist, index=False)
    else:
        # Si el archivo ya tiene datos, añadimos sin repetir el encabezado
        nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)

# --- FUNCIÓN 4: ANÁLISIS DE PUERTOS ---
def analizar_puertos(df, y_pred):
    df_resultados = df.copy()
    df_resultados['Prediccion'] = y_pred
    ataques = df_resultados[df_resultados['Prediccion'] == 1]
    
    # Buscamos la columna de puerto (a veces tiene espacios al inicio/final)
    col_puerto = next((c for c in df.columns if 'port' in c.lower()), None)
    
    if col_puerto and not ataques.empty:
        top_puertos = ataques[col_puerto].value_counts().head(5)
        return top_puertos
    return None
