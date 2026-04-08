import pandas as pd
import numpy as np
import os

def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo, fecha_simulada, puerto_top):
    """Guarda los resultados de la simulación en un archivo CSV."""
    nuevo_registro = pd.DataFrame([{
        "Fecha": str(fecha_simulada),
        "Dataset": nombre_dataset,
        "Registros_Procesados": total,
        "Ataques_Detectados": ataques,
        "Puerto_Critico": puerto_top,
        "Tiempo_Ejecucion_Seg": round(tiempo, 2)
    }])
    
    if not os.path.isfile(archivo_hist) or os.path.getsize(archivo_hist) == 0:
        nuevo_registro.to_csv(archivo_hist, index=False)
    else:
        nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)

def obtener_metricas_resumen(archivo_hist):
    """Lee el historial para las gráficas de la Pestaña 2."""
    if os.path.exists(archivo_hist):
        try:
            df = pd.read_csv(archivo_hist)
            if not df.empty:
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                return df.sort_values('Fecha')
        except:
            return None
    return None
