import pandas as pd
import numpy as np
import os

def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo, fecha_simulada, puerto_top, acc):
    nuevo_registro = pd.DataFrame([{
        "Fecha": str(fecha_simulada),
        "Dataset": nombre_dataset,
        "Total": total,
        "Ataques": ataques,
        "Puerto": f"Port {puerto_top}",
        "Tiempo": round(tiempo, 2),
        "Accuracy": f"{acc:.2%}" # Guardamos el porcentaje para el reporte
    }]).
    
    if not os.path.isfile(archivo_hist) or os.path.getsize(archivo_hist) == 0:
        nuevo_registro.to_csv(archivo_hist, index=False)
    else:
        nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)

def obtener_metricas_resumen(archivo_hist):
    if os.path.exists(archivo_hist):
        try:
            df = pd.read_csv(archivo_hist)
            if not df.empty:
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                return df.sort_values('Fecha')
        except:
            return None
    return None
