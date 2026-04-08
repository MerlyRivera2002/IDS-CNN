import pandas as pd
import numpy as np
import os

def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo, fecha_simulada, puerto_top, acc):
    """Guarda los resultados de la simulación. Recibe exactamente 8 argumentos."""
    nuevo_registro = pd.DataFrame([{
        "Fecha": str(fecha_simulada),
        "Dataset": nombre_dataset,
        "Total": total,
        "Ataques": ataques,
        "Puerto": f"Port {puerto_top}",
        "Tiempo": round(tiempo, 2),
        "Accuracy": round(float(acc), 4)
    }])
    
    if not os.path.isfile(archivo_hist) or os.path.getsize(archivo_hist) == 0:
        nuevo_registro.to_csv(archivo_hist, index=False)
    else:
        nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)

def obtener_metricas_resumen(archivo_hist):
    """Carga los datos para las gráficas de la Pestaña 2."""
    if os.path.exists(archivo_hist):
        try:
            df = pd.read_csv(archivo_hist)
            if not df.empty:
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                return df.sort_values('Fecha')
        except:
            return None
    return None
