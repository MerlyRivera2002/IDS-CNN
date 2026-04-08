import pandas as pd
import numpy as np
import os

def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo, fecha_simulada, puerto_top, acc):
    """
    Guarda los resultados de la simulación de forma persistente.
    Usa el modo 'a' (append) para acumular datos para el Capítulo 4.
    """
    normales = total - ataques
    nuevo_registro = pd.DataFrame([{
        "Fecha": str(fecha_simulada),
        "Dataset": nombre_dataset,
        "Total": total,
        "Normales": normales,
        "Ataques": ataques,
        "Puerto": f"Port {puerto_top}",
        "Tiempo": round(tiempo, 2),
        "Accuracy": round(float(acc), 4)
    }])
    
    try:
        if not os.path.isfile(archivo_hist):
            nuevo_registro.to_csv(archivo_hist, index=False)
        else:
            # mode='a' permite que los datos se graben uno debajo del otro
            nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)
        return True
    except Exception as e:
        print(f"Error al guardar historial: {e}")
        return False

def obtener_metricas_resumen(archivo_hist):
    """
    Lee el historial y prepara los datos para las gráficas de la Pestaña 2.
    """
    if os.path.exists(archivo_hist):
        try:
            df = pd.read_csv(archivo_hist)
            if not df.empty:
                # Convertimos a datetime y ordenamos para que la gráfica de tendencia sea fluida
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                return df.sort_values('Fecha', ascending=True)
        except Exception as e:
            print(f"Error al leer historial: {e}")
            return None
    return None

def generar_estadisticas_puertos(df):
    """
    Función extra para el reporte: Agrupa ataques por puerto para ver 
    cuál es el más crítico en todo el historial.
    """
    if df is not None and not df.empty:
        # Agrupamos por puerto y sumamos los ataques detectados en cada uno
        resumen = df.groupby('Puerto')['Ataques'].sum().reset_index()
        return resumen.sort_values('Ataques', ascending=False)
    return None
