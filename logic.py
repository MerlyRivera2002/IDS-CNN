import pandas as pd
import os
from datetime import datetime

def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo, fecha_simulada, puerto_top, acc, precision=None, recall=None, f1=None):
    """
    Guarda los resultados de la simulación incluyendo hora exacta.
    """
    normales = total - ataques
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nuevo_registro = pd.DataFrame([{
        "Fecha": str(fecha_simulada),
        "Hora": ahora,
        "Dataset": nombre_dataset,
        "Total": total,
        "Normales": normales,
        "Ataques": ataques,
        "Accuracy": round(float(acc), 4) if acc is not None else None,
        "Precision": round(float(precision), 4) if precision is not None else None,
        "Recall": round(float(recall), 4) if recall is not None else None,
        "F1": round(float(f1), 4) if f1 is not None else None,
        "Puerto": f"Port {puerto_top}",
        "Tiempo (s)": round(tiempo, 2)
    }])
    
    try:
        if not os.path.isfile(archivo_hist):
            nuevo_registro.to_csv(archivo_hist, index=False)
        else:
            nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)
        return True
    except Exception as e:
        print(f"Error al guardar historial: {e}")
        return False

def obtener_metricas_resumen(archivo_hist):
    if os.path.exists(archivo_hist):
        try:
            df = pd.read_csv(archivo_hist)
            if not df.empty:
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                # Asegurar columnas numéricas
                for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df.sort_values('Fecha', ascending=True)
        except Exception as e:
            print(f"Error al leer historial: {e}")
            return None
    return None
