import pandas as pd
import os
import time

def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo, fecha_simulada, puerto_top):
    # Creamos el registro con la fecha que elegiste en el calendario
    nuevo_registro = pd.DataFrame([{
        "Fecha": fecha_simulada,
        "Dataset": nombre_dataset,
        "Registros_Procesados": total,
        "Ataques_Detectados": ataques,
        "Puerto_Critico": puerto_top,
        "Tiempo_Ejecucion_Seg": round(tiempo, 2)
    }])
    
    # Si el archivo no existe, lo crea con cabecera. Si existe, añade fila abajo.
    if not os.path.isfile(archivo_hist):
        nuevo_registro.to_csv(archivo_hist, index=False)
    else:
        nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)

def obtener_puerto_critico(df, predicciones):
    # Filtramos solo las filas que la IA marcó como ataque (1)
    ataques_df = df[predicciones == 1]
    if not ataques_df.empty:
        # Buscamos el puerto que más se repite en la columna ' Destination Port'
        # El [0] es porque mode() devuelve una lista
        return ataques_df[' Destination Port'].mode()[0]
    return "N/A"
