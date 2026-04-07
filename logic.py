import pandas as pd
import os

def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo, fecha_simulada, puerto_top):
    # Creamos el registro con la fecha elegida
    nuevo_registro = pd.DataFrame([{
        "Fecha": str(fecha_simulada),
        "Dataset": nombre_dataset,
        "Registros": total,
        "Ataques": ataques,
        "Puerto_Critico": puerto_top,
        "Tiempo_Seg": round(tiempo, 2)
    }])
    
    # Guardado seguro
    if not os.path.isfile(archivo_hist) or os.path.getsize(archivo_hist) == 0:
        nuevo_registro.to_csv(archivo_hist, index=False)
    else:
        nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)

def obtener_puerto_critico(df, predicciones):
    # Filtramos solo las filas detectadas como ataque
    ataques_df = df[predicciones == 1]
    
    if not ataques_df.empty:
        # Buscamos dinámicamente la columna que contenga 'Port'
        cols_puerto = [c for c in df.columns if 'Port' in c]
        if cols_puerto:
            nombre_col = cols_puerto[0]
            # Retorna el valor más frecuente (Moda)
            return ataques_df[nombre_col].mode()[0]
            
    return "N/A"
