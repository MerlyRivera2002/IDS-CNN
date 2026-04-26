import pandas as pd
import os
import requests
import base64
from datetime import datetime
import streamlit as st

# ==================================================
# CONFIGURACIÓN DE GITHUB - ¡CAMBIA ESTOS VALORES!
# ==================================================
REPO_OWNER = "MerlyRivera2002"   # Ejemplo: "juanperez"
REPO_NAME = "https://github.com/MerlyRivera2002/IDS-CNN"       # Ejemplo: "ids-tesis"
FILE_PATH = "historial.csv"        # Nombre del archivo en el repo

def guardar_en_historial(archivo_hist, nombre_dataset, total, ataques, tiempo, fecha_simulada, puerto_top, acc, precision=None, recall=None, f1=None):
    """
    Guarda los resultados. Usa API de GitHub si hay token, sino guarda local.
    """
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normales = total - ataques
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

    # Intentar usar GitHub si tenemos token
    github_token = st.secrets.get("GITHUB_TOKEN", None)
    if github_token:
        try:
            url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
            headers = {"Authorization": f"token {github_token}"}
            
            # Obtener archivo actual si existe
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                file_data = response.json()
                content = base64.b64decode(file_data['content']).decode('utf-8')
                df_existente = pd.read_csv(pd.compat.StringIO(content))
                df_nuevo = pd.concat([df_existente, nuevo_registro], ignore_index=True)
                sha = file_data['sha']
            else:
                df_nuevo = nuevo_registro
                sha = None
            
            # Subir nuevo contenido
            csv_content = df_nuevo.to_csv(index=False)
            encoded = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
            data = {
                "message": f"Agregar simulación {ahora}",
                "content": encoded,
                "branch": "main"
            }
            if sha:
                data["sha"] = sha
            
            put_resp = requests.put(url, headers=headers, json=data)
            if put_resp.status_code in [200, 201]:
                print("✅ Guardado en GitHub")
                return True
            else:
                print(f"❌ Error GitHub: {put_resp.text}")
        except Exception as e:
            print(f"❌ Excepción GitHub: {e}")
    
    # Fallback: guardar localmente
    try:
        if not os.path.isfile(archivo_hist):
            nuevo_registro.to_csv(archivo_hist, index=False)
        else:
            nuevo_registro.to_csv(archivo_hist, mode='a', header=False, index=False)
        return True
    except Exception as e:
        print(f"Error local: {e}")
        return False

def obtener_metricas_resumen(archivo_hist):
    """
    Obtiene datos desde GitHub (si hay token) o local.
    """
    github_token = st.secrets.get("GITHUB_TOKEN", None)
    if github_token:
        try:
            url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
            headers = {"Authorization": f"token {github_token}"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                file_data = response.json()
                content = base64.b64decode(file_data['content']).decode('utf-8')
                df = pd.read_csv(pd.compat.StringIO(content))
                if not df.empty:
                    df['Fecha'] = pd.to_datetime(df['Fecha'])
                    for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df.sort_values('Fecha', ascending=True)
        except Exception as e:
            print(f"Error lectura GitHub: {e}")
    
    # Fallback a local
    if os.path.exists(archivo_hist):
        try:
            df = pd.read_csv(archivo_hist)
            if not df.empty:
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df.sort_values('Fecha', ascending=True)
        except Exception as e:
            print(f"Error lectura local: {e}")
    return None
