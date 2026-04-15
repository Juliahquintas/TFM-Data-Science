import os
import shutil
import pandas as pd
from pathlib import Path

def organizar_audios():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    PROCESSED_DIR = DATA_DIR / 'metadata'
    OUTPUT_DIR = DATA_DIR / 'audios_ordenados'
    
    meta_nv_path = PROCESSED_DIR / 'metadata_neurovoz.csv'
    meta_pg_path = PROCESSED_DIR / 'metadata_pcgita.csv'
    
    if not meta_nv_path.exists() or not meta_pg_path.exists():
        print("Error: No se encuentran los archivos de metadatos.")
        print("Por favor, ejecuta primero el notebook de preprocesamiento para generarlos.")
        return
        
    df_nv = pd.read_csv(meta_nv_path)
    df_pg = pd.read_csv(meta_pg_path)
    
    # Mapeo de etiquetas a nombres de carpetas
    label_map = {
        'HC': 'Sano',
        'PD': 'Enfermo'
    }
    
    dataset_map = {
        'neurovoz': 'Neurovoz',
        'pc-gita': 'PC-GITA',
        'pcgita': 'PC-GITA'
    }
    
    if OUTPUT_DIR.exists():
        print(f"Limpiando directorio destino: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    dfs = [df_nv, df_pg]
    total_copiados = 0
    
    for df in dfs:
        for _, row in df.iterrows():
            dataset_name = dataset_map.get(row['dataset'].lower(), row['dataset'])
            estado = label_map.get(row['label'], row['label'])
            vocal = row['vocal']
            
            dest_folder = OUTPUT_DIR / dataset_name / estado / vocal
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            origen = Path(row['filepath'])
            destino = dest_folder / origen.name
            
            if origen.exists():
                shutil.copy2(origen, destino)
                total_copiados += 1
            else:
                print(f"Alerta: No se encontró el archivo: {origen}")
                
    print(f"Proceso completado. Se han organizado {total_copiados} audios en: {OUTPUT_DIR}")

if __name__ == "__main__":
    print("Iniciando organización de audios...")
    organizar_audios()
