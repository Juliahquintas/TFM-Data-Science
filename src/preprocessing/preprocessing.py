import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')


# =========================================================================
# RUTAS NECESARIAS PARA EL PREPROCESAMIENTO
# =========================================================================
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / 'data').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
INPUT_DIRS = [
    PROJECT_ROOT / 'data' / 'original' / 'neurovoz',
    PROJECT_ROOT / 'data' / 'original' / 'pc-gita'
]

OUTPUT_BASE_DIR = PROJECT_ROOT / 'data' / 'processed'




# =========================================================================
# CONFIGURACIÓN DE VALORES PARA EL PREPROCESAMIENTO
# =========================================================================
# 1. Silencios
REMOVE_SILENCE = True
SILENCE_TOP_DB = 30 # Umbral dB recomendado. (10 es agresivo, 25-30 estándar)

# 2. La Duración ahora ES DINÁMICA (No poner número, el sistema lo buscará solo)

# 3. Frecuencia de muestreo (Hz)
TARGET_SR = 22050




# =========================================================================
# LÓGICA MATEMÁTICA (FASE 2)
# =========================================================================
def procesar_audio(input_path, output_path, target_dur_sec):
    try:
        y, sr = librosa.load(input_path, sr=TARGET_SR)
        
        # 1. DETECCIÓN Y ELIMINACIÓN DE SILENCIOS (Solo actuando en extremos)
        if REMOVE_SILENCE:
            y_trimmed, index = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)
            if len(y_trimmed) < len(y):
                y = y_trimmed
        # 2. SELECCIÓN DE LA DURACIÓN EXACTA (Limitar al Mínimo que le pasemos)
        max_samples = int(target_dur_sec * TARGET_SR)
        
        if len(y) > max_samples:
            # Si el audio es más largo de lo admitido, se corta la cola (crop)
            y = y[:max_samples]
        elif len(y) < max_samples:
            # (Raro, pero just_in_case) Si falta algo, rellenar ceros
            padding = max_samples - len(y)
            y = np.pad(y, (0, padding), 'constant')
            
        # 3. NORMALIZACIÓN DE AMPLITUD (Min-Max a ±1.0)
        max_amp = np.max(np.abs(y))
        if max_amp > 0:
            y = y / max_amp
        # y = librosa.util.normalize(y, norm=np.inf)
            
        # 4. GUARDAR EN DISCO EL AUDIO PROCESADO
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, y, TARGET_SR)
        
        return {
            'success': True,
            'dur': len(y) / TARGET_SR,
            'amp_max': float(np.max(y)),
            'amp_min': float(np.min(y))
        }
        
    except Exception as e:
        print(f"[ERROR] {input_path.name}: {e}")
        return {'success': False}


# =========================================================================
# MOTOR PRINCIPAL DE DOS FASES
# =========================================================================
def main():
    print("-" * 60)
    print(f"  • Eliminar Silencios: {REMOVE_SILENCE} (Umbral: {SILENCE_TOP_DB}dB)")
    print(f"  • Duración Objetivo:  Cálculo Dinámico del Mínimo Global")
    print(f"  • Resampling a:       {TARGET_SR} Hz")
    print("-" * 60)
    
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Recolectar absolutamente todos los archivos de voz originales
    archivos_wav = []
    for d in INPUT_DIRS:
        archivos_wav.extend(list(d.rglob('*.wav')))
        
    print(f"\nSe van a procesar y estandarizar {len(archivos_wav)} audios...\n")
    
    from tqdm import tqdm # Importamos tqdm seguro para terminales y scripts
    
    # ----------------------------------------------------------------------
    # FASE 1: ESCANEAR MÍNIMOS Y SILENCIOS AL VUELO
    # ----------------------------------------------------------------------
    min_dur_cruda = float('inf')
    nombre_min_cruda = ""
    
    min_dur_trim = float('inf')
    nombre_min_trim = ""
    
    total_audios_recortados = 0
    
    for path in tqdm(archivos_wav, desc="Fase 1: Escaneando mínimos y silencios"):
        try:
            y_test, sr_test = librosa.load(path, sr=TARGET_SR)
            duracion_original = len(y_test) / sr_test
            
            # Guardamos cuál es el más corto antes de tocar nada
            if 0 < duracion_original < min_dur_cruda:
                min_dur_cruda = duracion_original
                nombre_min_cruda = path.name
                
            if REMOVE_SILENCE:
                y_trimmed_test, _ = librosa.effects.trim(y_test, top_db=SILENCE_TOP_DB)
                
                # Si se ha recortado algo, avisamos por pantalla con los segundos extirpados
                if len(y_trimmed_test) < len(y_test):
                    total_audios_recortados += 1
                    segundos_recortados = duracion_original - (len(y_trimmed_test) / sr_test)
                    tqdm.write(f"[SILENCIO ELIMINADO] {path.name} (se han extirpado {segundos_recortados:.3f} s)")

                y_test = y_trimmed_test # Aplicamos el trim para medir la duración final
            
            duracion_final = len(y_test) / sr_test
            
            # Guardamos cuál es el más corto DESPUÉS de quitar silencios
            if 0 < duracion_final < min_dur_trim:
                min_dur_trim = duracion_final
                nombre_min_trim = path.name
        except:
            continue
            
    print("\n" + "-" * 50)
    print(f" AUDITORÍA DE DURACIONES:")
    print(f"   ▶ Audios recortados por silencios mudos: {total_audios_recortados} de {len(archivos_wav)}")
    print(f"   ▶ Audio Original más corto:  {nombre_min_cruda} ({min_dur_cruda:.5f} s)")
    print(f"   ▶ Audio más corto (limpio):  {nombre_min_trim} ({min_dur_trim:.5f} s)")
    print("-" * 50)
    print("   → Todos los audios van a ser recortados o rellenados a esta última escala.\n")
    
    min_dur_global = min_dur_trim
    
    # ----------------------------------------------------------------------
    # FASE 2: PROCESAMIENTO MULTIPLE
    # ----------------------------------------------------------------------
    procesados_ok = 0
    stats = {
        'neurovoz': {'count': 0, 'durations': [], 'amp_max': [], 'amp_min': []},
        'pc-gita': {'count': 0, 'durations': [], 'amp_max': [], 'amp_min': []}
    }
    
    for path in tqdm(archivos_wav, desc="Fase 2: Procesando Dataset Definitivo "):
        
        # Mantener exactamente la misma estructura de sub-carpetas
        if "pc-gita" in str(path).lower():
            dataset_root = INPUT_DIRS[1]
            dataset_name = "pc-gita"
        else:
            dataset_root = INPUT_DIRS[0]
            dataset_name = "neurovoz"
            
        try:
            internal_path = path.relative_to(dataset_root)
            # Si la plataforma original tenía los audios dentro de una subcarpeta "audios", 
            # la saltamos para que Control y Patologicas queden directamente en la raíz del dataset
            if len(internal_path.parts) > 0 and internal_path.parts[0].lower() == 'audios':
                internal_path = Path(*internal_path.parts[1:])
        except ValueError:
            internal_path = Path(path.name)
            
        final_out_path = OUTPUT_BASE_DIR / dataset_name / internal_path
        
        # Le inyectamos la Duración Mínima calculada dinámicamente en la Fase 1
        info = procesar_audio(path, final_out_path, target_dur_sec=min_dur_global)
        
        if info['success']:
            procesados_ok += 1
            stats[dataset_name]['count'] += 1
            stats[dataset_name]['durations'].append(info['dur'])
            stats[dataset_name]['amp_max'].append(info['amp_max'])
            stats[dataset_name]['amp_min'].append(info['amp_min'])
            
    print("\n" + "=" * 60)
    print(f"¡PROCESO MASTERIZADO COMPLETADO! {procesados_ok} / {len(archivos_wav)} audios finales generados.")
    print(f"El nuevo dataset pulido y unificado en tiempos se encuentra en:")
    print(f"   {OUTPUT_BASE_DIR}")
    print("=" * 60)
    
    print("\n RESUMEN DEL DATASET GENERADO:")
    for ds_name in ['neurovoz', 'pc-gita']:
        s = stats[ds_name]
        c = s['count']
        if c > 0:
            dur_valid = s['durations']
            amp_max_valid = s['amp_max']
            amp_min_valid = s['amp_min']
            
            print(f"\n--- {ds_name.upper()} ---")
            print(f"  • Total Audios Finales: {c}")
            print(f"  • Frecuencia Estandar:  {TARGET_SR} Hz (Todos los audios exactos)")
            print(f"  • Duración (Segundos):  Min: {min(dur_valid):.5f} | Media: {sum(dur_valid)/c:.5f} | Max: {max(dur_valid):.5f}")
            print(f"  • Amplitud del pico:    Pico inferior: {min(amp_min_valid):.4f} | Pico superior: {max(amp_max_valid):.4f}")
    print("\n")
if __name__ == '__main__':
    main()