"""
Script para reorganizar los audios de NeuroVoz siguiendo la estructura de PC-GITA.

Estructura actual de NeuroVoz:
    neurovoz/audios/HC_A1_0034.wav   (todo plano, en una sola carpeta)

Estructura objetivo (como PC-GITA):
    neurovoz/audios/Control/A/0034_A1.wav
    neurovoz/audios/Patologicas/A/0034_A1.wav

Mapeo:
    - HC  → Control
    - PD  → Patologicas
    - Se organiza por vocal (A, E, I, O, U)
    - Las tareas que NO son vocales se organizan en subcarpetas propias
      (PATAKA, ESPONTANEA, GANGA, etc.)
    - Nombre del archivo: {PatientID}_{Task}{Repetition}.wav
      Ejemplo: HC_A1_0034.wav  →  Control/A/0034_A1.wav
               PD_GANGA_0010.wav → Patologicas/GANGA/0010_GANGA.wav

IMPORTANTE: Este script COPIA los archivos (no los mueve), para no perder
los originales. Una vez verificado que todo está correcto, se puede borrar
la carpeta original de audios manualmente.

Uso:
    python scripts/reorganizar_neurovoz.py
    python scripts/reorganizar_neurovoz.py --dry-run   # Solo muestra qué haría
    python scripts/reorganizar_neurovoz.py --move       # Mover en vez de copiar
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Ruta base del proyecto (relativa a la ubicación de este script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Rutas de entrada y salida
NEUROVOZ_DIR = PROJECT_ROOT / "data" / "original" / "neurovoz"
AUDIOS_ORIGINALES = NEUROVOZ_DIR / "audios"
AUDIOS_REORGANIZADOS = NEUROVOZ_DIR / "audios_reorganizados"

# Mapeo de grupo → nombre de carpeta (igual que PC-GITA)
GROUP_MAP = {
    "HC": "Control",
    "PD": "Patologicas",
}

# Vocales que se agrupan por la letra inicial (como en PC-GITA)
# VOWELS = {"A", "E", "I", "O", "U"}
VOWELS = {"A", "E"}

# Patrón para parsear los nombres de archivo de NeuroVoz
# Ejemplos:
#   HC_A1_0034.wav          → group=HC, task=A1, patient_id=0034
#   PD_GANGA_0010.wav       → group=PD, task=GANGA, patient_id=0010
#   HC_PAN_VINO_0034.wav    → group=HC, task=PAN_VINO, patient_id=0034
#   HC_PATATA_BLANDA_0073.wav → group=HC, task=PATATA_BLANDA, patient_id=0073
#   PD_FREE_0023.wav        → group=PD, task=FREE, patient_id=0023
FILENAME_PATTERN = re.compile(
    r"^(HC|PD)_(.+)_(\d{4})\.wav$", re.IGNORECASE
)


def parse_filename(filename: str) -> dict | None:
    """
    Parsea un nombre de archivo de NeuroVoz y extrae sus componentes.

    Returns:
        dict con keys: group, task, patient_id, vowel_letter, repetition
        o None si no se puede parsear.
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None

    group = match.group(1).upper()       # HC o PD
    task = match.group(2).upper()        # A1, GANGA, PATATA_BLANDA, etc.
    patient_id = match.group(3)          # 0034

    # Determinar si es una vocal y extraer la letra y repetición
    # Patrones de vocals: A1, A2, A3, E1, E2, E3, I1, I2, I3, O1, O2, O3, U1, U2
    vowel_match = re.match(r"^([AEIOU])(\d+)$", task)
    if vowel_match:
        vowel_letter = vowel_match.group(1)  # A, E, I, O, U
        repetition = vowel_match.group(2)     # 1, 2, 3
        subfolder = vowel_letter
    else:
        vowel_letter = None
        repetition = None
        subfolder = task  # GANGA, PATAKA, ESPONTANEA, etc.

    return {
        "group": group,
        "task": task,
        "patient_id": patient_id,
        "vowel_letter": vowel_letter,
        "repetition": repetition,
        "subfolder": subfolder,
    }


def build_new_filename(parsed: dict) -> str:
    """
    Construye el nuevo nombre de archivo con el paciente primero.

    Ejemplos:
        HC_A1_0034.wav          →  0034_A1.wav
        PD_GANGA_0010.wav       →  0010_GANGA.wav
        HC_PAN_VINO_0034.wav    →  0034_PAN_VINO.wav
    """
    return f"{parsed['patient_id']}_{parsed['task']}.wav"


def build_new_path(parsed: dict) -> Path:
    """
    Construye la ruta completa del archivo reorganizado.

    Ejemplos:
        HC_A1_0034.wav → Control/A/0034_A1.wav
        PD_GANGA_0010.wav → Patologicas/GANGA/0010_GANGA.wav
    """
    group_folder = GROUP_MAP[parsed["group"]]
    subfolder = parsed["subfolder"]
    new_filename = build_new_filename(parsed)
    return Path(group_folder) / subfolder / new_filename


def reorganizar(dry_run: bool = False, move: bool = False):
    """
    Ejecuta la reorganización de audios.

    Args:
        dry_run: Si True, solo muestra qué haría sin hacer cambios.
        move: Si True, mueve los archivos en vez de copiarlos.
    """
    if not AUDIOS_ORIGINALES.exists():
        print(f"[ERROR] No se encuentra la carpeta de audios: {AUDIOS_ORIGINALES}")
        return

    # Obtener todos los .wav de la carpeta original
    wav_files = sorted(AUDIOS_ORIGINALES.glob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No se encontraron archivos .wav en {AUDIOS_ORIGINALES}")
        return

    print(f"[ORIGEN]  {AUDIOS_ORIGINALES}")
    print(f"[DESTINO] {AUDIOS_REORGANIZADOS}")
    print(f"Total archivos .wav encontrados: {len(wav_files)}")
    print(f"{'>>> MODO DRY-RUN (sin cambios)' if dry_run else '>>> Ejecutando reorganizacion...'}")
    if move:
        print("[!] Modo MOVER archivos (no copiar)")
    print("=" * 70)

    # Estadísticas
    stats = defaultdict(lambda: defaultdict(int))
    errores = []
    procesados = 0
    skipped = 0

    for wav_file in wav_files:
        filename = wav_file.name
        parsed = parse_filename(filename)

        if parsed is None:
            errores.append(f"No se pudo parsear: {filename}")
            skipped += 1
            continue

        # Solo quedarnos con las vocales definidas en VOWELS
        # Saltar palabras (GANGA, FREE, etc.) y vocales no incluidas en VOWELS
        if parsed["vowel_letter"] is None or parsed["vowel_letter"] not in VOWELS:
            skipped += 1
            continue

        # Construir la ruta destino
        relative_path = build_new_path(parsed)
        dest_path = AUDIOS_REORGANIZADOS / relative_path
        group_name = GROUP_MAP[parsed["group"]]

        # Actualizar estadísticas
        stats[group_name][parsed["subfolder"]] += 1

        if dry_run:
            print(f"  {filename:40s} -> {relative_path}")
        else:
            # Crear carpeta destino si no existe
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copiar o mover el archivo
            if move:
                shutil.move(str(wav_file), str(dest_path))
            else:
                shutil.copy2(str(wav_file), str(dest_path))

        procesados += 1

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE REORGANIZACION")
    print("=" * 70)

    for group_name in sorted(stats.keys()):
        print(f"\n  [{group_name}]/")
        subfolders = stats[group_name]
        for subfolder in sorted(subfolders.keys()):
            count = subfolders[subfolder]
            print(f"      {subfolder:20s} -> {count:4d} archivos")
        total_group = sum(subfolders.values())
        print(f"      {'-' * 30}")
        print(f"      {'TOTAL':20s} -> {total_group:4d} archivos")

    print(f"\n  [OK] Archivos procesados: {procesados}")
    if skipped:
        print(f"  [!]  Archivos omitidos:  {skipped}")

    if errores:
        print(f"\n  [ERROR] ERRORES ({len(errores)}):")
        for err in errores:
            print(f"      - {err}")

    if dry_run:
        print(f"\n[TIP] Ejecuta sin --dry-run para aplicar los cambios.")
    else:
        print(f"\n[OK] Reorganizacion completada en: {AUDIOS_REORGANIZADOS}")
        if not move:
            print(f"[TIP] Los archivos originales se mantienen en: {AUDIOS_ORIGINALES}")
            print(f"      Puedes eliminarlos manualmente una vez verificado.")


def actualizar_metadata(dry_run: bool = False):
    """
    Actualiza los CSVs de metadata:
    1. Filtra solo las filas de vocales (definidas en VOWELS)
    2. Actualiza la columna 'Audio' con la nueva ruta reorganizada
       Ejemplo: data/audios/PD_A1_0004.wav -> data/audios_reorganizados/Patologicas/A/0004_A1.wav
    3. Guarda los nuevos CSVs (los originales se mantienen)
    """
    import pandas as pd

    METADATA_DIR = NEUROVOZ_DIR / "metadata"
    csv_files = {
        "metadata_hc.csv": "metadata_hc_vocales.csv",
        "metadata_pd.csv": "metadata_pd_vocales.csv",
    }

    print("\n" + "=" * 70)
    print("ACTUALIZANDO METADATA CSVs")
    print("=" * 70)

    for original_name, new_name in csv_files.items():
        original_path = METADATA_DIR / original_name
        new_path = METADATA_DIR / new_name

        if not original_path.exists():
            print(f"  [!] No se encuentra: {original_path}")
            continue

        # Leer el CSV
        df = pd.read_csv(original_path)
        total_original = len(df)

        # Extraer el nombre del archivo de la columna Audio
        # Formato: data/audios/PD_A1_0004.wav
        def procesar_audio(audio_path):
            """Parsea y transforma la ruta del audio. Devuelve nueva ruta o None si no es vocal."""
            if pd.isna(audio_path):
                return None

            # Extraer solo el nombre del archivo
            filename = str(audio_path).replace("\\", "/").split("/")[-1]
            parsed = parse_filename(filename)

            if parsed is None:
                return None

            # Filtrar: solo vocales en VOWELS
            if parsed["vowel_letter"] is None or parsed["vowel_letter"] not in VOWELS:
                return None

            # Construir la nueva ruta
            relative_path = build_new_path(parsed)
            return f"data/audios_reorganizados/{relative_path}".replace("\\", "/")

        # Aplicar transformacion
        df["Audio_nuevo"] = df["Audio"].apply(procesar_audio)

        # Filtrar filas sin audio valido (no vocales)
        df_filtrado = df[df["Audio_nuevo"].notna()].copy()

        # Reemplazar la columna Audio con la nueva ruta
        df_filtrado["Audio"] = df_filtrado["Audio_nuevo"]
        df_filtrado = df_filtrado.drop(columns=["Audio_nuevo"])

        total_filtrado = len(df_filtrado)
        eliminados = total_original - total_filtrado

        if dry_run:
            print(f"\n  {original_name}:")
            print(f"    Filas originales:  {total_original}")
            print(f"    Filas con vocales: {total_filtrado}")
            print(f"    Filas eliminadas:  {eliminados}")
            print(f"    Se guardaria en:   {new_name}")
            # Mostrar ejemplos
            if len(df_filtrado) > 0:
                print(f"    Ejemplo Audio:")
                print(f"      Antes:   {df.iloc[0]['Audio']}")
                ejemplo = df_filtrado.iloc[0]["Audio"]
                print(f"      Despues: {ejemplo}")
        else:
            df_filtrado.to_csv(new_path, index=False)
            print(f"\n  {original_name} -> {new_name}")
            print(f"    Filas: {total_original} -> {total_filtrado} (eliminadas {eliminados} no-vocales)")

    if dry_run:
        print(f"\n[TIP] Ejecuta sin --dry-run para guardar los CSVs.")
    else:
        print(f"\n[OK] Metadata actualizada en: {METADATA_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganizar audios de NeuroVoz al estilo PC-GITA"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar que haria, sin hacer cambios reales"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Mover archivos en vez de copiarlos (cuidado!)"
    )

    args = parser.parse_args()
    reorganizar(dry_run=args.dry_run, move=args.move)
    actualizar_metadata(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

