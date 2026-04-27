import os
import random
import json
from pathlib import Path
from collections import defaultdict

# Fijar semilla para que los K-Folds sean siempre los mismos en el futuro
random.seed(42)

def get_subject_id_from_filename(filename, dataset_name):
    """
    Extrae el ID del paciente a partir del nombre del archivo.
    Neurovoz: 0034_A1.wav -> 0034
    PC-GITA: AVPEPUDEAC0001a1.wav -> AVPEPUDEAC0001
    """
    basename = Path(filename).stem
    if dataset_name == 'neurovoz':
        return basename.split('_')[0]
    elif dataset_name == 'pc-gita':
        import re
        match = re.match(r'(AVPE[a-zA-Z0-9]{10})', basename, re.IGNORECASE)
        if match:
             return match.group(0)
        if '_' in basename:
            return basename.split('_')[0]
        else:
            return basename[:-2]
    return basename

def scan_dataset(base_dir):
    """
    Agrupa todos los paths de audios por: Clase -> Sujeto -> [rutas_archivos]
    """
    dataset_dict = defaultdict(lambda: defaultdict(list))
    
    for class_name in ['Control', 'Patologicas']:
        class_dir = base_dir / class_name
        if not class_dir.exists():
            continue
            
        for filepath in class_dir.rglob('*.wav'):
            rel_path = filepath.relative_to(base_dir.parent)
            subj_id = get_subject_id_from_filename(filepath.name, base_dir.name)
            dataset_dict[class_name][subj_id].append(str(rel_path).replace('\\\\', '/')) # Normalizamos rutas a formato estandar
            
    return dataset_dict

def create_k_chunks(subj_list, K):
    """ Divide una lista de sujetos en K partes lo más iguales posible """
    n = len(subj_list)
    # Ej: Si n=50 y K=5 -> [0:10], [10:20]...
    return [subj_list[i * n // K: (i + 1) * n // K] for i in range(K)]

def generate_k_folds_from_chunks(chunks, K):
    """
    Toma K pedazos (chunks) y genera K distribuciones.
    Para el Fold 'i':
     - TEST = chunk[i]
     - VALID = chunk[i+1] (circular)
     - TRAIN = el resto de chunks
    Como K=5, esto deja 1 de test (20%), 1 de valid (20%), y 3 de train (60%).
    """
    folds = []
    for i in range(K):
        test_chunk = chunks[i]
        val_chunk = chunks[(i + 1) % K]
        train_chunk = []
        for j in range(K):
            if j != i and j != (i + 1) % K:
                train_chunk.extend(chunks[j])
        folds.append({
            'train_subjs': train_chunk,
            'val_subjs': val_chunk,
            'test_subjs': test_chunk
        })
    return folds

def build_final_folds_structure(K):
    # Genera la estructura vacía donde iremos metiendo las rutas
    return [{
        'train_subjs': [], 'val_subjs': [], 'test_subjs': [],
        'train_subjs_hc': [], 'val_subjs_hc': [], 'test_subjs_hc': [],
        'train_subjs_pa': [], 'val_subjs_pa': [], 'test_subjs_pa': [],
        'train_files': [], 'val_files': [], 'test_files': []
    } for _ in range(K)]

def split_pc_gita_kfold(dataset_dict, K=5):
    """ División clásica K-Fold para PC-GITA (que tiene repeticiones fijas). """
    final_folds = build_final_folds_structure(K)
    
    for class_name, subjects in dataset_dict.items():
        # Extrar sujetos (50 sanos y luego 50 enf.)
        subj_list = list(subjects.keys())
        random.shuffle(subj_list)
        
        # Despedazar en K=5 (10 sujetos por bloque)
        chunks = create_k_chunks(subj_list, K)
        # Asignar qué bloques van a train/val/test según el Fold
        class_folds = generate_k_folds_from_chunks(chunks, K)
        
        # Volcar sujetos y audios en el objeto final
        for k in range(K):
            final_folds[k]['train_subjs'].extend(class_folds[k]['train_subjs'])
            final_folds[k]['val_subjs'].extend(class_folds[k]['val_subjs'])
            final_folds[k]['test_subjs'].extend(class_folds[k]['test_subjs'])
            
            if class_name == 'Control':
                final_folds[k]['train_subjs_hc'].extend(class_folds[k]['train_subjs'])
                final_folds[k]['val_subjs_hc'].extend(class_folds[k]['val_subjs'])
                final_folds[k]['test_subjs_hc'].extend(class_folds[k]['test_subjs'])
            else:
                final_folds[k]['train_subjs_pa'].extend(class_folds[k]['train_subjs'])
                final_folds[k]['val_subjs_pa'].extend(class_folds[k]['val_subjs'])
                final_folds[k]['test_subjs_pa'].extend(class_folds[k]['test_subjs'])
            
            for s in class_folds[k]['train_subjs']: final_folds[k]['train_files'].extend(subjects[s])
            for s in class_folds[k]['val_subjs']:   final_folds[k]['val_files'].extend(subjects[s])
            for s in class_folds[k]['test_subjs']:  final_folds[k]['test_files'].extend(subjects[s])
            
    return final_folds

def split_neurovoz_kfold(dataset_dict, K=5, num_iterations=2500):
    """ División variada para Neurovoz. Itera miles de veces hasta hallar la partición
        donde los K chunks tienen casi exactamente la misma cantidad de audios """
    final_folds = build_final_folds_structure(K)
    
    for class_name, subjects in dataset_dict.items():
        subj_list = list(subjects.keys())
        total_audios = sum(len(audios) for audios in subjects.values())
        target_audios = total_audios / K  # Ej: si hay 150 audios, lo ideal son 30 por chunk
        
        best_chunks = None
        best_error = float('inf')
        
        # Buscar hiper-aleatoriamente la distribución ideal
        for _ in range(num_iterations):
            random.shuffle(subj_list)
            chunks = create_k_chunks(subj_list, K)
            
            # Penalización: Qué tan lejos nos quedamos del target_audios en cada pedazo
            error = sum((sum(len(subjects[s]) for s in chunk) - target_audios)**2 for chunk in chunks)
                
            if error < best_error:
                best_error = error
                best_chunks = chunks
                
        class_folds = generate_k_folds_from_chunks(best_chunks, K)
        
        # Volcar
        for k in range(K):
            final_folds[k]['train_subjs'].extend(class_folds[k]['train_subjs'])
            final_folds[k]['val_subjs'].extend(class_folds[k]['val_subjs'])
            final_folds[k]['test_subjs'].extend(class_folds[k]['test_subjs'])
            
            if class_name == 'Control':
                final_folds[k]['train_subjs_hc'].extend(class_folds[k]['train_subjs'])
                final_folds[k]['val_subjs_hc'].extend(class_folds[k]['val_subjs'])
                final_folds[k]['test_subjs_hc'].extend(class_folds[k]['test_subjs'])
            else:
                final_folds[k]['train_subjs_pa'].extend(class_folds[k]['train_subjs'])
                final_folds[k]['val_subjs_pa'].extend(class_folds[k]['val_subjs'])
                final_folds[k]['test_subjs_pa'].extend(class_folds[k]['test_subjs'])
            
            for s in class_folds[k]['train_subjs']: final_folds[k]['train_files'].extend(subjects[s])
            for s in class_folds[k]['val_subjs']:   final_folds[k]['val_files'].extend(subjects[s])
            for s in class_folds[k]['test_subjs']:  final_folds[k]['test_files'].extend(subjects[s])
            
    return final_folds

def main():
    # ===============
    # PARAMETRO CLAVE
    # ===============
    K = 5  # <--- Puedes cambiar esto por 3, 10, etc., en el futuro.
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
    
    neurovoz_dir = PROCESSED_DIR / 'neurovoz'
    pc_gita_dir = PROCESSED_DIR / 'pc-gita'
    
    print(f"Analizando datasets...")
    nv_data = scan_dataset(neurovoz_dir)
    pc_data = scan_dataset(pc_gita_dir)
    
    print(f"Generando particiones subject-wise para {K}-Folds Cross Validation...")
    nv_splits = split_neurovoz_kfold(nv_data, K=K)
    pc_splits = split_pc_gita_kfold(pc_data, K=K)
    
    final_splits = {
        'metadata': {'K_folds': K},
        'neurovoz': nv_splits,
        'pc-gita': pc_splits
    }
    
    output_file = PROJECT_ROOT / 'data' / 'data_splits.json'
    with open(output_file, 'w') as f:
        json.dump(final_splits, f, indent=4)
        
    print(f"\\nParticiones {K}-Fold guardadas exitosamente en: {output_file}")
    
    # ------------------
    # MOSTRAR ESTADÍSTICAS
    # ------------------
    for ds_name, folds in [('neurovoz', nv_splits), ('pc-gita', pc_splits)]:
        print(f"\\n{'='*45}")
        print(f" REPORTE DE {K}-FOLDS PARA: {ds_name.upper()}")
        print(f"{'='*45}")
        
        for i, fold_dict in enumerate(folds):
            print(f"\\n> FOLD {i + 1}")
            
            # Pacientes
            n_t_subj = len(fold_dict['train_subjs'])
            n_v_subj = len(fold_dict['val_subjs'])
            n_te_subj = len(fold_dict['test_subjs'])
            
            hc_t = len(fold_dict['train_subjs_hc']); pa_t = len(fold_dict['train_subjs_pa'])
            hc_v = len(fold_dict['val_subjs_hc']); pa_v = len(fold_dict['val_subjs_pa'])
            hc_te = len(fold_dict['test_subjs_hc']); pa_te = len(fold_dict['test_subjs_pa'])
            
            # Audios
            n_t_audios = len(fold_dict['train_files'])
            n_v_audios = len(fold_dict['val_files'])
            n_te_audios = len(fold_dict['test_files'])
            
            total_audios = n_t_audios + n_v_audios + n_te_audios
            
            print(f"  • TRAIN: {n_t_audios:3} audios ({n_t_audios/total_audios*100:.1f}%) | {n_t_subj:2} pacientes ({hc_t:2} Sanos, {pa_t:2} Patológicos)")
            print(f"  • VALID: {n_v_audios:3} audios ({n_v_audios/total_audios*100:.1f}%) | {n_v_subj:2} pacientes ({hc_v:2} Sanos, {pa_v:2} Patológicos)")
            print(f"  • TEST:  {n_te_audios:3} audios ({n_te_audios/total_audios*100:.1f}%) | {n_te_subj:2} pacientes ({hc_te:2} Sanos, {pa_te:2} Patológicos)")

if __name__ == '__main__':
    main()
