"""
Experiment Logger para TFM Parkinson
======================================
Guarda cada experimento (modelo, hiperparametros, metricas, fold)
en un CSV acumulativo y en un JSON individual para trazabilidad total.

Uso:
    logger = ExperimentLogger(log_dir="experiments/")
    logger.log(
        model_name   = "baseline_cnn1d",
        dataset      = "neurovoz",
        fold         = 1,
        hyperparams  = {"n_filters": [32,64,128], "dropout": 0.3, "lr": 1e-3},
        metrics      = {"accuracy": 0.87, "auc_roc": 0.91, ...},
        notes        = "Prueba con kernel_size=7",
    )
"""

import os
import json
import csv
import datetime


class ExperimentLogger:
    """
    Registra experimentos de forma acumulativa en:
      - experiments_log.csv  : tabla resumen (una fila por experimento)
      - <run_id>.json        : detalle completo del experimento
    """

    CSV_COLUMNS = [
        "run_id", "timestamp", "model_name", "dataset", "fold",
        "accuracy", "sensibilidad", "especificidad", "f1", "mcc", "auc_roc",
        "train_loss_final", "val_loss_final", "epochs_trained",
        "notes",
    ]

    def __init__(self, log_dir: str = "experiments"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "experiments_log.csv")
        self._init_csv()

    def _init_csv(self):
        """Crea el CSV con cabeceras si no existe."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()

    def log(
        self,
        model_name: str,
        dataset: str,
        fold: int,
        hyperparams: dict,
        metrics: dict,
        history: dict = None,
        notes: str = "",
    ) -> str:
        """
        Registra un experimento.

        Args:
            model_name  : Nombre del modelo (ej. "baseline_cnn1d").
            dataset     : Dataset usado ("neurovoz" / "pc-gita").
            fold        : Numero de fold (1..K).
            hyperparams : Diccionario con todos los hiperparametros.
            metrics     : Diccionario con metricas del test set
                          (accuracy, sensibilidad, especificidad, f1, mcc, auc_roc).
            history     : Historial de loss por epoch {train_loss:[...], val_loss:[...]}.
            notes       : Nota libre sobre la prueba.

        Returns:
            run_id : Identificador unico del experimento.
        """
        ts     = datetime.datetime.now()
        run_id = ts.strftime("%Y%m%d_%H%M%S") + f"_{model_name}_fold{fold}"

        # ── Fila para el CSV ──────────────────
        row = {col: "" for col in self.CSV_COLUMNS}
        row["run_id"]          = run_id
        row["timestamp"]       = ts.strftime("%Y-%m-%d %H:%M:%S")
        row["model_name"]      = model_name
        row["dataset"]         = dataset
        row["fold"]            = fold
        row["notes"]           = notes

        for metric_key in ["accuracy", "sensibilidad", "especificidad", "f1", "mcc", "auc_roc"]:
            row[metric_key] = round(metrics.get(metric_key, float("nan")), 4)

        if history:
            train_losses = history.get("train_loss", [])
            val_losses   = history.get("val_loss",   [])
            row["train_loss_final"] = round(train_losses[-1], 4) if train_losses else ""
            row["val_loss_final"]   = round(val_losses[-1],   4) if val_losses   else ""
            row["epochs_trained"]   = len(train_losses)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writerow(row)

        # ── JSON detallado ────────────────────
        detail = {
            "run_id":      run_id,
            "timestamp":   ts.isoformat(),
            "model_name":  model_name,
            "dataset":     dataset,
            "fold":        fold,
            "hyperparams": hyperparams,
            "metrics":     metrics,
            "history":     history or {},
            "notes":       notes,
        }
        json_path = os.path.join(self.log_dir, f"{run_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2, ensure_ascii=False)

        print(f"[Logger] Experimento guardado: {run_id}")
        print(f"  CSV  → {self.csv_path}")
        print(f"  JSON → {json_path}")
        return run_id

    def print_summary(self):
        """Imprime en pantalla el resumen de todos los experimentos registrados."""
        if not os.path.exists(self.csv_path):
            print("[Logger] No hay experimentos registrados todavia.")
            return

        print(f"\n{'='*80}")
        print(f" RESUMEN DE EXPERIMENTOS  ({self.csv_path})")
        print(f"{'='*80}")
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print("  (vacio)")
            return

        header = f"{'run_id':<45} {'dataset':<10} {'fold':<5} {'acc':>6} {'auc':>6} {'f1':>6}"
        print(header)
        print("-" * 80)
        for r in rows:
            print(
                f"{r['run_id']:<45} {r['dataset']:<10} {r['fold']:<5} "
                f"{r['accuracy']:>6} {r['auc_roc']:>6} {r['f1']:>6}"
            )
        print()
