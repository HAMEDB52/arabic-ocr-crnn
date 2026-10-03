"""تدريب نموذج CRNN على بيانات صناعية، مع تقييم CER/WER بعد كل تشغيل."""
from __future__ import annotations

import json
from pathlib import Path

from tensorflow import keras

from .data import make_datasets
from .metrics import corpus_metrics
from .model import build_model, ctc_loss
from .infer import predict_batch


def train(
    *,
    n_train: int = 3000,
    n_val: int = 400,
    epochs: int = 12,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_chars: int = 24,
    output_dir: str | Path = "artifacts",
    seed: int = 7,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, (x_val, y_val_txt) = make_datasets(
        n_train, n_val, batch_size=batch_size, seed=seed, max_chars=max_chars
    )
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=ctc_loss)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ],
    )

    predictions = predict_batch(model, x_val)
    metrics = corpus_metrics(y_val_txt, predictions)
    metrics["final_val_loss"] = float(min(history.history["val_loss"]))
    metrics["epochs_run"] = len(history.history["loss"])

    model.save(output_dir / "arabic_crnn.keras")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(
            [{"reference": r, "prediction": p} for r, p in zip(y_val_txt[:50], predictions[:50])],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return metrics
