"""عرض سريع: توليد عيّنات، ثم التعرّف عليها بالنموذج المدرَّب إن وُجد."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arabic_ocr.metrics import corpus_metrics  # noqa: E402
from arabic_ocr.synth import generate_dataset, save_samples  # noqa: E402

paths = save_samples(ROOT / "samples", 8)
print(f"تم توليد {len(paths)} عيّنة في مجلد samples/\n")

model_path = ROOT / "artifacts" / "arabic_crnn.keras"
if not model_path.exists():
    print("لا يوجد نموذج مدرَّب بعد. درّب أولاً:\n  python -m arabic_ocr.cli train --epochs 30")
    raise SystemExit(0)

from arabic_ocr.infer import load_model, predict_batch  # noqa: E402

images, references = generate_dataset(24, seed=999, max_chars=20)
predictions = predict_batch(load_model(model_path), images)

for ref, pred in list(zip(references, predictions))[:10]:
    mark = "✓" if ref == pred else "✗"
    print(f"{mark} المرجع: {ref:<28} التنبؤ: {pred}")

m = corpus_metrics(references, predictions)
print(f"\nCER {m['cer']:.3f} · WER {m['wer']:.3f} · مطابقة تامة {m['exact_match']:.1%}")
