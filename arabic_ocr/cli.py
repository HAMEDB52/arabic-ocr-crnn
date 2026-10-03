"""arabic-ocr — تدريب النموذج، التعرّف على صورة، وتوليد عيّنات."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arabic-ocr", description="تعرّف ضوئي عربي بـ CNN-LSTM + CTC")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="تدريب النموذج على بيانات صناعية")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--train-size", type=int, default=3000)
    t.add_argument("--val-size", type=int, default=400)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--out", default="artifacts")

    r = sub.add_parser("predict", help="التعرّف على صورة سطر نصي")
    r.add_argument("image")
    r.add_argument("--model", default="artifacts/arabic_crnn.keras")

    s = sub.add_parser("samples", help="توليد عيّنات صور للفحص البصري")
    s.add_argument("--out", default="samples")
    s.add_argument("--count", type=int, default=8)

    args = p.parse_args(argv)

    if args.command == "train":
        from .train import train

        metrics = train(n_train=args.train_size, n_val=args.val_size,
                        epochs=args.epochs, batch_size=args.batch_size, output_dir=args.out)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    if args.command == "predict":
        from .infer import load_model, predict_image

        if not Path(args.model).exists():
            print(f"النموذج غير موجود: {args.model} — درّب أولاً عبر: arabic-ocr train")
            return 1
        print(predict_image(load_model(args.model), args.image))
        return 0

    from .synth import save_samples

    paths = save_samples(args.out, args.count)
    print(f"تم حفظ {len(paths)} عيّنة في {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
