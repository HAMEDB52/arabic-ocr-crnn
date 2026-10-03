"""مولّد بيانات صناعية لأسطر نصية عربية.

يستخدم تخطيط النص المعقّد في Pillow (raqm) ليصل الحروف ويعكس الاتجاه صحيحاً —
وهي النقطة التي تفسد فيها معظم مولّدات البيانات الجاهزة مع العربية.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .charset import is_supported

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"

# مفردات مستمدة من لغة المستندات الإدارية والفواتير
LEXICON = [
    "فاتورة", "ضريبية", "الرقم", "التاريخ", "المبلغ", "الإجمالي", "الخصم",
    "العميل", "المورد", "الكمية", "الوحدة", "السعر", "الضريبة", "ريال",
    "اسم", "العنوان", "الهاتف", "البريد", "السجل", "التجاري", "الشركة",
    "مؤسسة", "بند", "وصف", "الدفع", "نقدا", "شبكة", "استلام", "تسليم",
    "رقم", "الحساب", "المرجع", "الصفحة", "التوقيع", "الختم", "الموظف",
]
WORD_PATTERN = ["{w}", "{w} {w}", "{w} {w} {w}", "{w} {n}", "{w} {w} {n}", "{n}"]

IMG_HEIGHT = 32
IMG_WIDTH = 256


def _fonts(sizes=(18, 20, 22, 24)) -> list[tuple[str, int]]:
    paths = sorted(str(p) for p in FONT_DIR.glob("*.ttf"))
    if not paths:
        raise FileNotFoundError(f"لا توجد خطوط في {FONT_DIR}")
    return [(p, s) for p in paths for s in sizes]


def random_text(rng: random.Random, max_chars: int = 24) -> str:
    """توليد نص عشوائي من المفردات والأرقام ضمن حد أقصى للطول."""
    for _ in range(12):
        pattern = rng.choice(WORD_PATTERN)
        text = pattern
        while "{w}" in text:
            text = text.replace("{w}", rng.choice(LEXICON), 1)
        while "{n}" in text:
            text = text.replace("{n}", str(rng.randint(1, 99999)), 1)
        text = text.replace("أ", "ا").replace("إ", "ا")  # تبسيط مقصود للهمزات
        if len(text) <= max_chars and is_supported(text):
            return text
    return "فاتورة"


def render(text: str, *, font_path: str, font_size: int, rng: random.Random | None = None,
           width: int = IMG_WIDTH, height: int = IMG_HEIGHT) -> np.ndarray:
    """رسم سطر نصي عربي كصورة رمادية مُطبَّعة في المدى [0, 1]."""
    rng = rng or random.Random()
    canvas = Image.new("L", (width * 2, height * 2), color=255)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path, font_size)
    draw.text((width * 2 - 8, height), text, font=font, fill=0,
              anchor="rm", direction="rtl", language="ar")

    bbox = canvas.getbbox()                       # قص الهوامش الفارغة
    if bbox:
        canvas = canvas.crop(bbox)
    canvas = _fit(canvas, width, height)

    if rng.random() < 0.5:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    if rng.random() < 0.5:
        arr = np.clip(arr + np.random.normal(0, rng.uniform(0.01, 0.05), arr.shape), 0, 1)
    return arr.astype(np.float32)


def _fit(img: Image.Image, width: int, height: int) -> Image.Image:
    """تحجيم مع حفظ النسبة ثم لصق على خلفية بيضاء بعرض ثابت."""
    scale = min(width / max(img.width, 1), height / max(img.height, 1))
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    out = Image.new("L", (width, height), color=255)
    out.paste(img, (width - img.width, (height - img.height) // 2))   # محاذاة لليمين
    return out


def generate_dataset(n: int, *, seed: int = 7, max_chars: int = 24
                     ) -> tuple[np.ndarray, list[str]]:
    """إنتاج (صور، نصوص) جاهزة للتدريب أو التقييم."""
    rng = random.Random(seed)
    np.random.seed(seed)
    combos = _fonts()
    images, texts = [], []
    for _ in range(n):
        text = random_text(rng, max_chars)
        font_path, font_size = rng.choice(combos)
        images.append(render(text, font_path=font_path, font_size=font_size, rng=rng))
        texts.append(text)
    return np.stack(images)[..., None], texts


def save_samples(directory: str | Path, n: int = 8, seed: int = 11) -> list[Path]:
    """حفظ عيّنات للفحص البصري — التحقق من صحة التشكيل البصري قبل التدريب."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    images, texts = generate_dataset(n, seed=seed)
    paths = []
    for i, (img, text) in enumerate(zip(images, texts)):
        path = directory / f"sample_{i:02d}.png"
        Image.fromarray((img[..., 0] * 255).astype("uint8")).save(path)
        (directory / f"sample_{i:02d}.txt").write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
