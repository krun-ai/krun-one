"""M1 synthetic media: app screenshots of a user message, and rendered business documents (EN / PT / ES / FR).

Everything is generated from (seed, inputs) only — no external images — with DejaVu / Noto fonts (permissive
licenses, shipped in data/m1/fonts). Each document returns its reading-order TEXT (the text fast-path equivalent,
used by the frozen text teacher) and its ground-truth fields.
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path(__file__).resolve().parents[4] / "data/m1/fonts"
SANS = ["DejaVuSans.ttf", "NotoSans-Regular.ttf", "DejaVuSansCondensed.ttf"]
BOLD = ["DejaVuSans-Bold.ttf", "NotoSans-Bold.ttf", "DejaVuSerif-Bold.ttf"]
SERIF = ["DejaVuSerif.ttf", "NotoSerif-Regular.ttf"]
MONO = ["DejaVuSansMono.ttf"]
_FONTS: dict = {}
LOCALES = {"en": "en_US", "pt": "pt_BR", "es": "es_ES", "fr": "fr_FR"}
CURRENCY = {"en": "$", "pt": "R$", "es": "€", "fr": "€"}
_FAKERS: dict = {}


def font(name: str, size: int):
    k = (name, size)
    if k not in _FONTS:
        _FONTS[k] = ImageFont.truetype(str(FONT_DIR / name), size)
    return _FONTS[k]


def faker(lang: str, seed: int):
    from faker import Faker

    if lang not in _FAKERS:
        _FAKERS[lang] = Faker(LOCALES[lang])
    f = _FAKERS[lang]
    f.seed_instance(seed)
    return f


def _rgb(rng, lo=0, hi=255):
    return tuple(rng.randint(lo, hi) for _ in range(3))


# ------------------------------------------------------------------------------------------------ screenshots
APP_TEXT = {"en": ("Messages", "Support chat", "Type a message", "Search", "New ticket", "Describe your request", "Send"),
            "pt": ("Mensagens", "Chat de suporte", "Digite uma mensagem", "Buscar", "Novo chamado", "Descreva sua solicitação", "Enviar"),
            "es": ("Mensajes", "Chat de soporte", "Escribe un mensaje", "Buscar", "Nuevo ticket", "Describe tu solicitud", "Enviar"),
            "fr": ("Messages", "Chat d'assistance", "Écrire un message", "Rechercher", "Nouveau ticket", "Décrivez votre demande", "Envoyer")}


def render_screenshot(text: str, lang: str, seed: int) -> Image.Image:
    """A phone/desktop app screen whose main content is the user's message (chat bubble, ticket form or search)."""
    rng = random.Random(seed)
    L = APP_TEXT.get(lang, APP_TEXT["en"])
    kind = rng.choice(["chat", "chat", "ticket", "search"])
    W = rng.choice([420, 480, 540, 640])
    H = int(W * rng.uniform(1.1, 1.8))
    dark = rng.random() < 0.3
    bg = (24, 24, 28) if dark else (rng.randint(235, 255),) * 3
    fg = (235, 235, 235) if dark else (20, 20, 20)
    accent = _rgb(rng, 30, 200)
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    fs = rng.randint(22, 30)
    fnt = font(rng.choice(SANS), fs)
    d.rectangle([0, 0, W, 64], fill=accent)
    d.text((20, 18), L[1] if kind == "chat" else (L[4] if kind == "ticket" else L[3]), font=font(BOLD[0], 24), fill=(255, 255, 255))
    wrap = max(12, int((W - 90) / (fs * 0.55)))
    lines = textwrap.wrap(text, wrap) or [text]
    y = 100
    if kind == "chat":
        for _ in range(rng.randint(0, 2)):          # earlier greeting bubbles (not the decision content)
            g = rng.choice({"en": ["Hi!", "Hello"], "pt": ["Oi!", "Olá"], "es": ["¡Hola!", "Buenas"], "fr": ["Bonjour", "Salut"]}[lang])
            d.rounded_rectangle([20, y, 20 + 30 + int(len(g) * fs * 0.6), y + fs + 24], 14, fill=(60, 60, 66) if dark else (225, 225, 230))
            d.text((34, y + 10), g, font=fnt, fill=fg)
            y += fs + 44
        bh = len(lines) * (fs + 8) + 24
        x0 = W - 40 - max(int(len(l) * fs * 0.56) for l in lines) - 28
        x0 = max(40, x0)
        d.rounded_rectangle([x0, y, W - 20, y + bh], 18, fill=accent)
        for i, l in enumerate(lines):
            d.text((x0 + 14, y + 12 + i * (fs + 8)), l, font=fnt, fill=(255, 255, 255))
        d.rectangle([0, H - 70, W, H], fill=(40, 40, 46) if dark else (245, 245, 248))
        d.text((24, H - 50), L[2], font=font(SANS[0], 20), fill=(140, 140, 140))
    elif kind == "ticket":
        d.text((24, y), L[5], font=font(BOLD[0], 20), fill=fg)
        y += 36
        bh = len(lines) * (fs + 8) + 30
        d.rectangle([20, y, W - 20, y + bh], outline=(150, 150, 150), width=2)
        for i, l in enumerate(lines):
            d.text((34, y + 14 + i * (fs + 8)), l, font=fnt, fill=fg)
        y += bh + 30
        d.rounded_rectangle([W - 180, y, W - 20, y + 50], 10, fill=accent)
        d.text((W - 150, y + 12), L[6], font=font(BOLD[0], 22), fill=(255, 255, 255))
    else:
        d.rounded_rectangle([20, y, W - 20, y + len(lines) * (fs + 8) + 28], 24, outline=accent, width=3)
        for i, l in enumerate(lines):
            d.text((44, y + 14 + i * (fs + 8)), l, font=fnt, fill=fg)
    if rng.random() < 0.3:                          # phone-photo of a screen: blur / small rotation
        im = im.rotate(rng.uniform(-3, 3), expand=True, fillcolor=(90, 90, 90)).filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    return im


# ------------------------------------------------------------------------------------------------ documents
DOC_TEXT = {
    "en": {"date": "Date", "due": "Due date", "total": "TOTAL", "qty": "Qty", "item": "Description", "price": "Price",
           "tax": "Tax ID", "bank": "Bank details", "paid": "PAID", "sign": "Signature", "to": "Bill to", "no": "No."},
    "pt": {"date": "Data", "due": "Vencimento", "total": "TOTAL", "qty": "Qtd", "item": "Descrição", "price": "Preço",
           "tax": "CNPJ", "bank": "Dados bancários", "paid": "PAGO", "sign": "Assinatura", "to": "Cliente", "no": "Nº"},
    "es": {"date": "Fecha", "due": "Vencimiento", "total": "TOTAL", "qty": "Cant.", "item": "Descripción", "price": "Precio",
           "tax": "NIF", "bank": "Datos bancarios", "paid": "PAGADO", "sign": "Firma", "to": "Cliente", "no": "N.º"},
    "fr": {"date": "Date", "due": "Échéance", "total": "TOTAL", "qty": "Qté", "item": "Désignation", "price": "Prix",
           "tax": "N° TVA", "bank": "Coordonnées bancaires", "paid": "PAYÉ", "sign": "Signature", "to": "Client", "no": "N°"}}
# which optional fields each doc type may show (probability), item table and amounts
DOC_PROFILE = {
    "invoice": {"item_table": .95, "due_date": .8, "tax_id": .8, "bank_details": .6, "signature": .2, "barcode": .2, "logo": .7, "paid_stamp": .35},
    "receipt": {"item_table": .9, "due_date": 0, "tax_id": .5, "bank_details": 0, "signature": 0, "barcode": .5, "logo": .5, "paid_stamp": .3},
    "purchase_order": {"item_table": .95, "due_date": .5, "tax_id": .6, "bank_details": .1, "signature": .6, "barcode": .1, "logo": .7, "paid_stamp": 0},
    "bank_statement": {"item_table": .95, "due_date": 0, "tax_id": .1, "bank_details": .9, "signature": 0, "barcode": 0, "logo": .9, "paid_stamp": 0},
    "payslip": {"item_table": .9, "due_date": 0, "tax_id": .8, "bank_details": .6, "signature": .3, "barcode": 0, "logo": .6, "paid_stamp": .1},
    "shipping_label": {"item_table": 0, "due_date": 0, "tax_id": .1, "bank_details": 0, "signature": 0, "barcode": .95, "logo": .5, "paid_stamp": 0},
    "medical_prescription": {"item_table": .4, "due_date": 0, "tax_id": 0, "bank_details": 0, "signature": .95, "barcode": .2, "logo": .4, "paid_stamp": 0},
    "rental_agreement": {"item_table": 0, "due_date": .5, "tax_id": .3, "bank_details": .4, "signature": .9, "barcode": 0, "logo": .2, "paid_stamp": 0},
    "job_application_form": {"item_table": 0, "due_date": 0, "tax_id": .2, "bank_details": .1, "signature": .7, "barcode": 0, "logo": .5, "paid_stamp": 0},
    "delivery_note": {"item_table": .95, "due_date": 0, "tax_id": .4, "bank_details": 0, "signature": .7, "barcode": .5, "logo": .6, "paid_stamp": 0},
    "utility_bill": {"item_table": .7, "due_date": .95, "tax_id": .5, "bank_details": .5, "signature": 0, "barcode": .8, "logo": .9, "paid_stamp": .2},
    "boarding_pass": {"item_table": 0, "due_date": 0, "tax_id": 0, "bank_details": 0, "signature": 0, "barcode": .95, "logo": .8, "paid_stamp": 0},
}
HAS_TOTAL = {"invoice", "receipt", "purchase_order", "bank_statement", "payslip", "utility_bill", "delivery_note"}


def _money(v: float, lang: str) -> str:
    s = f"{v:,.2f}"
    if lang in ("pt", "es", "fr"):
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{CURRENCY[lang]} {s}"


def render_document(doc_type: str, title: str, lang: str, seed: int) -> tuple[Image.Image, str, dict]:
    """-> (page image, reading-order text, truth {fields: {..: bool}, total: float|None, paid: bool, signed: bool})."""
    rng = random.Random(seed)
    fk = faker(lang, seed)
    T = DOC_TEXT[lang]
    prof = DOC_PROFILE[doc_type]
    fields = {k: rng.random() < p for k, p in prof.items()}
    W, H = 850, 1100
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    serif = rng.random() < 0.3
    body = font(rng.choice(SERIF if serif else SANS), rng.randint(18, 22))
    small = font(SANS[0], 16)
    big = font(rng.choice(BOLD), rng.randint(40, 52))
    text: list[str] = []
    y = 40
    company = fk.company()
    if fields["logo"]:
        c = _rgb(rng, 20, 200)
        (d.ellipse if rng.random() < .5 else d.rectangle)([40, y, 120, y + 80], fill=c)
        x_t = 140
    else:
        x_t = 40
    d.text((x_t, y), company, font=font(BOLD[0], 26), fill=(30, 30, 30))
    text.append(company)
    addr = fk.address().replace("\n", ", ")
    d.text((x_t, y + 38), addr[:60], font=small, fill=(80, 80, 80))
    text.append(addr[:60])
    y += 110
    d.text((40, y), title.upper(), font=big, fill=(20, 20, 20))
    text.append(title.upper())
    y += 80
    num = f"{T['no']} {rng.randint(1000, 999999)}"
    date = fk.date_this_decade().strftime("%d/%m/%Y" if lang != "en" else "%m/%d/%Y")
    d.text((40, y), f"{num}    {T['date']}: {date}", font=body, fill=(30, 30, 30))
    text.append(f"{num} {T['date']}: {date}")
    y += 40
    if fields["due_date"]:
        due = fk.future_date().strftime("%d/%m/%Y" if lang != "en" else "%m/%d/%Y")
        d.text((40, y), f"{T['due']}: {due}", font=body, fill=(160, 20, 20))
        text.append(f"{T['due']}: {due}")
        y += 40
    if fields["tax_id"]:
        tid = f"{T['tax']}: {rng.randint(10, 99)}.{rng.randint(100, 999)}.{rng.randint(100, 999)}/{rng.randint(1000, 9999)}"
        d.text((40, y), tid, font=body, fill=(30, 30, 30))
        text.append(tid)
        y += 40
    person = fk.name()
    d.text((40, y), f"{T['to']}: {person}", font=body, fill=(30, 30, 30))
    text.append(f"{T['to']}: {person}")
    y += 60
    total = None
    if fields["item_table"]:
        n = rng.randint(2, 7)
        scale = rng.choice([5, 20, 80, 300, 1500])
        d.rectangle([40, y, W - 40, y + 36], fill=(230, 230, 235))
        d.text((50, y + 6), T["item"], font=body, fill=(0, 0, 0))
        d.text((W - 300, y + 6), T["qty"], font=body, fill=(0, 0, 0))
        d.text((W - 180, y + 6), T["price"], font=body, fill=(0, 0, 0))
        text.append(f"{T['item']} | {T['qty']} | {T['price']}")
        y += 44
        total = 0.0
        for _ in range(n):
            it = fk.word().capitalize() + " " + fk.word()
            q = rng.randint(1, 5)
            p = round(rng.uniform(0.2, 1.0) * scale, 2)
            total += q * p
            d.text((50, y), it[:30], font=body, fill=(30, 30, 30))
            d.text((W - 290, y), str(q), font=body, fill=(30, 30, 30))
            d.text((W - 180, y), _money(p, lang), font=body, fill=(30, 30, 30))
            text.append(f"{it[:30]} | {q} | {_money(p, lang)}")
            y += 34
        d.line([40, y + 6, W - 40, y + 6], fill=(0, 0, 0), width=2)
        y += 20
    elif doc_type in HAS_TOTAL:
        total = round(rng.uniform(5, 5000), 2)
    if doc_type not in HAS_TOTAL:
        total = None
        for _ in range(rng.randint(3, 7)):                      # body paragraph lines
            s = fk.sentence(nb_words=rng.randint(6, 12))
            d.text((40, y), s[:70], font=body, fill=(40, 40, 40))
            text.append(s[:70])
            y += 32
    if total is not None:
        tt = f"{T['total']}: {_money(total, lang)}"
        d.text((W - 40 - d.textlength(tt, font=font(BOLD[0], 32)), y + 10), tt, font=font(BOLD[0], 32), fill=(0, 0, 0))
        text.append(tt)
        y += 70
    if fields["bank_details"]:
        iban = f"{T['bank']}: " + fk.iban() if hasattr(fk, "iban") else f"{T['bank']}: {rng.randint(10**9, 10**10)}"
        d.text((40, y), iban[:70], font=small, fill=(30, 30, 30))
        text.append(iban[:70])
        y += 40
    if fields["barcode"]:
        x = 40
        yb = min(H - 150, y + 10)
        while x < 360:
            w = rng.choice([1, 2, 3, 4])
            if rng.random() < .55:
                d.rectangle([x, yb, x + w, yb + 70], fill=(0, 0, 0))
            x += w + rng.choice([1, 2])
        y = yb + 90
    if fields["signature"]:
        ys = H - 150
        pts, x = [], 480
        for _ in range(12):
            pts.append((x, ys + rng.randint(-18, 18)))
            x += rng.randint(12, 22)
        d.line(pts, fill=(20, 30, 120), width=3)
        d.line([470, ys + 30, 800, ys + 30], fill=(0, 0, 0), width=1)
        d.text((470, ys + 36), T["sign"], font=small, fill=(60, 60, 60))
        text.append(f"{T['sign']}: ______ (signed)")
    if fields["paid_stamp"]:
        st = Image.new("RGBA", (360, 130), (0, 0, 0, 0))
        sd = ImageDraw.Draw(st)
        sd.rectangle([4, 4, 355, 125], outline=(200, 20, 20, 220), width=8)
        sd.text((30, 22), T["paid"], font=font(BOLD[0], 72), fill=(200, 20, 20, 220))
        st = st.rotate(rng.uniform(-25, 25), expand=True)
        im.paste(st, (rng.randint(300, 460), rng.randint(300, 600)), st)
        text.append(f"[{T['paid']}]")
    # scan / photo degradations
    if rng.random() < 0.5:
        im = im.convert("L").convert("RGB")
    if rng.random() < 0.5:
        im = im.rotate(rng.uniform(-2.5, 2.5), expand=False, fillcolor=(250, 250, 250))
    if rng.random() < 0.4:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    truth = {"fields": fields, "total": total, "paid": fields["paid_stamp"], "signed": fields["signature"]}
    return im, "\n".join(text), truth
