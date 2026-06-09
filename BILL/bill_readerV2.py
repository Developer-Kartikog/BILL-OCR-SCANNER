#!/usr/bin/env python3
import os
import sys
import re
import json
import argparse
import time
import math
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# Dependency checks
_missing = []
try:
    import easyocr
except Exception:
    _missing.append("easyocr (pip install easyocr)")

try:
    import cv2
except Exception:
    _missing.append("opencv-python-headless or opencv-python (pip install opencv-python-headless)")

try:
    from PIL import Image, ImageOps, ImageFilter
except Exception:
    _missing.append("Pillow (pip install pillow)")

try:
    import pandas as pd
except Exception:
    _missing.append("pandas (pip install pandas)")

try:
    import matplotlib.pyplot as plt
except Exception:
    _missing.append("matplotlib (pip install matplotlib)")

try:
    import requests
except Exception:
    _missing.append("requests (pip install requests)")

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# pdf support optional
_HAS_PDF2IMAGE = True
try:
    from pdf2image import convert_from_path
except Exception:
    _HAS_PDF2IMAGE = False

if _missing:
    print("Missing Python packages:")
    for m in _missing:
        print(" -", m)
    print("\nInstall them then re-run. Example:")
    print("pip install easyocr opencv-python-headless pillow pandas matplotlib requests tqdm openpyxl")
    sys.exit(1)

# -------------------------
# Configuration / Patterns
# -------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}
DATE_PATTERNS = [
    r'\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b',
    r'\b(\d{4}[\/\-]\d{2}[\/\-]\d{2})\b',
    r'\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b',
    r'\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b'
]
INVOICE_PATTERNS = [
    r'\bInvoice(?:\sNo| No|#|:)?\s*[:#\-]?\s*([A-Za-z0-9\-\/]+)\b',
    r'\bInv(?:\.|o)?\s*[:#\-]?\s*([A-Za-z0-9\-\/]+)\b',
    r'\bBill\s*[:#\-]?\s*([A-Za-z0-9\-\/]+)\b'
]
AMOUNT_PATTERNS = [
    r'(?:Grand Total|GrandTotal|Total Amount|Total|Amount Due|Amount)\s*[:\-]?\s*(?:INR|Rs\.|Rs|₹)?\s*([0-9\.,]+)',
    r'(?:INR|Rs\.|Rs|₹)\s*([0-9\.,]+)'
]
GST_PATTERN = r'\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b'  # GSTIN len 15
VENDOR_PATTERN = r'^(?:\s*)([A-Z][A-Za-z0-9 &\.\-]{2,60})'  # first line vendor heuristic

CURRENCY_SYMBOLS = {'₹': 'INR', 'Rs': 'INR', 'Rs.': 'INR', '$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY'}

# -------------------------
# Utilities
# -------------------------
def ensure_out_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def list_files_recursive(folder, exts=None):
    p = Path(folder)
    files = []
    for f in p.rglob('*'):
        if f.is_file():
            if exts is None:
                files.append(str(f))
            else:
                if f.suffix.lower() in exts:
                    files.append(str(f))
    return sorted(files)

# -------------------------
# Image preprocessing
# -------------------------
def preprocess_image_cv2(path, enhance=True):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) if hasattr(cv2, 'imdecode') else cv2.imread(path)
    if img is None:
        # fallback to PIL
        im = Image.open(path).convert('RGB')
        arr = np.array(im)[:,:,::-1]
        img = arr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if enhance:
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        # adaptive threshold
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    # deskew
    coords = cv2.findNonZero(255 - gray)
    if coords is not None:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.1:
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return gray

# We'll use PIL/EasyOCR directly, but provide a simple PIL preprocess
def preprocess_pil_image(path, enhance=True):
    img = Image.open(path).convert('RGB')
    if enhance:
        img = ImageOps.exif_transpose(img)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        # contrast stretch
        img = ImageOps.autocontrast(img)
    return img

# -------------------------
# OCR wrapper
# -------------------------

_reader = None
def init_reader(gpu=False, lang_list=['en']):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(lang_list, gpu=gpu)
    return _reader

import numpy as np

def ocr_image_with_confidence(path, reader, use_preprocess=True):
    try:
        if use_preprocess:
            pil_img = preprocess_pil_image(path, enhance=True)
            arr = np.array(pil_img)
            raw = reader.readtext(arr, detail=1)  # list of (bbox, text, conf)
        else:
            raw = reader.readtext(path, detail=1)
    except Exception:
        try:
            img = Image.open(path).convert('RGB')
            raw = reader.readtext(np.array(img), detail=1)
        except Exception:
            return "", []

    # convert segments to lines with average confidence
    lines = []
    # Many receipts split text into small boxes; group by approximate y coordinate
    groups = {}
    for bbox, txt, conf in raw:
        # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        # compute center y
        try:
            ys = [pt[1] for pt in bbox]
        except Exception:
            ys = [pt[1] for pt in bbox]
        cy = sum(ys) / len(ys)
        key = int(round(cy / 10.0))  # bucket by 10px
        groups.setdefault(key, []).append((cy, txt, conf))

    for k in sorted(groups.keys()):
        items = sorted(groups[k], key=lambda x: x[0])
        texts = [t for (_, t, _) in items]
        confs = [c for (_, _, c) in items]
        line_text = " ".join(texts).strip()
        avg_conf = sum(confs) / len(confs) if confs else 0
        lines.append((line_text, avg_conf))

    full_text = "\n".join([ln for ln, _ in lines])
    return full_text, lines


# -------------------------
# PDF handling
# -------------------------
def images_from_pdf(path, dpi=300):
    if not _HAS_PDF2IMAGE:
        return []
    try:
        pages = convert_from_path(path, dpi=dpi)
        temp_paths = []
        for i, page in enumerate(pages):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            page.save(tmp.name, format='PNG')
            temp_paths.append(tmp.name)
        return temp_paths
    except Exception:
        return []

# -------------------------
# Extraction logic
# -------------------------
def find_first(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


# -------------------------
# Currency conversion (optional)
# -------------------------
def convert_currency(amount_str, from_code='USD', to_code='INR'):
    try:
        amt = float(amount_str.replace(',', '').strip())
    except:
        return None
    try:
        url = f"https://api.exchangerate.host/convert?from={from_code}&to={to_code}&amount={amt}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            j = r.json()
            return float(j.get('result', None))
    except Exception:
        return None

# -------------------------
# Per-file output writer
# -------------------------
def write_file_output(out_folder, image_path, ocr_text, structured):
    stem = Path(image_path).stem
    out_path = Path(out_folder) / f"{stem}_output.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== OCR RAW TEXT ===\n\n")
        f.write(ocr_text.strip() + "\n\n")
        f.write("=== EXTRACTED SUMMARY ===\n\n")
        rows = [
            ("Vendor", structured.get('vendor','')),
            ("Invoice", structured.get('invoice','')),
            ("Date", structured.get('date','')),
            ("GSTIN", structured.get('gstin','')),
            ("Amount (raw)", structured.get('amount_raw','')),
            ("Currency", structured.get('currency',''))
        ]
        # write pretty table-like
        maxk = max(len(r[0]) for r in rows)
        for k, v in rows:
            f.write(f"{k.ljust(maxk)} : {v}\n")
    return str(out_path)

def parse_number_token(tok):
    tok0 = tok.strip()
    tok0 = tok0.replace('O', '0').replace('o','0')  # common OCR fixes
    tok0 = tok0.replace('l', '1').replace('I','1')
    tok0 = tok0.replace(',', '')
    m = re.search(r'([0-9]+(?:\.[0-9]{1,2})?)', tok0)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None

def extract_structured_safe(text, lines_with_conf):
    data = {'vendor':'', 'invoice':'', 'date':'', 'gstin':'', 'amount_raw':'', 'currency':''}
    # vendor: first high-confidence alpha line
    for ln,conf in lines_with_conf[:6]:
        if conf >= 0.5 and re.search(r'[A-Za-z]{3,}', ln):
            data['vendor'] = ln.strip()
            break
    # invoice: first invoice pattern
    for p in INVOICE_PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            data['invoice'] = m.group(1).strip()
            break
    # date
    for pat in DATE_PATTERNS:
        m = re.search(pat, text)
        if m:
            data['date'] = m.group(1)
            break
    # gst
    m = re.search(GST_PATTERN, text)
    if m:
        data['gstin'] = m.group(1)
    # Amount extraction: prefer lines with keywords AND reasonable numeric
    candidate_amounts = []
    for ln, conf in lines_with_conf:
        low = ln.lower()
        if any(k in low for k in ['total','grand total','amount due','amount','net amount','balance']):
            # extract all numeric tokens on this line
            toks = re.findall(r'[\d\.,]{2,}', ln)
            for t in toks:
                num = parse_number_token(t)
                if num is not None:
                    candidate_amounts.append((num, ln, conf))
    # if none found, look for currency-prefixed tokens
    if not candidate_amounts:
        for ln, conf in lines_with_conf:
            toks = re.findall(r'(?:₹|rs\.?|inr|\$|€|£)\s*[\d\.,]+', ln, flags=re.IGNORECASE)
            for t in toks:
                num = parse_number_token(t)
                if num is not None:
                    candidate_amounts.append((num, ln, conf))
    # fallback: scan all numeric tokens but filter out absurdly large
    if not candidate_amounts:
        for ln, conf in lines_with_conf:
            toks = re.findall(r'[\d\.,]{3,}', ln)
            for t in toks:
                num = parse_number_token(t)
                if num is not None and num < 10_000_000:  # sanity cap
                    candidate_amounts.append((num, ln, conf))
    # pick best candidate: highest confidence, prefer higher numeric if from 'total' lines
    chosen = None
    if candidate_amounts:
        # score = (is_total_line * 10) + normalized_confidence + log(amount)
        scored = []
        for num, ln, conf in candidate_amounts:
            score = conf
            if re.search(r'\b(total|grand total|amount due|net amount|balance)\b', ln, re.IGNORECASE):
                score += 5.0
            # penalize very large numbers
            if num > 1e7:
                score -= 10.0
            scored.append((score, num, ln, conf))
        scored.sort(reverse=True, key=lambda x: x[0])
        chosen = scored[0]
        data['amount_raw'] = str(chosen[1])
    # currency detection
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in text:
            data['currency'] = code
            break
    return data

def produce_master_exports_safe(records, output_folder, export_csv=True, export_excel=True):
    df = pd.DataFrame(records)
    def to_float_safe(x):
        try:
            v = float(str(x).replace(',',''))
            if v > 1e8 or v < 0:  # discard absurd or negative totals
                return None
            return v
        except:
            return None
    df['amount'] = df['amount_raw'].apply(to_float_safe)
    total = df['amount'].dropna().sum() if 'amount' in df else 0.0
    # exports same as before but ensure types are native
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ensure_out_folder(output_folder)
    csv_path = Path(output_folder)/f"summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    xlsx_path = Path(output_folder)/f"summary_{timestamp}.xlsx"
    df.to_excel(xlsx_path, index=False)
    # generate chart and html as before...
    return {'csv': str(csv_path), 'xlsx': str(xlsx_path), 'total': float(total)}


# -------------------------
# CLI & GUI bootstrap
# -------------------------
def run_cli():
    p = argparse.ArgumentParser(prog="bill_reader_v2_full", description="Advanced Bill Reader V2 - Full Feature")
    p.add_argument("--input","-i", default="input", help="Input folder path")
    p.add_argument("--output","-o", default="outputs", help="Output folder path")
    p.add_argument("--recursive","-r", action="store_true", help="Scan recursively")
    p.add_argument("--no-excel", action="store_true", help="Disable Excel export")
    p.add_argument("--no-csv", action="store_true", help="Disable CSV export")
    p.add_argument("--convert-to", default=None, help="Convert currency to this code (e.g. INR)")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (currently single-threaded OCR by default)")
    p.add_argument("--watch", action="store_true", help="Watch mode (not implemented)")
    p.add_argument("--gui", action="store_true", help="Launch simple GUI")
    args = p.parse_args()
    if args.gui:
        launch_simple_gui(args.input, args.output)
        return
    process_all(args.input, args.output, recursive=args.recursive, export_csv=not args.no_csv, export_excel=not args.no_excel, workers=args.workers, convert_currency_to=args.convert_to, watch=args.watch)

# Simple GUI using tkinter (minimal)
def launch_simple_gui(default_in="input", default_out="outputs"):
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception:
        print("Tkinter not available in this environment.")
        return
    root = tk.Tk()
    root.title("Bill Reader V2 - Simple GUI")
    root.geometry("600x320")
    tk.Label(root, text="Bill Reader V2 - Drop folder or choose input/output", font=("Arial", 12, "bold")).pack(pady=8)
    in_var = tk.StringVar(value=default_in)
    out_var = tk.StringVar(value=default_out)
    tk.Label(root, text="Input folder").pack()
    tk.Entry(root, textvariable=in_var, width=60).pack()
    tk.Button(root, text="Browse Input", command=lambda: in_var.set(filedialog.askdirectory())).pack(pady=4)
    tk.Label(root, text="Output folder").pack()
    tk.Entry(root, textvariable=out_var, width=60).pack()
    tk.Button(root, text="Browse Output", command=lambda: out_var.set(filedialog.askdirectory())).pack(pady=4)
    status = tk.StringVar(value="Ready")
    tk.Label(root, textvariable=status, fg="blue").pack(pady=6)
    def run_proc():
        status.set("Running...")
        root.update()
        try:
            process_all(in_var.get(), out_var.get(), recursive=False, export_csv=True, export_excel=True)
            messagebox.showinfo("Done", "Processing finished. Check output folder.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        status.set("Ready")
    tk.Button(root, text="Run", bg="green", fg="white", command=run_proc).pack(pady=12)
    root.mainloop()

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    run_cli()
