import os
import json
import time
import traceback
from datetime import datetime
import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image

############################################
# CONFIG
############################################

TESSERACT_EXE = r"E:\New folder (4)\tesseract.exe"
POPPLER_BIN = r"E:\Release-24.08.0-0\poppler-24.08.0\Library\bin"

INPUT_FOLDER = r"E:\input"
OUTPUT_FOLDER = r"E:\output"
LOG_FOLDER = os.path.join(OUTPUT_FOLDER, "log")

# ---- NEW: images folder (only change you wanted) ----
IMAGES_FOLDER = os.path.join(OUTPUT_FOLDER, "images")

# ensure folders exist
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

ensure_dir(IMAGES_FOLDER)
ensure_dir(LOG_FOLDER)

# set tesseract
if os.path.isfile(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
else:
    print("[ERROR] Tesseract path incorrect:", TESSERACT_EXE)


############################################
# UTILS
############################################

def log_error(input_path: str, exc: Exception):
    base = os.path.splitext(os.path.basename(input_path))[0]
    log_path = os.path.join(LOG_FOLDER, f"{base}_error.log")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("==== ERROR ====\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"File: {input_path}\n")
        f.write(f"Error: {repr(exc)}\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        f.write("\n\n")


def get_output_dir(input_path: str) -> str:
    rel = os.path.relpath(os.path.dirname(input_path), INPUT_FOLDER)
    if rel in (".", ""):
        out_dir = OUTPUT_FOLDER
    else:
        out_dir = os.path.join(OUTPUT_FOLDER, rel)
    ensure_dir(out_dir)
    return out_dir


def already_processed(base_name: str, out_dir: str) -> bool:
    for f in os.listdir(out_dir):
        if f.startswith(base_name) and f.endswith(".json"):
            return True
    return False


############################################
# TEXT NORMALIZATION
############################################

_FA_MAP = str.maketrans({
    "ي": "Y",
    "ى": "Y",
    "ئ": "ی",
    "ك": "ک",
    "ۀ": "ه",
    "ة": "ه",
    "ؤ": "و",
    "أ": "ا",
    "إ": "ا",
    "ٱ": "ا",
})

def normalize_fa(text: str) -> str:
    if not text:
        return ""
    text = text.translate(_FA_MAP)
    text = re.sub(r"[\u200c-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
############################################
# IMAGE PREPROCESS BASE
############################################

def hough_deskew(gray):
    try:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

        angle = 0
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                ang = (theta - np.pi / 2) * 180 / np.pi
                if -20 < ang < 20:
                    angles.append(ang)
            if angles:
                angle = float(np.median(angles))

        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    except:
        return gray


def preprocess_base(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    kernel_sharp = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    gray = cv2.filter2D(gray, -1, kernel_sharp)

    gray = cv2.fastNlMeansDenoising(gray, h=13)

    gray = hough_deskew(gray)
    return gray


############################################
# PAGE TYPE DETECTION (book/admin/title)
############################################

def detect_page_type(gray):
    h, w = gray.shape
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - th

    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "book"

    areas = [cv2.contourArea(c) for c in cnts]
    heights = [cv2.boundingRect(c)[3] for c in cnts]

    density = sum(areas) / (h * w)
    avg_h = np.mean(heights)
    num = len(cnts)

    if num < 250 and avg_h > 24 and density < 0.38:
        return "admin"

    if density > 0.45 and num > 320:
        return "book"

    return "book"


############################################
# TITLE PAGE DETECTION
############################################

def detect_title_page(gray):
    h, w = gray.shape
    white_ratio = np.sum(gray > 240) / (h * w)
    if white_ratio > 0.65:
        return True

    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(255 - thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) < 25:
        return True

    return False


def remove_logo(gray):
    h, w = gray.shape
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    mask = np.zeros_like(gray)

    for c in cnts:
        x, y, w0, h0 = cv2.boundingRect(c)
        area = w0 * h0
        cy = y + h0 / 2
        if 0.05 < area / img_area < 0.4 and (h * 0.2 < cy < h * 0.8):
            cv2.rectangle(mask, (x, y), (x + w0, y + h0), 255, -1)

    result = gray.copy()
    result[mask == 255] = 255
    return result


############################################
# STAMP ENGINE (Enhanced)
############################################

def enhance_stamp(gray):
    img = gray.copy()

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
    img = clahe.apply(img)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.Laplacian(img, cv2.CV_8U)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)
    img[mask > 0] = 255

    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        35, 7
    )
    return img
import os
import json
import time
import traceback
from datetime import datetime
import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image

############################################
# CONFIG
############################################

TESSERACT_EXE = r"E:\New folder (4)\tesseract.exe"
POPPLER_BIN = r"E:\Release-24.08.0-0\poppler-24.08.0\Library\bin"

INPUT_FOLDER = r"E:\input"
OUTPUT_FOLDER = r"E:\output"
LOG_FOLDER = os.path.join(OUTPUT_FOLDER, "log")

# فولدر ثابت برای همه‌ی تصاویر خروجی
IMAGES_FOLDER = os.path.join(OUTPUT_FOLDER, "images")

############################################
# UTILS
############################################

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# مطمئن شو فولدرهای لاگ و تصاویر وجود دارند
ensure_dir(LOG_FOLDER)
ensure_dir(IMAGES_FOLDER)


def log_error(input_path: str, exc: Exception):
    ensure_dir(LOG_FOLDER)
    base = os.path.splitext(os.path.basename(input_path))[0]
    log_path = os.path.join(LOG_FOLDER, f"{base}_error.log")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("==== ERROR ====\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"File: {input_path}\n")
        f.write(f"Error: {repr(exc)}\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        f.write("\n\n")


def get_output_dir(input_path: str) -> str:
    rel = os.path.relpath(os.path.dirname(input_path), INPUT_FOLDER)
    if rel in (".", ""):
        out_dir = OUTPUT_FOLDER
    else:
        out_dir = os.path.join(OUTPUT_FOLDER, rel)
    ensure_dir(out_dir)
    return out_dir


def already_processed(base_name: str, out_dir: str) -> bool:
    for f in os.listdir(out_dir):
        if f.startswith(base_name) and f.endswith(".json"):
            return True
    return False


# تنظیم tesseract
if os.path.isfile(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
else:
    print("[ERROR] Tesseract path incorrect:", TESSERACT_EXE)


############################################
# TEXT NORMALIZATION
############################################

_FA_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ئ": "ی",
    "ك": "ک",
    "ۀ": "ه", "ة": "ه",
    "ؤ": "و",
    "أ": "ا", "إ": "ا", "ٱ": "ا",
})

def normalize_fa(text: str) -> str:
    if not text:
        return ""
    text = text.translate(_FA_MAP)
    text = re.sub(r"[\u200c-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


############################################
# IMAGE PREPROCESS BASE
############################################

def hough_deskew(gray):
    try:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

        angle = 0
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                ang = (theta - np.pi / 2) * 180 / np.pi
                if -20 < ang < 20:
                    angles.append(ang)
            if angles:
                angle = float(np.median(angles))

        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    except:
        return gray


def preprocess_base(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel_sharp)

    gray = cv2.fastNlMeansDenoising(gray, h=13)

    gray = hough_deskew(gray)
    return gray


############################################
# PAGE TYPE DETECTION (book/admin/title)
############################################

def detect_page_type(gray):
    h, w = gray.shape
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - th

    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "book"

    areas = [cv2.contourArea(c) for c in cnts]
    heights = [cv2.boundingRect(c)[3] for c in cnts]

    density = sum(areas) / (h * w)
    avg_h = np.mean(heights)
    num = len(cnts)

    if num < 250 and avg_h > 24 and density < 0.38:
        return "admin"

    if density > 0.45 and num > 320:
        return "book"

    return "book"


############################################
# TITLE PAGE + LOGO REMOVAL
############################################

def detect_title_page(gray):
    h, w = gray.shape
    white_ratio = np.sum(gray > 240) / (h * w)
    if white_ratio > 0.65:
        return True

    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(255 - thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) < 25:
        return True

    return False


def remove_logo(gray):
    h, w = gray.shape
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    mask = np.zeros_like(gray)

    for c in cnts:
        x, y, w0, h0 = cv2.boundingRect(c)
        area = w0 * h0
        cy = y + h0 / 2
        if 0.05 < area / img_area < 0.4 and (h * 0.2 < cy < h * 0.8):
            cv2.rectangle(mask, (x, y), (x + w0, y + h0), 255, -1)

    result = gray.copy()
    result[mask == 255] = 255
    return result


############################################
# *** STAMP ENGINE – Enhanced ***
############################################

def enhance_stamp(gray):
    """ تقویت مخصوص مهر، حتی با امضا و خط خوردگی """

    img = gray.copy()

    # 1. افزایش کنتراست محلی فقط ناحیه‌های مشکوک به مهر
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
    img = clahe.apply(img)

    # 2. تقویت لبه‌ها (برای دیده‌شدن نوشته داخل مهر)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.Laplacian(img, cv2.CV_8U)

    # 3. حذف خطوط امضا از روی مهر
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)
    img[mask > 0] = 255

    # 4. باینری‌سازی خیلی دقیق مخصوص مهر
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        35, 7
    )
    return img


############################################
# OCR ENGINE
############################################

def threshold(gray):
    _, o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    a = cv2.adaptiveThreshold(gray, 255,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY,
                              31, 11)
    return cv2.bitwise_or(o, a)


def ocr_blocks(img, page_type, is_title):
    config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"

    if page_type == "book":
        config = "--oem 1 --psm 4 -c preserve_interword_spaces=1"

    if is_title:
        config = "--oem 1 --psm 6"

    data = pytesseract.image_to_data(
        img,
        output_type=Output.DICT,
        lang="fas",
        config=config
    )

    blocks = {}
    N = len(data["text"])

    for i in range(N):
        txt = normalize_fa(data["text"][i])
        if not txt:
            continue

        b = data["block_num"][i]
        p = data["par_num"][i]
        l = data["line_num"][i]

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        key = (b, p)
        blocks.setdefault(key, [])
        blocks[key].append((l, txt, x, y, w, h))

    output = []

    for key, items in blocks.items():
        line_map = {}
        for (ln, txt, x, y, w, h) in items:
            line_map.setdefault(ln, [])
            line_map[ln].append((txt, x, y, w, h))

        lines = []
        for ln, words in sorted(line_map.items()):
            xs = [x for (_, x, _, _, _) in words]
            ys = [y for (_, _, y, _, _) in words]
            x2 = [x + w for (_, x, _, w, _) in words]
            y2 = [y + h for (_, _, y, _, h) in words]

            box = f"{min(xs)} {min(ys)} {max(x2)-min(xs)} {max(y2)-min(ys)}"

            line_text = " ".join(w[0] for w in words)
            line_text = normalize_fa(line_text)

            word_list = []
            for (txt, x, y, w, h) in words:
                word_list.append([txt, 0.95, f"{x} {y} {w} {h}"])

            lines.append({
                "box": box,
                "label": ln,
                "is_header": 0,
                "words": word_list,
                "probability": 0.95,
                "text": line_text
            })

        if not lines:
            continue

        xs = []
        ys = []
        xe = []
        ye = []

        for ln in lines:
            x, y, w, h = map(int, ln["box"].split())
            xs.append(x)
            ys.append(y)
            xe.append(x+w)
            ye.append(y+h)

        part = {
            "lines": lines,
            "box": f"{min(xs)} {min(ys)} {max(xe)-min(xs)} {max(ye)-min(ys)}",
            "text": "\n".join(l["text"] for l in lines),
            "direction": "rtl",
            "type": "text"
        }

        output.append({
            "parts": [part],
            "text": part["text"],
            "direction": "rtl",
            "type": "text"
        })

    return output


############################################
# PROCESS PDF
############################################

def process_pdf(pdf_path):
    print("[PDF] Processing:", pdf_path)

    out_dir = get_output_dir(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    if already_processed(base, out_dir):
        print("[SKIP] Already processed")
        return

    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_BIN)
    except Exception as e:
        log_error(pdf_path, e)
        return

    idx = 0
    for pil in pages:
        idx += 1

        # PNG حالا در فولدر مشترک images ذخیره می‌شود
        png_path = os.path.join(IMAGES_FOLDER, f"{base}_page{idx}.png")
        pil.save(png_path, "PNG")

        base_gray = preprocess_base(pil)
        page_type = detect_page_type(base_gray)

        is_title = detect_title_page(base_gray)
        if is_title:
            gray = remove_logo(base_gray)
        else:
            gray = base_gray

        if page_type == "admin":
            stamp = enhance_stamp(gray)
            final = threshold(stamp)
        else:
            final = threshold(gray)

        h, w = final.shape

        blocks = ocr_blocks(final, page_type, is_title)

        meta = {
            "page_url": png_path.replace("\\", "/"),
            "width": w,
            "height": h,
            "angle": 0,
            "page_type": page_type,
            "is_title_page": int(is_title)
        }

        blocks.append(meta)

        json_path = os.path.join(out_dir, f"{base}_page{idx}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=4)

        print("[OK] Saved:", json_path)


############################################
# PROCESS IMAGE
############################################

def process_image(path):
    print("[IMG] Processing:", path)

    out_dir = get_output_dir(path)
    base = os.path.splitext(os.path.basename(path))[0]

    try:
        pil = Image.open(path)
    except Exception as e:
        log_error(path, e)
        return

    # PNG حالا در فولدر مشترک images ذخیره می‌شود
    png_path = os.path.join(IMAGES_FOLDER, f"{base}_page1.png")
    pil.save(png_path, "PNG")

    base_gray = preprocess_base(pil)
    page_type = detect_page_type(base_gray)
    is_title = detect_title_page(base_gray)

    if is_title:
        gray = remove_logo(base_gray)
    else:
        gray = base_gray

    if page_type == "admin":
        stamp = enhance_stamp(gray)
        final = threshold(stamp)
    else:
        final = threshold(gray)

    h, w = final.shape

    blocks = ocr_blocks(final, page_type, is_title)
    meta = {
        "page_url": png_path.replace("\\", "/"),
        "width": w,
        "height": h,
        "angle": 0,
        "page_type": page_type,
        "is_title_page": int(is_title)
    }
    blocks.append(meta)

    json_path = os.path.join(out_dir, f"{base}_page1.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=4)

    print("[OK] Saved:", json_path)


############################################
# AUTO PROCESS
############################################

def process_existing():
    for root, _, files in os.walk(INPUT_FOLDER):
        for file in files:
            fp = os.path.join(root, file)
            if file.lower().endswith(".pdf"):
                process_pdf(fp)
            elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                process_image(fp)


############################################
# WATCHDOG
############################################

class FileHandler(FileSystemEventHandler):
    def on_created(self, e):
        if e.is_directory:
            return
        time.sleep(1)
        path = e.src_path
        if path.lower().endswith(".pdf"):
            process_pdf(path)
        elif path.lower().endswith((".png", ".jpg", ".jpeg")):
            process_image(path)

    def on_moved(self, e):
        if e.is_directory:
            return
        time.sleep(1)
        path = e.dest_path
        if path.lower().endswith(".pdf"):
            process_pdf(path)
        elif path.lower().endswith((".png", ".jpg", ".jpeg")):
            process_image(path)


def start_watch():
    obs = Observer()
    handler = FileHandler()
    obs.schedule(handler, INPUT_FOLDER, recursive=True)
    obs.start()
    print("[WATCH] Folder:", INPUT_FOLDER)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()


############################################
# MAIN
############################################

if __name__ == "__main__":
    print("[START] Smart OCR v4.0 (Enhanced + Stamp Engine)")
    process_existing()
    start_watch()
