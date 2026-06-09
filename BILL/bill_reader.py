import re
import easyocr
from PIL import Image
import os
import sys

invoice_no_pattern = re.compile(r"(?:Invoice|Bill)\D*(\d+)", re.IGNORECASE)
date_pattern = re.compile(r"\b(?:\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})\b")
amount_pattern = re.compile(r"(?:Total|Amount)\D*(\d+(?:\.\d{2})?)", re.IGNORECASE)

def extract_invoice_data(text):
    invoice_no = invoice_no_pattern.search(text)
    date = date_pattern.search(text)
    amount = amount_pattern.search(text)

    return {
        "Invoice Number": invoice_no.group(1) if invoice_no else "N/A",
        "Date": date.group(0) if date else "N/A",
        "Amount": amount.group(1) if amount else "N/A"
    }

def process_folder(input_folder, output_folder):
    print(f"\n🔍 Scanning folder: {input_folder}")
    if not os.path.exists(input_folder):
        print("❌ Error: Input folder not found!")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)
    summary_lines = []
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        print("⚠️ No images found in the input folder!")
        return

    print(f"📸 Found {len(image_files)} image(s). Starting OCR...\n")
    reader = easyocr.Reader(['en'])

    for idx, file in enumerate(image_files, 1):
        img_path = os.path.join(input_folder, file)
        print(f"[{idx}/{len(image_files)}] Processing: {file}")

        try:
            result = reader.readtext(img_path, detail=0)
            text = "\n".join(result)
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")
            continue

        if not text.strip():
            print(f"⚠️ No readable text found in {file}. Skipping...")
            continue

        data = extract_invoice_data(text)
        print(f"🧾 Extracted: {data}")

        txt_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Invoice Summary\n")
            f.write("====================\n")
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
            f.write("\nRaw OCR Text:\n")
            f.write("====================\n")
            f.write(text.strip())

        summary_lines.append(f"{file}\t{data['Invoice Number']}\t{data['Date']}\t{data['Amount']}")

    master_summary = os.path.join(output_folder, "master_summary.txt")
    with open(master_summary, "w", encoding="utf-8") as f:
        f.write("File\tInvoice Number\tDate\tAmount\n")
        f.write("="*70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\n✅ Done! {len(summary_lines)} file(s) processed.")
    print(f"📁 Results saved to: {output_folder}")
    print(f"📄 Master summary created: {master_summary}\n")

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    process_folder(input_folder, output_folder)