📘 OCR Automation Pipeline

A Python-based OCR automation project that continuously monitors an input directory, converts PDF files to images, extracts text using Tesseract OCR, and generates structured JSON outputs.
This system is designed for organizations that need automated text extraction, document processing, and error logging in real time.

🚀 Features

Automated Folder Watching
Monitors an input folder and triggers processing as soon as a new PDF file appears.

PDF to Image Conversion
Converts PDF pages into high-quality PNG images using pdf2image.

OCR Text Extraction
Uses Tesseract OCR (Persian + English) to extract text and bounding box data.

JSON Output Generator
Automatically organizes extracted text into structured JSON files.

Error Handling & Logging
Logs all processing errors into a dedicated log directory.

Fully Automated Workflow
No manual interaction required — just drop a PDF file into the input folder.

🛠️ Technologies Used

Python 3.x

pdf2image

OpenCV (cv2)

Tesseract OCR

Watchdog

Pillow (PIL)

NumPy

📂 Project Structure
ocr/
│
├── main.py                   # Main automation script
├── .idea/                    # IDE project settings
├── input/                    # Drop PDF files here
├── output/                   # Contains JSON + processed files
└── logs/                     # Stores all error logs

⚙️ Installation
1️⃣ Install required Python packages:
pip install -r requirements.txt

2️⃣ Install Tesseract OCR

Make sure Tesseract is installed and update the script paths accordingly:

TESSERACT_EXE = r"E:\path\to\tesseract.exe"
POPPLER_BIN = r"E:\path\to\poppler\bin"

▶️ How to Run

Run the main script:

python main.py


Then simply place your PDF files inside the input folder.

The application will:

Convert PDF → PNG

Extract text

Generate JSON output

Log any errors automatically

📄 Example Output (JSON)
{
  "file": "document.pdf",
  "pages": [
    {
      "page_number": 1,
      "text": "Extracted text goes here...",
      "blocks": [...]
    }
  ]
}

📌 Notes

Supports Persian (Farsi) and English OCR.

Designed for fully automated document-processing pipelines.

Easy to extend with database storage, APIs, or cloud integration.

👤 Author

Hosein Elyasi
Python Developer & Automation Engineer
GitHub: https://github.com/ho3eincloner
