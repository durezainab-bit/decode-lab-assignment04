"""
Project 4 - Image/Text Recognition (Basic) -> Path 1: OCR
Pipeline: Load -> Grayscale -> Gaussian Blur -> Adaptive Threshold -> Tesseract -> 80% confidence filter -> Visual output

Install:
    pip install opencv-python pytesseract numpy
    + install the Tesseract engine itself:
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
      Linux:   sudo apt install tesseract-ocr
      Mac:     brew install tesseract

Run:
    python ocr_project4.py sample.jpg
"""
import sys
import cv2
import pytesseract

# Windows only: uncomment and fix the path if Tesseract is not found
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CONF_THRESHOLD = 80   # project minimum standard (80%)
PSM = 3               # 3 = auto layout, 6 = one text block, 7 = one line, 11 = sparse text


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             # Step 1: grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)                  # Step 2: blur (noise removal)
    thresh = cv2.adaptiveThreshold(                           # Step 3: adaptive thresholding
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10)
    return gray, thresh


def recognise(thresh):
    data = pytesseract.image_to_data(
        thresh, config=f"--psm {PSM}", output_type=pytesseract.Output.DICT)
    results = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if word and conf >= CONF_THRESHOLD:                   # the 80% gate
            results.append({
                "text": word, "conf": conf,
                "box": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
            })
    return results


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read image: {path}")

    gray, thresh = preprocess(img)
    results = recognise(thresh)

    if not results:
        print(f"No text found with confidence >= {CONF_THRESHOLD}%. Try a clearer image or another PSM.")
        return

    out = img.copy()
    for r in results:
        x, y, w, h = r["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(out, f'{r["text"]} ({r["conf"]:.0f}%)', (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    avg = sum(r["conf"] for r in results) / len(results)
    print("=" * 40)
    print("RECOGNISED TEXT:")
    print(" ".join(r["text"] for r in results))
    print("-" * 40)
    print(f"Words kept (>= {CONF_THRESHOLD}%): {len(results)}")
    print(f"Average confidence: {avg:.1f}%")
    print("Benchmark:", "PASS" if avg >= CONF_THRESHOLD else "FAIL")
    print("=" * 40)

    cv2.imwrite("output_boxes.jpg", out)
    cv2.imwrite("output_threshold.jpg", thresh)
    print("Saved: output_boxes.jpg, output_threshold.jpg")

    cv2.imshow("Original + detected text", out)
    cv2.imshow("Pre-processed (adaptive threshold)", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()