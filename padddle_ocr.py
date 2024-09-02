import cv2
from paddleocr import PaddleOCR, draw_ocr


# Initialize the PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Use 'ch' for Chinese, 'en' for English

# Load image
image_path = 'tag.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Perform OCR
results = ocr.ocr(image, cls=True)

if not results:
    print("No results from OCR.")
    exit()

# Print results
for result in results:
    for line in result:
        print(f'Text: {line[1][0]}, Confidence: {line[1][1]}')

# Extract boxes, texts, and scores
boxes = [elements[0] for elements in results[0]]
pairs = [elements[1] for elements in results[0]]
txts = [pair[0] for pair in pairs]
scores = [pair[1] for pair in pairs]

# Convert image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Draw the result on the image
image_with_boxes = draw_ocr(image_rgb, boxes, txts, scores, font_path='C:\Windows\Fonts\Arial.ttf')

# Convert back to BGR before saving
image_with_boxes_bgr = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)

# Save results image
cv2.imwrite('result.jpg', image_with_boxes_bgr)

print("OCR results saved to result.jpg")
