from PIL import Image
import pytesseract

# Set Tesseract path (update this path to match your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Paths to the uploaded images
image_path_1 = "Manjur_Ansari_1.jpg"
image_path_2 = "Manjur_Ansari_2.jpg"


# Function to extract text from an image file
def extract_text_from_image(image_path):
    try:
        # Open the image using PIL
        img = Image.open(image_path)
        # Extract text using pytesseract
        text = pytesseract.image_to_string(img)
        # append the text to a file
        with open('output1.txt', 'a') as file:
            file.write(text)
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"


# Extract text from the images using pytesseract
# text_1 = extract_text_from_image(image_path_1)
# text_2 = extract_text_from_image(image_path_2)

extract_text_from_image("1.png")