from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator

# Pipeline para descripción de imágenes
caption_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Abrir imagen
image = Image.open("images/algo.jfif")

# Obtener la descripción en inglés
caption_result = caption_pipe(image)
caption_en = caption_result[0]['generated_text']
print("Descripción en inglés:", caption_en)

# Traducir al español usando Google Translate
caption_es = GoogleTranslator(source='auto', target='es').translate(caption_en)
print("Descripción en español:", caption_es)
