from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator
import io

app = FastAPI()

# Cargar pipeline solo una vez al iniciar el servicio
caption_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@app.post("/describe-image/")
async def describe_image(file: UploadFile = File(...)):
    try:
        # Leer imagen en memoria
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Obtener descripción en inglés
        caption_result = caption_pipe(image)
        caption_en = caption_result[0]['generated_text']

        # Traducir al español
        caption_es = GoogleTranslator(source='auto', target='es').translate(caption_en)

        # Retornar ambas descripciones
        return JSONResponse(content={
            "description_en": caption_en,
            "description_es": caption_es
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
