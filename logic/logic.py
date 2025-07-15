import os
import json
import fitz  # PyMuPDF
from openai import OpenAI

def pdf_a_txt(pdf_path, txt_dir):
    """Convierte un PDF a archivo de texto"""
    os.makedirs(txt_dir, exist_ok=True)
    txt_path = os.path.join(txt_dir, os.path.basename(pdf_path).replace('.pdf', '.txt'))
    
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return txt_path
    except Exception as e:
        print(f"Error procesando {pdf_path}: {str(e)}")
        return None

def procesar_con_lmstudio(texto, client):
    """Envía el texto a LMStudio para extraer información estructurada"""
    prompt = f"""Extrae la siguiente información del artículo científico en formato JSON:
- título (string)
- autores (lista)
- instituciones (lista)
- palabras_clave (lista)
- resumen (string)

Texto del artículo:
{texto[:4000]}"""  # Limita el texto para no sobrecargar el modelo

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct-v0.3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"error": "No se pudo parsear la respuesta"}

def pipeline_simple(pdf_dir, output_json):
    """Procesa todos los PDFs en una carpeta y guarda los resultados en JSON"""
    client = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")
    
    resultados = []
    txt_dir = os.path.join(os.path.dirname(output_json), "textos_tmp")
    
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            
            # Paso 1: Convertir PDF a texto
            txt_path = pdf_a_txt(pdf_path, txt_dir)
            if not txt_path:
                continue
                
            # Paso 2: Leer el texto
            with open(txt_path, 'r', encoding='utf-8') as f:
                texto = f.read()
            
            # Paso 3: Procesar con LMStudio
            datos = procesar_con_lmstudio(texto, client)
            datos["archivo_origen"] = pdf_file
            resultados.append(datos)
            
            print(f"Procesado: {pdf_file}")
    
    # Guardar resultados
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    return resultados