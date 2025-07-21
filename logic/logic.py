import os
import json
import concurrent.futures
import threading

# --- CONFIGURACIÓN LLM (LMStudio) ---
LMSTUDIO_URL = "http://localhost:5000/v1/chat/completions"
LLM_MODEL = "mistralai/mistral-7b-instruct-v0.3"
HEADERS = {"Content-Type": "application/json"}

# --- 1. EXTRACCIÓN DE TEXTO DE PDFS ---
def extraer_textos_de_pdfs(pdf_dir, txt_dir):
    """Extrae el texto de todos los PDFs en pdf_dir y los guarda como .txt en txt_dir."""
    from shutil import rmtree
    
    if os.path.exists(txt_dir):
        rmtree(txt_dir)
    os.makedirs(txt_dir, exist_ok=True)
    
    # Intentar usar las librerías de PDF directamente
    try:
        # Intentar PyPDF2 primero
        try:
            import PyPDF2
            print("Usando PyPDF2 para extraer texto de PDFs...")
            _extract_with_pypdf2(pdf_dir, txt_dir)
            return
        except ImportError:
            pass
        
        # Intentar pdfplumber
        try:
            import pdfplumber
            print("Usando pdfplumber para extraer texto de PDFs...")
            _extract_with_pdfplumber(pdf_dir, txt_dir)
            return
        except ImportError:
            pass
        
        # Intentar PyMuPDF
        try:
            import fitz  # PyMuPDF
            print("Usando PyMuPDF para extraer texto de PDFs...")
            _extract_with_pymupdf(pdf_dir, txt_dir)
            return
        except ImportError:
            pass
        
        # Intentar pdfminer
        try:
            from pdfminer.high_level import extract_text
            print("Usando pdfminer para extraer texto de PDFs...")
            _extract_with_pdfminer(pdf_dir, txt_dir)
            return
        except ImportError:
            pass
        
        # Si ninguna librería está disponible
        raise RuntimeError("No se encontró ninguna librería de PDF instalada. Instale al menos una de estas librerías: PyPDF2, pdfplumber, PyMuPDF, o pdfminer.")
        
    except Exception as e:
        raise RuntimeError(f"Error al extraer texto de PDFs: {e}")

def _extract_with_pypdf2(pdf_dir, txt_dir):
    """Extrae texto usando PyPDF2"""
    import PyPDF2
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise RuntimeError(f"No se encontraron archivos PDF en {pdf_dir}")
    
    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        txt_path = os.path.join(txt_dir, filename.replace('.pdf', '.txt'))
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():  # Solo guardar si hay texto
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
                print(f"Extraído: {filename}")
            else:
                print(f"Advertencia: No se pudo extraer texto de {filename}")
                
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

def _extract_with_pdfplumber(pdf_dir, txt_dir):
    """Extrae texto usando pdfplumber"""
    import pdfplumber
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise RuntimeError(f"No se encontraron archivos PDF en {pdf_dir}")
    
    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        txt_path = os.path.join(txt_dir, filename.replace('.pdf', '.txt'))
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():  # Solo guardar si hay texto
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
                print(f"Extraído: {filename}")
            else:
                print(f"Advertencia: No se pudo extraer texto de {filename}")
                
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

def _extract_with_pymupdf(pdf_dir, txt_dir):
    """Extrae texto usando PyMuPDF"""
    import fitz
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise RuntimeError(f"No se encontraron archivos PDF en {pdf_dir}")
    
    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        txt_path = os.path.join(txt_dir, filename.replace('.pdf', '.txt'))
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
            
            if text.strip():  # Solo guardar si hay texto
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
                print(f"Extraído: {filename}")
            else:
                print(f"Advertencia: No se pudo extraer texto de {filename}")
                
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

def _extract_with_pdfminer(pdf_dir, txt_dir):
    """Extrae texto usando pdfminer"""
    from pdfminer.high_level import extract_text
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise RuntimeError(f"No se encontraron archivos PDF en {pdf_dir}")
    
    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        txt_path = os.path.join(txt_dir, filename.replace('.pdf', '.txt'))
        
        try:
            text = extract_text(pdf_path)
            if text and text.strip():  # Solo guardar si hay texto
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
                print(f"Extraído: {filename}")
            else:
                print(f"Advertencia: No se pudo extraer texto de {filename}")
                
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

# --- 2. PROCESAMIENTO DE TXT A JSON ESTRUCTURADO ---
def procesar_txts_a_json(txt_dir, output_json):
    """Procesa todos los .txt en txt_dir y genera un JSON estructurado con metadatos de artículos."""
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")
    
    files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
    if not files:
        raise RuntimeError(f"No se encontraron archivos .txt en {txt_dir}")
    
    output_json_list = []
    
    def process_file(filename):
        try:
            with open(os.path.join(txt_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error leyendo {filename}: {e}")
            return None
        
        # Crear estructura base del JSON
        article_data = {
            "Nombre de Articulo": "",
            "Campo de Estudio": "Ciencias Sociales",
            "Autores Principales": [],
            "Autores Secundarios": [],
            "Institucion Principal": "",
            "Instituciones Secundarias": [],
            "Pais": "Cuba",
            "Numero de Palabras": len(text.split()),
            "Palabras Clave": [],
            "Referencias Bibliograficas": [],
            "Autores de Articulos Referenciados": [],
            "Instituciones de Articulos Referenciados": [],
            "Archivo": filename
        }
        
        # Un solo prompt para extraer toda la información
        prompt = f"""Analiza el siguiente artículo científico y extrae TODA la información solicitada. Responde con el formato exacto:
TITULO: [título del artículo]
CAMPO: [uno de: Matematicas y Computacion, Fisica, Biologia, Quimica, Ciencias Sociales, Agricultura y Botanica, Deportes y Cultura Fisica, Arte, Politica e Historia]
AUTORES_PRINCIPALES: [nombres de autores principales separados por comas]
AUTORES_SECUNDARIOS: [nombres de autores secundarios separados por comas]
INSTITUCION_PRINCIPAL: [nombre de la institución principal]
INSTITUCIONES_SECUNDARIAS: [nombres de instituciones secundarias separadas por comas]
PAIS: [país de origen]
PALABRAS_CLAVE: [palabras clave separadas por comas]
REFERENCIAS: [títulos de referencias bibliográficas separadas por punto y coma]
AUTORES_REFERENCIADOS: [nombres de autores de artículos referenciados separados por comas]
INSTITUCIONES_REFERENCIADAS: [nombres de instituciones de artículos referenciados separadas por comas]

Texto del artículo:
{text[:4000]}"""
        
        try:
            response = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct-v0.3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parsear la respuesta
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('TITULO:'):
                    titulo = line.replace('TITULO:', '').strip()
                    if titulo and len(titulo) > 5:
                        article_data["Nombre de Articulo"] = titulo
                        
                elif line.startswith('CAMPO:'):
                    campo = line.replace('CAMPO:', '').strip()
                    campos_validos = ["Matematicas y Computacion", "Fisica", "Biologia", "Quimica", "Ciencias Sociales", "Agricultura y Botanica", "Deportes y Cultura Fisica", "Arte", "Politica e Historia"]
                    if campo in campos_validos:
                        article_data["Campo de Estudio"] = campo
                        
                elif line.startswith('AUTORES_PRINCIPALES:'):
                    autores_texto = line.replace('AUTORES_PRINCIPALES:', '').strip()
                    if autores_texto and autores_texto.lower() not in ['no encontrado', 'ninguno', 'no hay', 'n/a']:
                        autores = [a.strip() for a in autores_texto.split(',') if a.strip()]
                        article_data["Autores Principales"] = autores[:5]
                        
                elif line.startswith('AUTORES_SECUNDARIOS:'):
                    autores_sec_texto = line.replace('AUTORES_SECUNDARIOS:', '').strip()
                    if autores_sec_texto and autores_sec_texto.lower() not in ['no encontrado', 'ninguno', 'no hay', 'n/a']:
                        autores_sec = [a.strip() for a in autores_sec_texto.split(',') if a.strip()]
                        article_data["Autores Secundarios"] = autores_sec[:5]
                        
                elif line.startswith('INSTITUCION_PRINCIPAL:'):
                    institucion = line.replace('INSTITUCION_PRINCIPAL:', '').strip()
                    if institucion and len(institucion) > 3 and institucion.lower() not in ['no encontrado', 'ninguno', 'no hay', 'n/a']:
                        article_data["Institucion Principal"] = institucion
                        
                elif line.startswith('INSTITUCIONES_SECUNDARIAS:'):
                    inst_sec_texto = line.replace('INSTITUCIONES_SECUNDARIAS:', '').strip()
                    if inst_sec_texto and inst_sec_texto.lower() not in ['no encontrado', 'ninguno', 'no hay', 'n/a']:
                        inst_sec = [i.strip() for i in inst_sec_texto.split(',') if i.strip()]
                        article_data["Instituciones Secundarias"] = inst_sec[:5]
                        
                elif line.startswith('PAIS:'):
                    pais = line.replace('PAIS:', '').strip()
                    if pais and len(pais) > 2 and pais.lower() not in ['no encontrado', 'ninguno', 'no hay', 'n/a']:
                        article_data["Pais"] = pais
                        
                elif line.startswith('PALABRAS_CLAVE:'):
                    palabras_texto = line.replace('PALABRAS_CLAVE:', '').strip()
                    if palabras_texto and palabras_texto.lower() not in ['no encontrado', 'ninguno', 'no hay', 'n/a']:
                        palabras = [p.strip() for p in palabras_texto.split(',') if p.strip()]
                        article_data["Palabras Clave"] = palabras[:10]
                        
                elif line.startswith('REFERENCIAS:'):
                    refs_texto = line.replace('REFERENCIAS:', '').strip()
                    if refs_texto and refs_texto.lower() not in ['no encontrado', 'ninguno', 'no hay', 'n/a']:
                        refs = [r.strip() for r in refs_texto.split(';') if r.strip()]
                        article_data["Referencias Bibliograficas"] = refs[:20]
                        
                elif line.startswith('AUTORES_REFERENCIADOS:'):
                    autores_ref_texto = line.replace('AUTORES_REFERENCIADOS:', '').strip()
                    if autores_ref_texto and autores_ref_texto.lower() not in ['no encontrado', 'ninguno', 'no hay', 'n/a']:
                        autores_ref = [a.strip() for a in autores_ref_texto.split(',') if a.strip()]
                        article_data["Autores de Articulos Referenciados"] = autores_ref[:20]
                        
                elif line.startswith('INSTITUCIONES_REFERENCIADAS:'):
                    inst_ref_texto = line.replace('INSTITUCIONES_REFERENCIADAS:', '').strip()
                    if inst_ref_texto and inst_ref_texto.lower() not in ['no encontrado', 'ninguno', 'no hay', 'n/a']:
                        inst_ref = [i.strip() for i in inst_ref_texto.split(',') if i.strip()]
                        article_data["Instituciones de Articulos Referenciados"] = inst_ref[:20]
                        
        except Exception as e:
            print(f"Error procesando {filename} con LLM: {e}")
            # Si falla, usar el filename como título
            article_data["Nombre de Articulo"] = filename.replace('.txt', '').replace('_', ' ')
        
        return article_data
    
    print(f"Procesando {len(files)} archivos con LLM Studio...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_file, filename) for filename in files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                output_json_list.append(result)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_json_list, f, ensure_ascii=False, indent=2)
    
    print(f"JSON estructurado guardado en: {output_json}")
    return output_json_list

# --- 3. PIPELINE SIMPLIFICADO ---
def pipeline_completo_desde_pdfs(pdf_dir, data_dir, output_json_path):
    """Pipeline simplificado: PDF → TXT → JSON estructurado."""
    txt_dir = os.path.join(data_dir, "texto")
    
    print(f"Iniciando pipeline desde PDFs en: {pdf_dir}")
    
    # 1. Extraer texto de PDFs
    print("Paso 1: Extrayendo texto de PDFs...")
    extraer_textos_de_pdfs(pdf_dir, txt_dir)
    
    # 2. Procesar TXT a JSON estructurado
    print("Paso 2: Procesando texto con LLM Studio...")
    json_articulos = procesar_txts_a_json(txt_dir, output_json_path)
    
    print(f"Pipeline completado. Procesados {len(json_articulos)} artículos.")
    return json_articulos

# --- PARA STREAMLIT ---
def extraer_y_estructurar_desde_pdfs(pdf_dir, output_json_path=None):
    """Función para Streamlit: ejecuta el pipeline simplificado y retorna el JSON estructurado."""
    # Si no se especifica ruta, crear Json_Estructurado.json en la misma carpeta del script
    if output_json_path is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Subir dos niveles desde logic
        output_json_path = os.path.join(script_dir, "Json_Estructurado.json")
    
    data_dir = os.path.dirname(output_json_path)
    json_final = pipeline_completo_desde_pdfs(pdf_dir, data_dir, output_json_path)
    
    return json_final, output_json_path