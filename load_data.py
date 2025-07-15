import streamlit as st
from logic.logic import pipeline_simple
import os

st.set_page_config(page_title="Procesador de PDFs", layout="wide")
st.title("📄➡️📊 Procesador de Artículos Científicos")

# Configuración
pdf_dir = st.text_input("Ruta de la carpeta con PDFs:", help="Ej: C:/MisDocumentos/PDFs")
output_json = st.text_input("Nombre del archivo JSON de salida:", "articulos_procesados.json")

if st.button("🚀 Procesar PDFs", type="primary"):
    if not pdf_dir or not os.path.isdir(pdf_dir):
        st.error("¡Debes ingresar una carpeta válida con archivos PDF!")
    else:
        with st.spinner("Procesando PDFs. Esto puede tardar varios minutos..."):
            try:
                resultados = pipeline_simple(pdf_dir, output_json)
                
                st.success(f"✅ ¡Listo! {len(resultados)} artículos procesados")
                
                # Mostrar primer resultado
                with st.expander("Ver primer artículo procesado"):
                    st.json(resultados[0])
                
                # Botón de descarga
                with open(output_json, "rb") as f:
                    st.download_button(
                        "💾 Descargar JSON",
                        f,
                        file_name=output_json,
                        mime="application/json"
                    )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.code(str(e))  # Muestra detalles del error