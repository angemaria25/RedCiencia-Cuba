import streamlit as st
import json

st.set_page_config(
    page_title="Cargar Datos",
    page_icon="📤",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📤 Cargar Archivo JSON")
st.markdown("""
Sube tu archivo JSON con datos de artículos científicos para comenzar.
""")

#Widget para cargar archivo
uploaded_file = st.file_uploader(
    label=" ",
    type=["json"],
    help="Formato esperado: Lista de artículos o objeto con clave 'articulos'."
)

#Procesamiento solo si se sube un archivo
if uploaded_file is not None:  
    try:
        data = json.load(uploaded_file)
        
        #Convertir a lista uniforme
        articulos = data["articulos"] if isinstance(data, dict) and "articulos" in data else data
        
        if not isinstance(articulos, list):
            raise ValueError("El JSON no contiene una lista de artículos.")
        
        #Guardar en session_state
        st.session_state['articulos'] = articulos
        st.success(f"✅ ¡Archivo cargado! ({len(articulos)} artículos disponibles)")
        
        #Vista previa colapsable
        with st.expander("🔍 Ver muestra de los datos", expanded=False):
            st.json(articulos[:1])  #Muestra solo 1 artículo para no saturar

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
        st.session_state.clear()  #Limpiar datos corruptos