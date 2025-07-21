"""
Sistema de Análisis de Redes Científicas
========================================

Aplicación principal para el análisis y visualización de redes de colaboración científica.

Autor: [Tu nombre]
Fecha: 2025
"""

import streamlit as st
import sys
import os

# Agregar el directorio src al path para importaciones
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Redes Científicas",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importaciones principales
from src.ui.interfaz_principal import main_interface

if __name__ == "__main__":
    main_interface()