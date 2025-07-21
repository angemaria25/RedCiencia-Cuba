"""
Módulo de Exploración de Redes Científicas
==========================================

Contiene las funciones principales para la exploración y análisis de diferentes tipos de redes.
"""

import streamlit as st
# from ..analysis import Metricas
# from ..analysis import DataScience
from .exploracion_analisis_general import analisis_general_tab
from .exploracion_autores import autores_tab
from .exploracion_instituciones import instituciones_tab
from .exploracion_articulos import articulos_tab

# Nuevas funciones para las pestañas reorganizadas
def redes_colaboracion_tab(articulos):
    """Pestaña para análisis de redes de colaboración"""
    # Establecer directamente la red de colaboración autor-autor
    st.session_state.red_seleccionada = "Red de Colaboración Autor-Autor"
    
    # Llamar a la función de autores
    autores_tab(articulos)

def redes_tematicas_tab(articulos):
    """Pestaña para análisis temático y de campos"""
    enfoque = st.radio(
        "Elige la red a visualizar:",
        [
            "Investigadores y Áreas (Red Autor-Campo de Estudio)",
            "Instituciones y Disciplinas (Red Institución-Campo de Estudio)"
        ],
        key="enfoque_tematico"
    )
    
    # Establecer la red seleccionada y llamar a la función correspondiente
    if "Investigadores y Áreas" in enfoque:
        st.session_state.red_seleccionada = "Red Autor-Campo de Estudio"
        autores_tab(articulos)
    else:
        st.session_state.red_seleccionada = "Red Institución-Campo de Estudio"
        instituciones_tab(articulos)

def redes_institucionales_tab(articulos):
    """Pestaña para análisis institucional"""
    perspectiva = st.radio(
        "Elige la red a visualizar:",
        [
            "Colaboración Interinstitucional (Red Institución-Institución)",
            "Vínculos Institución-Investigador (Red Institución-Autor)"
        ],
        key="perspectiva_institucional"
    )
    
    # Establecer la red seleccionada según la opción
    if "Colaboración Interinstitucional" in perspectiva:
        st.session_state.red_seleccionada = "Red Institución-Institución"
    else:
        st.session_state.red_seleccionada = "Red Institución-Autor"
    
    # Llamar a la función de instituciones
    instituciones_tab(articulos)

def exploracion_section(articulos):
    """Función principal de la sección de exploración."""
    # Título en azul como en la página de inicio
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; margin-bottom: 1rem;'>Análisis de Redes Científicas</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Usar st.info para crear el contenedor con borde azul
    st.info("""
    **Explora el ecosistema científico a través de múltiples perspectivas de análisis de redes:**
    
    🤝 Redes de Colaboración: Descubre patrones de trabajo conjunto entre investigadores, revelando estructuras de colaboración directa en el ecosistema científico.
    
    📚 Análisis Temático: Mapea las conexiones entre disciplinas y especialidades, identificando cómo investigadores e instituciones se relacionan con diferentes áreas del conocimiento.
    
    🏛️ Redes Institucionales: Analiza el panorama organizacional de la ciencia, explorando alianzas estratégicas entre instituciones y sus vínculos con el talento investigador.
    
    *Cada red ofrece métricas avanzadas, detección de comunidades, análisis de centralidad y visualizaciones interactivas para comprender la estructura y dinámica del sistema científico.*
    """)

    if not articulos:
        st.warning("No hay datos para mostrar.")
        return

    tabs = st.tabs(["🤝 Redes de Colaboración", "📚 Análisis Temático", "🏛️ Redes Institucionales"])
    
    with tabs[0]:
        redes_colaboracion_tab(articulos)
    with tabs[1]:
        redes_tematicas_tab(articulos)
    with tabs[2]:
        redes_institucionales_tab(articulos)