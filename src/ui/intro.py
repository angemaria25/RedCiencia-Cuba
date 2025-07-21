import streamlit as st

def show_intro():
    # Crear columnas para un diseño más moderno
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #1f77b4; margin-bottom: 0.5rem;'> Sistema de Análisis de Redes Científicas</h1>
            <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
                Plataforma para el análisis multidimensional del ecosistema científico cubano
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sección principal con información del sistema
    st.markdown("""
    Esta plataforma te permite explorar el ecosistema científico desde múltiples perspectivas:
    
    """)
    
    # Crear tres columnas para las características principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤝 **Colaboraciones**
        - Redes de coautoría
        - Patrones de trabajo conjunto
        - Comunidades científicas
        """)
    
    with col2:
        st.markdown("""
        ### 📚 **Temáticas**
        - Mapas de conocimiento
        - Especialización disciplinaria
        - Interdisciplinariedad
        """)
    
    with col3:
        st.markdown("""
        ### 🏛️ **Instituciones**
        - Alianzas estratégicas
        - Ecosistema organizacional
        - Capacidad de investigación
        """)
    
    st.markdown("---")
    
    # Sección de funcionalidades
    st.markdown("""
    ## **Funcionalidades**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Análisis Cuantitativo:**
        - Métricas de centralidad y conectividad
        - Detección automática de comunidades
        - Análisis de clustering y modularidad
        - Estadísticas descriptivas avanzadas
        
        **Visualización Interactiva:**
        - Grafos dinámicos y responsivos
        - Mapas de calor y nubes de palabras
        - Filtros y búsquedas personalizadas
        - Exportación de resultados
        """)
    
    with col2:
        st.markdown("""
        **Exploración Profunda:**
        - Perfiles detallados de investigadores
        - Análisis institucional comparativo
        - Trayectorias de colaboración
        - Identificación de tendencias emergentes
        
        **Insights Estratégicos:**
        - Identificación de actores clave
        - Oportunidades de colaboración
        - Fortalezas y brechas temáticas
        - Recomendaciones de networking
        """)
    
    st.markdown("---")
    
    # Sección de inicio rápido
    st.markdown("""
    ## 🚀 **Comienza tu Análisis**
    
    **Paso 1:** Ve a la sección **"Cargar Datos"** y sube tu archivo JSON con información de publicaciones científicas.
    
    **Paso 2:** Dirígete a **"Análisis de Redes"** y selecciona el tipo de red que deseas explorar.
    
    **Paso 3:** Interactúa con las visualizaciones, explora comunidades y descubre patrones ocultos en los datos.
    """)
    
    # Llamada a la acción
    st.markdown("""
    <div style='background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 2rem 0;'>
        <h3 style='color: #1f77b4; margin-top: 0;'>💡 ¿List@ para explorar?</h3>
        <p style='margin-bottom: 0; font-size: 1.1rem;'>
            Descubre conexiones inesperadas, identifica oportunidades de colaboración y obtén insights 
            valiosos sobre la estructura y dinámica de tu ecosistema científico.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer con información adicional
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #888; border-top: 1px solid #eee; margin-top: 3rem;'>
        <p><strong>Sistema de Análisis de Redes Científicas</strong> </p>
    </div>
    """, unsafe_allow_html=True)