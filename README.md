# Sistema de Análisis de Redes Científicas 🔬

Un sistema completo para el análisis y visualización de redes de colaboración científica desarrollado con Streamlit y NetworkX.

## 🚀 Cómo Ejecutar

```bash
streamlit run interfaz.py
```

## 📊 Funcionalidades

### 🤝 Redes de Colaboración
- **Colaboración entre Investigadores**: Análisis de redes autor-autor

### 📚 Análisis Temático
- **Investigadores y Áreas**: Relación entre autores y campos de estudio
- **Instituciones y Disciplinas**: Conexión entre instituciones y áreas temáticas

### 🏛️ Redes Institucionales
- **Colaboración Interinstitucional**: Análisis de colaboraciones entre instituciones
- **Vínculos Institución-Investigador**: Relaciones entre instituciones y autores

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework de aplicaciones web
- **NetworkX**: Análisis y manipulación de redes
- **Pandas**: Manipulación de datos
- **Matplotlib**: Visualización de datos
- **WordCloud**: Generación de nubes de palabras
- **Plotly**: Visualizaciones interactivas

## 📋 Requisitos

```bash
pip install streamlit networkx pandas matplotlib wordcloud plotly
```

## 🔧 Configuración

Las configuraciones del sistema se encuentran en `config/settings.py` donde puedes modificar:
- Colores de visualización
- Rutas de archivos
- Parámetros de análisis
- Configuraciones de la interfaz

## 📈 Tipos de Análisis Disponibles

1. **Análisis de Autores**
   - Redes de colaboración
   - Métricas de centralidad
   - Detección de comunidades
   - Análisis de productividad

2. **Análisis de Instituciones**
   - Colaboraciones interinstitucionales
   - Diversidad temática
   - Análisis de impacto

3. **Análisis Temático**
   - Mapas de conocimiento
   - Evolución de campos
   - Interdisciplinariedad

