# Sistema de Análisis de Redes Científicas 🔬

Un sistema completo para el análisis y visualización de redes de colaboración científica desarrollado con Streamlit y NetworkX.

## 📁 Estructura del Proyecto

```
📁 Redes-Complejas/
├── 📁 src/                           # Código fuente principal
│   ├── 📁 core/                     # Funcionalidades principales
│   ├── 📁 ui/                       # Interfaz de usuario
│   │   ├── interfaz_principal.py    # Interfaz principal
│   │   ├── exploracion.py           # Módulo de exploración
│   │   ├── exploracion_autores.py   # Análisis de autores
│   │   ├── exploracion_instituciones.py # Análisis de instituciones
│   │   ├── exploracion_articulos.py # Análisis de artículos
│   │   ├── exploracion_analisis_general.py # Análisis general
│   │   └── intro.py                 # Página de inicio
│   ├── 📁 analysis/                 # Análisis y métricas
│   │   ├── DataScience.py           # Análisis de datos científicos
│   │   ├── DataScience_analisis_general.py # Análisis general
│   │   └── Metricas.py              # Métricas de redes
│   ├── 📁 visualization/            # Visualización de grafos
│   │   ├── graphs.py                # Construcción de grafos
│   │   └── graphs_render.py         # Renderizado de grafos
│   └── 📁 utils/                    # Utilidades
│       └── load_data.py             # Carga de datos
├── 📁 data/                         # Archivos de datos
│   └── Json_Bien_Referenciados_normalizado.json
├── 📁 config/                       # Configuraciones
│   └── settings.py                  # Configuraciones del sistema
├── 📁 lib/                          # Librerías externas (si las hay)
├── 📁 logic/                        # Lógica de negocio adicional
├── 📁 webapp/                       # Aplicación web adicional
├── main.py                          # Archivo principal de ejecución
├── interfaz.py                      # Interfaz original (legacy)
├── streamlit.py                     # Aplicación Streamlit adicional
└── README.md                        # Este archivo
```

## 🚀 Cómo Ejecutar

### Opción 1: Nueva Estructura (Recomendada)
```bash
python main.py
```

### Opción 2: Interfaz Original (Legacy)
```bash
streamlit run interfaz.py
```

## 📊 Funcionalidades

### 🤝 Redes de Colaboración
- **Colaboración entre Investigadores**: Análisis de redes autor-autor
- **Jerarquías de Autoría**: Análisis de roles principales y secundarios

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

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Autores

- **Tu Nombre** - *Desarrollo inicial* - [TuGitHub](https://github.com/tuusuario)

## 🙏 Agradecimientos

- A la comunidad de NetworkX por las herramientas de análisis de redes
- A Streamlit por facilitar el desarrollo de aplicaciones web
- A todos los contribuidores del proyecto