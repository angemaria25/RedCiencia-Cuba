# RedCiencia Cuba - Plataforma de Ciencia de Datos

## 🚀 Nueva Funcionalidad: Exploración Interactiva de Datos Científicos

Esta aplicación ahora incluye herramientas avanzadas de ciencia de datos que permiten explorar y analizar los datos científicos cubanos de manera interactiva, respondiendo preguntas específicas sobre autores, instituciones, colaboraciones y temáticas de investigación.

## 📊 Funcionalidades Principales

### 1. **Explorador de Datos Científicos** 🔍
Permite explorar los datos de manera interactiva para responder preguntas como:

#### 🧑‍🔬 Análisis de Autores
- **Top autores más productivos**: Identifica quiénes publican más papers
- **Información detallada por autor**: Instituciones, temáticas, número de publicaciones
- **Perfil completo**: Lista de todos los papers de un autor específico
- **Diversidad de investigación**: Cuántas temáticas diferentes aborda cada autor

#### 🏛️ Análisis de Instituciones  
- **Ranking de instituciones**: Las más productivas en términos de publicaciones
- **Diversidad institucional**: Relación entre productividad y diversidad temática
- **Especialización**: Principales temáticas y autores por institución
- **Análisis comparativo**: Visualización de productividad vs diversidad

#### 🤝 Análisis de Colaboraciones
- **Top colaboraciones**: Parejas de autores que más colaboran
- **Patrones por temática**: En qué áreas hay más colaboración
- **Redes de colaboración**: Identificación de grupos de investigación

#### 📚 Análisis de Temáticas
- **Áreas más investigadas**: Ranking de temáticas por número de papers
- **Especialistas por área**: Principales autores e instituciones en cada temática
- **Palabras clave**: Análisis de frecuencia de términos por temática
- **Perfil temático**: Información completa de cada área de investigación

#### 🔍 Búsqueda Específica
- **Búsqueda por autor**: Encuentra todos los papers de un investigador
- **Búsqueda por institución**: Papers asociados a una institución específica
- **Búsqueda por palabra clave**: Investigaciones relacionadas con términos específicos
- **Búsqueda por título**: Localiza papers por contenido del título

### 2. **Análisis Avanzado de Ciencia de Datos** 📈

#### 🎯 Clustering de Autores por Similitud
- **Agrupamiento automático**: Identifica grupos de autores con intereses similares
- **Análisis TF-IDF**: Basado en palabras clave y temáticas de investigación
- **Visualización PCA**: Representación gráfica de los clusters
- **Caracterización de clusters**: Palabras clave más comunes por grupo

#### 🌐 Análisis de Centralidad Avanzado
- **Múltiples métricas**: Grado, intermediación, cercanía, eigenvector
- **Análisis comparativo**: Correlaciones entre diferentes métricas de centralidad
- **Visualizaciones interactivas**: Gráficos de dispersión con líneas de tendencia
- **Rankings especializados**: Top autores según cada métrica

#### 📈 Métricas de Impacto y Productividad
- **Análisis de productividad**: Distribución de papers por autor
- **Tasa de colaboración**: Proporción de trabajos colaborativos vs individuales
- **Diversidad temática**: Cuántas áreas diferentes aborda cada autor
- **Métricas derivadas**: Promedio de coautores, diversidad institucional
- **Visualizaciones avanzadas**: Histogramas y gráficos de dispersión

## 🛠️ Instalación y Uso

### Requisitos
```bash
pip install -r requirements.txt
```

### Ejecutar la aplicación
```bash
streamlit run app.py
```

## 🎯 Casos de Uso Específicos

### Para Investigadores
- **"¿Quiénes son los expertos en mi área?"** → Análisis de Temáticas
- **"¿Con quién podría colaborar?"** → Análisis de Colaboraciones + Clustering
- **"¿Qué instituciones trabajan en mi tema?"** → Análisis de Instituciones por temática

### Para Administradores de Ciencia
- **"¿Cuáles son las fortalezas de investigación de Cuba?"** → Análisis de Temáticas + Instituciones
- **"¿Qué tan colaborativa es nuestra comunidad científica?"** → Análisis de Colaboraciones + Métricas de Productividad
- **"¿Quiénes son los investigadores más influyentes?"** → Análisis de Centralidad

### Para Estudiantes
- **"¿Quién investiga sobre X tema?"** → Búsqueda Específica + Análisis de Temáticas
- **"¿Qué instituciones son líderes en mi área de interés?"** → Análisis de Instituciones
- **"¿Cuáles son las tendencias de investigación?"** → Análisis de Temáticas + Palabras Clave

## 📊 Tipos de Preguntas que Puedes Responder

### Sobre Autores
- ¿Quién es el autor más productivo en matemáticas?
- ¿En qué instituciones trabaja un autor específico?
- ¿Cuántas temáticas diferentes aborda cada investigador?
- ¿Quiénes son los autores más colaborativos?

### Sobre Instituciones  
- ¿Cuál es la institución más productiva?
- ¿Qué institución tiene mayor diversidad temática?
- ¿En qué se especializa la Universidad de La Habana?
- ¿Qué instituciones colaboran más entre sí?

### Sobre Colaboraciones
- ¿Qué parejas de autores colaboran más frecuentemente?
- ¿En qué temáticas hay más colaboración?
- ¿Existen grupos de investigación claramente definidos?
- ¿Cómo se distribuye la colaboración por institución?

### Sobre Temáticas
- ¿Cuáles son las áreas de investigación más activas?
- ¿Quiénes son los especialistas en cada área?
- ¿Qué palabras clave son más frecuentes por temática?
- ¿Qué instituciones lideran cada área de investigación?

## 🔧 Funcionalidades Técnicas

### Filtros Interactivos
- **Por temática**: Enfócate en áreas específicas de investigación
- **Por institución**: Analiza la producción de instituciones específicas
- **Por colaboraciones mínimas**: Ajusta el umbral de colaboración en las redes
- **Por tamaño de red**: Controla la complejidad de las visualizaciones

### Visualizaciones Avanzadas
- **Gráficos de barras interactivos**: Rankings y comparaciones
- **Gráficos de dispersión**: Relaciones entre variables
- **Matrices de correlación**: Análisis de relaciones entre métricas
- **Redes interactivas**: Exploración visual de conexiones
- **Histogramas**: Distribuciones de variables

### Análisis Estadístico
- **Clustering K-means**: Agrupamiento automático de autores
- **Análisis PCA**: Reducción de dimensionalidad para visualización
- **TF-IDF**: Análisis de similitud textual
- **Métricas de centralidad**: Análisis de redes complejas
- **Estadísticas descriptivas**: Resúmenes y agregaciones

## 🎓 Valor Educativo

Esta herramienta es perfecta para:
- **Enseñanza de ciencia de datos**: Casos reales con datos científicos
- **Análisis de redes**: Aplicación práctica de teoría de grafos
- **Investigación bibliométrica**: Análisis de patrones de publicación
- **Toma de decisiones**: Basada en evidencia científica

## 🚀 Próximas Mejoras

- Análisis temporal de tendencias
- Predicción de colaboraciones futuras
- Análisis de impacto por citaciones
- Integración con bases de datos externas
- Exportación de reportes automáticos

---

**¡Explora, descubre y analiza la ciencia cubana como nunca antes!** 🇨🇺🔬📊