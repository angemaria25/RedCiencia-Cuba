

import streamlit as st
from ..analysis import Metricas
from ..analysis.DataScience_analisis_general import texto_analisis_general
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import pandas as pd

def crear_tarjeta_metrica(titulo, valor, descripcion, icono="📊", color="#1f77b4"):
    """Crea una tarjeta visual para mostrar una métrica"""
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}05);
        border: 1px solid {color}30;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 24px; margin-right: 10px;">{icono}</span>
            <h4 style="margin: 0; color: {color}; font-weight: 600;">{titulo}</h4>
        </div>
        <div style="font-size: 32px; font-weight: bold; color: #2c3e50; margin: 10px 0;">
            {valor}
        </div>
        <div style="font-size: 14px; color: #666; line-height: 1.4;">
            {descripcion}
        </div>
    </div>
    """

def crear_grafico_barras_colaboracion(resumen):
    """Crea gráfico de barras para métricas de colaboración"""
    fig = go.Figure()
    
    categorias = ['Autores por Artículo', 'Instituciones por Artículo', 'Colaboraciones Autores', 'Colaboraciones Instituciones']
    valores = [
        resumen.get('promedio_autores_por_articulo', 0),
        resumen.get('promedio_inst_por_articulo', 0),
        resumen.get('promedio_colab_autores', 0),
        resumen.get('promedio_colab_inst', 0)
    ]
    
    colores = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    fig.add_trace(go.Bar(
        x=categorias,
        y=valores,
        marker_color=colores,
        text=[f'{v:.2f}' for v in valores],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Métricas de Colaboración Promedio",
        xaxis_title="Tipo de Colaboración",
        yaxis_title="Promedio",
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def crear_grafico_distribucion(resumen):
    """Crea gráfico de distribución de artículos por autor/institución"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribución por Autores', 'Distribución por Instituciones'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Datos para autores
    max_art_aut = resumen.get('max_articulos_por_autor', 0)
    min_art_aut = resumen.get('min_articulos_por_autor', 0)
    prom_art_aut = resumen.get('promedio_articulos_por_autor', 0)
    
    fig.add_trace(
        go.Bar(x=['Mínimo', 'Promedio', 'Máximo'], 
               y=[min_art_aut, prom_art_aut, max_art_aut],
               name='Autores',
               marker_color='#3498db'),
        row=1, col=1
    )
    
    # Datos para instituciones
    max_art_inst = resumen.get('max_articulos_por_inst', 0)
    min_art_inst = resumen.get('min_articulos_por_inst', 0)
    prom_art_inst = resumen.get('promedio_articulos_por_inst', 0)
    
    fig.add_trace(
        go.Bar(x=['Mínimo', 'Promedio', 'Máximo'], 
               y=[min_art_inst, prom_art_inst, max_art_inst],
               name='Instituciones',
               marker_color='#e74c3c'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Distribución de Artículos",
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def crear_nube_palabras_interactiva(top_palabras):
    """Crea una nube de palabras interactiva"""
    if not top_palabras:
        return None
    
    # Preparar datos para el gráfico de burbujas
    palabras = []
    scores = []
    for palabra, score in top_palabras[:10]:  # Top 10
        palabras.append(palabra.title())
        scores.append(float(score) if isinstance(score, (int, float)) else 1.0)
    
    # Normalizar scores para el tamaño de las burbujas
    max_score = max(scores) if scores else 1
    sizes = [30 + (score/max_score) * 50 for score in scores]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.random.uniform(0, 10, len(palabras)),
        y=np.random.uniform(0, 10, len(palabras)),
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=scores,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Relevancia")
        ),
        text=palabras,
        textposition="middle center",
        textfont=dict(size=12, color="white"),
        hovertemplate='<b>%{text}</b><br>Relevancia: %{marker.color:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Palabras Clave Más Relevantes",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def mostrar_top_por_campo(top_por_campo):
    """Muestra las palabras clave top por campo de estudio"""
    if not top_por_campo:
        return
    
    st.subheader("�� Temas por Campo de Estudio")
    
    # Crear columnas para mostrar los campos
    campos = list(top_por_campo.keys())
    if len(campos) <= 2:
        cols = st.columns(len(campos))
    else:
        cols = st.columns(3)
    
    for i, campo in enumerate(campos):
        with cols[i % len(cols)]:
            palabras_campo = top_por_campo[campo]
            if palabras_campo:
                # Crear una tarjeta para cada campo
                palabras_texto = ", ".join([p[0].title() for p in palabras_campo[:3]])
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 5px 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin: 0 0 10px 0; color: white;">{campo}</h4>
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">{palabras_texto}</p>
                </div>
                """, unsafe_allow_html=True)

def analisis_general_tab(articulos):
    st.subheader("📊 Análisis General de la Red Científica")
    
    # Obtener métricas
    resumen = Metricas.resumen_global_red(articulos)
    
    # Sección 1: Métricas principales en tarjetas
    st.markdown("### 📈 Métricas Principales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(crear_tarjeta_metrica(
            "Total de Artículos",
            f"{resumen.get('num_articulos', 0):,}",
            "Número total de publicaciones científicas analizadas",
            "📄",
            "#3498db"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(crear_tarjeta_metrica(
            "Autores Únicos",
            f"{resumen.get('num_autores', 0):,}",
            "Investigadores que participaron en las publicaciones",
            "👥",
            "#e74c3c"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(crear_tarjeta_metrica(
            "Instituciones",
            f"{resumen.get('num_instituciones', 0):,}",
            "Organizaciones involucradas en la investigación",
            "🏛️",
            "#2ecc71"
        ), unsafe_allow_html=True)
    
    # Sección 2: Métricas de colaboración
    st.markdown("### 🤝 Análisis de Colaboración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(crear_grafico_barras_colaboracion(resumen), use_container_width=True)
    
    with col2:
        st.plotly_chart(crear_grafico_distribucion(resumen), use_container_width=True)
    
    # Sección 3: Análisis de temas
    st.markdown("### 🔍 Análisis Temático")
    
    top_palabras = resumen.get('top_palabras', [])
    if top_palabras:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_nube = crear_nube_palabras_interactiva(top_palabras)
            if fig_nube:
                st.plotly_chart(fig_nube, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏆 Top Palabras Clave")
            for i, (palabra, score) in enumerate(top_palabras[:5], 1):
                score_normalizado = float(score) if isinstance(score, (int, float)) else 1.0
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #f39c12{int(score_normalizado*100):02d}, transparent);
                    padding: 8px 12px;
                    margin: 5px 0;
                    border-radius: 5px;
                    border-left: 4px solid #f39c12;
                ">
                    <strong>{i}. {palabra.title()}</strong><br>
                    <small>Relevancia: {score_normalizado:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Sección 4: Temas por campo
    top_por_campo = resumen.get('top_por_campo', {})
    if top_por_campo:
        mostrar_top_por_campo(top_por_campo)
    
    # Sección 5: Resumen ejecutivo
    st.markdown("### 📋 Resumen Ejecutivo")
    
    # Crear un resumen visual con métricas clave
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(crear_tarjeta_metrica(
            "Colaboración Promedio",
            f"{resumen.get('promedio_autores_por_articulo', 0):.1f}",
            "autores por artículo en promedio",
            "🔗",
            "#9b59b6"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(crear_tarjeta_metrica(
            "Productividad",
            f"{resumen.get('promedio_articulos_por_autor', 0):.1f}",
            "artículos por autor en promedio",
            "⚡",
            "#1abc9c"
        ), unsafe_allow_html=True)
    
    # Mensaje de navegación
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <h4 style="margin: 0 0 10px 0; color: white;">🚀 Explora Más</h4>
        <p style="margin: 0; opacity: 0.9;">
            Para un análisis más profundo de la red científica, explora las demás pestañas de este panel.
            Cada sección ofrece perspectivas únicas sobre la estructura y dinámicas de colaboración.
        </p>
    </div>
    """, unsafe_allow_html=True)
