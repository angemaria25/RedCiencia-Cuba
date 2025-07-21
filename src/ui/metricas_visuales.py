import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from ..analysis.Metricas import (
    basic_graph_metrics, 
    degree_stats, 
    clustering_metrics, 
    weight_stats,
    directed_metrics,
    distance_metrics,
    centralities,
    assortativity
)

def crear_dashboard_metricas_grafo(G, titulo="Métricas del Grafo"):
    """Crea un dashboard visual completo para las métricas de un grafo"""
    
    st.subheader(f"📊 {titulo}")
    
    if G is None or G.number_of_nodes() == 0:
        st.warning("No hay datos suficientes para mostrar métricas del grafo.")
        return
    
    # Obtener todas las métricas
    basic_metrics = basic_graph_metrics(G)
    degree_metrics = degree_stats(G)
    cluster_metrics = clustering_metrics(G)
    weight_metrics = weight_stats(G)
    directed_metrics_data = directed_metrics(G) if G.is_directed() else None
    distance_metrics_data = distance_metrics(G)
    centrality_data = centralities(G)
    assortativity_coeff = assortativity(G)
    

def crear_tarjeta_metrica_simple(titulo, valor, icono, color):
    """Crea una tarjeta simple para métricas básicas"""
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}05);
        border: 1px solid {color}30;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 24px; margin-bottom: 5px;">{icono}</div>
        <div style="font-size: 24px; font-weight: bold; color: {color}; margin: 5px 0;">
            {valor}
        </div>
        <div style="font-size: 12px; color: #666; font-weight: 500;">
            {titulo}
        </div>
    </div>
    """

def crear_tarjeta_metrica_avanzada(titulo, valor, descripcion, icono, color):
    """Crea una tarjeta para métricas avanzadas con descripción"""
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}05);
        border: 1px solid {color}30;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 20px; margin-right: 8px;">{icono}</span>
            <h5 style="margin: 0; color: {color}; font-weight: 600;">{titulo}</h5>
        </div>
        <div style="font-size: 24px; font-weight: bold; color: #2c3e50; margin: 8px 0;">
            {valor}
        </div>
        <div style="font-size: 11px; color: #666; line-height: 1.3;">
            {descripcion}
        </div>
    </div>
    """

def crear_grafico_distribucion_grados(G):
    """Crea un gráfico de la distribución de grados"""
    degrees = [d for n, d in G.degree()]
    
    if not degrees:
        return go.Figure().add_annotation(text="No hay datos de grados", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    # Crear histograma
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=degrees,
        nbinsx=min(20, len(set(degrees))),
        marker_color='#3498db',
        opacity=0.7,
        name='Distribución de Grados'
    ))
    
    # Añadir línea de promedio
    mean_degree = np.mean(degrees)
    fig.add_vline(x=mean_degree, line_dash="dash", line_color="red",
                  annotation_text=f"Promedio: {mean_degree:.2f}")
    
    fig.update_layout(
        title="Distribución de Grados",
        xaxis_title="Grado",
        yaxis_title="Frecuencia",
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def crear_grafico_metricas_conectividad(basic_metrics, cluster_metrics, distance_metrics):
    """Crea un gráfico de barras con métricas de conectividad"""
    
    metricas = []
    valores = []
    colores = []
    
    # Densidad
    metricas.append("Densidad")
    valores.append(basic_metrics['densidad'])
    colores.append('#3498db')
    
    # Clustering promedio
    metricas.append("Clustering Promedio")
    valores.append(cluster_metrics['clustering_avg'])
    colores.append('#e74c3c')
    
    # Clustering global
    metricas.append("Clustering Global")
    valores.append(cluster_metrics['clustering_global'])
    colores.append('#2ecc71')
    
    # Conectividad (si está disponible)
    if distance_metrics and distance_metrics['is_connected']:
        metricas.append("Conectado")
        valores.append(1.0)
        colores.append('#f39c12')
    else:
        metricas.append("Desconectado")
        valores.append(0.0)
        colores.append('#95a5a6')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metricas,
        y=valores,
        marker_color=colores,
        text=[f'{v:.3f}' for v in valores],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Métricas de Conectividad",
        xaxis_title="Métrica",
        yaxis_title="Valor",
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def mostrar_top_centralidad(top_data, titulo, color):
    """Muestra el top de una centralidad específica"""
    if not top_data:
        st.info(f"No hay datos disponibles para {titulo}")
        return
    
    # Crear gráfico de barras horizontal
    nodos = [item[0] for item in top_data]
    valores = [item[1] for item in top_data]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=nodos,
        x=valores,
        orientation='h',
        marker_color=color,
        text=[f'{v:.3f}' for v in valores],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f"Top 3 - {titulo}",
        xaxis_title="Valor de Centralidad",
        yaxis_title="Nodo",
        height=250,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def mostrar_metricas_dirigidas(directed_data):
    """Muestra métricas específicas para grafos dirigidos"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(crear_tarjeta_metrica_simple(
            "Reciprocidad",
            f"{directed_data.get('reciprocidad', 0):.3f}" if directed_data.get('reciprocidad') is not None else "N/A",
            "🔄",
            "#9b59b6"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(crear_tarjeta_metrica_simple(
            "Comp. Fuertes",
            directed_data.get('scc', 0),
            "💪",
            "#1abc9c"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(crear_tarjeta_metrica_simple(
            "Fuentes/Sumideros",
            f"{directed_data.get('sources', 0)}/{directed_data.get('sinks', 0)}",
            "🚰",
            "#e67e22"
        ), unsafe_allow_html=True)

def crear_comparacion_grafos(grafos_dict):
    """Crea una comparación visual entre múltiples grafos"""
    
    st.subheader("📊 Comparación de Grafos")
    
    if not grafos_dict or len(grafos_dict) < 2:
        st.warning("Se necesitan al menos 2 grafos para hacer una comparación.")
        return
    
    # Recopilar métricas de todos los grafos
    datos_comparacion = []
    
    for nombre, grafo in grafos_dict.items():
        if grafo and grafo.number_of_nodes() > 0:
            basic = basic_graph_metrics(grafo)
            cluster = clustering_metrics(grafo)
            
            datos_comparacion.append({
                'Grafo': nombre,
                'Nodos': basic['nodos'],
                'Aristas': basic['aristas'],
                'Densidad': basic['densidad'],
                'Grado Promedio': basic['grado_medio'],
                'Clustering Global': cluster['clustering_global'],
                'Clustering Promedio': cluster['clustering_avg'],
                'Componentes': basic['componentes']
            })
    
    if not datos_comparacion:
        st.warning("No hay grafos válidos para comparar.")
        return
    
    df = pd.DataFrame(datos_comparacion)
    
    # Crear gráficos de comparación
    col1, col2 = st.columns(2)
    
    with col1:
        # Comparación de tamaño (nodos y aristas)
        fig_tamano = go.Figure()
        
        fig_tamano.add_trace(go.Bar(
            name='Nodos',
            x=df['Grafo'],
            y=df['Nodos'],
            marker_color='#3498db'
        ))
        
        fig_tamano.add_trace(go.Bar(
            name='Aristas',
            x=df['Grafo'],
            y=df['Aristas'],
            marker_color='#e74c3c'
        ))
        
        fig_tamano.update_layout(
            title="Comparación de Tamaño",
            xaxis_title="Grafo",
            yaxis_title="Cantidad",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig_tamano, use_container_width=True)
    
    with col2:
        # Comparación de métricas de conectividad
        fig_conectividad = go.Figure()
        
        fig_conectividad.add_trace(go.Scatter(
            x=df['Densidad'],
            y=df['Clustering Global'],
            mode='markers+text',
            text=df['Grafo'],
            textposition="top center",
            marker=dict(
                size=df['Grado Promedio'] * 5,  # Tamaño proporcional al grado promedio
                color=df['Componentes'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Componentes")
            ),
            hovertemplate='<b>%{text}</b><br>Densidad: %{x:.3f}<br>Clustering: %{y:.3f}<extra></extra>'
        ))
        
        fig_conectividad.update_layout(
            title="Densidad vs Clustering",
            xaxis_title="Densidad",
            yaxis_title="Clustering Global",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig_conectividad, use_container_width=True)
    
    # Tabla de comparación
    st.markdown("### 📋 Tabla de Comparación")
    st.dataframe(df.round(3), use_container_width=True)

def crear_seccion_metricas_visuales(G, tipo_red, autores_sin_colab=[]):
    """Crea una sección visual interactiva para mostrar las métricas de la red en lugar de texto plano"""
    
    
    if G is None or G.number_of_nodes() == 0:
        st.warning("No hay datos suficientes para mostrar métricas.")
        return
    
    # Obtener métricas básicas
    num_nodos = G.number_of_nodes()
    num_aristas = G.number_of_edges()
    grado_promedio = sum(dict(G.degree()).values()) / num_nodos if num_nodos > 0 else 0
    
    # Obtener top autores por grado
    grados = dict(G.degree())
    top_autores = sorted(grados.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Obtener número de componentes
    if G.is_directed():
        num_componentes = nx.number_weakly_connected_components(G)
        componentes = list(nx.weakly_connected_components(G))
    else:
        num_componentes = nx.number_connected_components(G)
        componentes = list(nx.connected_components(G))
    
    # Tamaño del componente más grande
    if componentes:
        componente_mayor = max(componentes, key=len)
        tamano_mayor = len(componente_mayor)
    else:
        tamano_mayor = 0
    
    # Sección 1: Métricas principales en tarjetas grandes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 10px 0;
        ">
            <div style="font-size: 48px; font-weight: bold; margin-bottom: 10px;">
                {num_nodos}
            </div>
            <div style="font-size: 18px; opacity: 0.9;">
                👥 Autores en la Red
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 10px 0;
        ">
            <div style="font-size: 48px; font-weight: bold; margin-bottom: 10px;">
                {num_aristas}
            </div>
            <div style="font-size: 18px; opacity: 0.9;">
                🔗 Colaboraciones
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 10px 0;
        ">
            <div style="font-size: 48px; font-weight: bold; margin-bottom: 10px;">
                {grado_promedio:.1f}
            </div>
            <div style="font-size: 18px; opacity: 0.9;">
                📊 Colaboraciones Promedio
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sección 2: Top autores más activos
    if top_autores:
        st.markdown("### 🏆 Autores Más Activos")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (autor, colaboraciones) in enumerate(top_autores):
            colores = ["#FFD700", "#C0C0C0", "#CD7F32"]  # Oro, Plata, Bronce
            iconos = ["🥇", "🥈", "🥉"]
            
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {colores[i]}20, {colores[i]}10);
                    border: 2px solid {colores[i]};
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    margin: 5px 0;
                ">
                    <div style="font-size: 32px; margin-bottom: 10px;">
                        {iconos[i]}
                    </div>
                    <div style="font-size: 24px; font-weight: bold; color: #2c3e50; margin-bottom: 8px;">
                        {colaboraciones}
                    </div>
                    <div style="font-size: 14px; color: #666; font-weight: 500; line-height: 1.3;">
                        {autor}
                    </div>
                    <div style="font-size: 12px; color: #888; margin-top: 5px;">
                        colaboraciones
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Sección 3: Estructura de la red
    st.markdown("### 🕸️ Estructura de la Red")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de componentes
        fig = go.Figure()
        
        if len(componentes) > 1:
            tamanos_componentes = [len(comp) for comp in componentes]
            tamanos_componentes.sort(reverse=True)
            
            fig.add_trace(go.Bar(
                x=[f"Componente {i+1}" for i in range(min(10, len(tamanos_componentes)))],
                y=tamanos_componentes[:10],
                marker_color='rgba(55, 128, 191, 0.7)',
                text=tamanos_componentes[:10],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Tamaño de Componentes",
                xaxis_title="Componente",
                yaxis_title="Número de Autores",
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
        else:
            fig.add_annotation(
                text="Red completamente conectada",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="green")
            )
            fig.update_layout(height=300)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Información de conectividad
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #a8edea, #fed6e3);
            padding: 20px;
            border-radius: 12px;
            margin: 10px 0;
        ">
            <h4 style="margin: 0 0 15px 0; color: #2c3e50;">🔗 Conectividad</h4>
            <div style="margin: 10px 0;">
                <strong>Componentes:</strong> {num_componentes}
            </div>
            <div style="margin: 10px 0;">
                <strong>Componente mayor:</strong> {tamano_mayor} autores
            </div>
            <div style="margin: 10px 0;">
                <strong>Cobertura:</strong> {(tamano_mayor/num_nodos*100):.1f}% de la red
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Información sobre autores sin colaboraciones
        if autores_sin_colab:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ffeaa7, #fab1a0);
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border-left: 4px solid #e17055;
            ">
                <h5 style="margin: 0 0 10px 0; color: #2c3e50;">⚠️ Autores Aislados</h5>
                <div style="color: #2c3e50;">
                    <strong>{len(autores_sin_colab)}</strong> autores sin colaboraciones registradas
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Sección 4: Análisis de influencia (solo para redes de colaboración)
    if tipo_red == "Red de Colaboración Autor-Autor" and top_autores:
        
        # Calcular centralidad de intermediación para los top autores
        try:
            betweenness = nx.betweenness_centrality(G)
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
            
            if top_betweenness:
                # Crear gráfico de influencia
                fig = go.Figure()
                
                autores_inf = [item[0] for item in top_betweenness]
                valores_inf = [item[1] for item in top_betweenness]
                
                fig.add_trace(go.Bar(
                    y=autores_inf,
                    x=valores_inf,
                    orientation='h',
                    marker_color='rgba(255, 99, 132, 0.7)',
                    text=[f'{v:.3f}' for v in valores_inf],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Top 5 - Autores con Mayor Influencia (Intermediación)",
                    xaxis_title="Centralidad de Intermediación",
                    yaxis_title="Autor",
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("No se pudo calcular la centralidad de intermediación para esta red.")
    
    # Crear un resumen visual con insights clave
    if tipo_red == "Red de Colaboración Autor-Autor":
        insight_text = f"""
        Esta red de colaboración científica conecta a **{num_nodos} investigadores** a través de 
        **{num_aristas} colaboraciones**. La red muestra un patrón de colaboración donde cada autor 
        colabora en promedio con **{grado_promedio:.1f} colegas**.
        """
        
        if top_autores:
            autor_top = top_autores[0][0]
            colabs_top = top_autores[0][1]
            insight_text += f" **{autor_top}** lidera la red con **{colabs_top} colaboraciones**, "
            insight_text += f"estableciéndose como un nodo central en el ecosistema científico."
        
        if num_componentes > 1:
            insight_text += f" La red se organiza en **{num_componentes} grupos independientes**, "
            insight_text += f"donde el grupo principal incluye **{tamano_mayor} autores** "
            insight_text += f"({(tamano_mayor/num_nodos*100):.1f}% de la red)."
        else:
            insight_text += " La red está **completamente conectada**, lo que facilita el flujo de información entre todos los investigadores."
        
        if autores_sin_colab:
            insight_text += f" Adicionalmente, **{len(autores_sin_colab)} autores** no han establecido colaboraciones registradas, "
            insight_text += "representando oportunidades de integración futura."
    
    else:
        insight_text = f"""
        Esta red {tipo_red.lower()} conecta **{num_nodos} entidades** a través de 
        **{num_aristas} relaciones**, mostrando un patrón de conectividad promedio de 
        **{grado_promedio:.1f} conexiones por nodo**.
        """
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    "<h4 style="margin: 0 0 15px 0; color: white;">🎯 Insights Clave</h4>
        <p style="margin: 0; line-height: 1.6; font-size: 16px; opacity: 0.95;">
            {insight_text}
        </p>
    </div>
    """, unsafe_allow_html=True)