import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd

def crear_seccion_metricas_visuales(G, tipo_red, autores_sin_colab=[]):
    """Crea una sección visual interactiva para mostrar las métricas de la red en lugar de texto plano"""
    
    if G is None or G.number_of_nodes() == 0:
        st.warning("No hay datos suficientes para mostrar métricas.")
        return
    
    # Obtener métricas básicas
    num_nodos = G.number_of_nodes()
    num_aristas = G.number_of_edges()
    grado_promedio = sum(dict(G.degree()).values()) / num_nodos if num_nodos > 0 else 0
    
    # Formatear números como enteros sin separadores de miles
    num_nodos_fmt = f"{num_nodos}"
    num_aristas_fmt = f"{num_aristas}"
    
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
    
    # Explicación inicial para usuarios
    st.info("""
    Esta sección muestra cómo los investigadores se conectan y colaboran entre sí. 
    Cada número representa una característica importante de la red de colaboración científica.
    """)
    
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
                {num_nodos_fmt}
            </div>
            <div style="font-size: 18px; opacity: 0.9;">
                Total de Investigadores
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Número total de autores que aparecen en las publicaciones analizadas")
    
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
                {num_aristas_fmt}
            </div>
            <div style="font-size: 18px; opacity: 0.9;">
                Colaboraciones Totales
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Número de veces que dos investigadores han trabajado juntos en una publicación")
    
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
                Colaboradores Promedio
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("En promedio, cada investigador colabora con este número de colegas")
    
    # Sección 2: Top autores más activos
    if top_autores:
        st.subheader("Investigadores Más Activos")
        st.info("Estos son los investigadores que más colaboraciones han establecido con otros colegas.")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (autor, colaboraciones) in enumerate(top_autores):
            colores = ["#FFD700", "#C0C0C0", "#CD7F32"]  # Oro, Plata, Bronce
            iconos = ["🥇", "🥈", "🥉"]
            posiciones = ["1er lugar", "2do lugar", "3er lugar"]
            
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
                        colaboraciones ({posiciones[i]})
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Información de conectividad (sin gráfico)
        tamano_mayor_fmt = f"{tamano_mayor}"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #a8edea, #fed6e3);
            padding: 20px;
            border-radius: 12px;
            margin: 10px 0;
        ">
            <h4 style="margin: 0 0 15px 0; color: #2c3e50;">Resumen de Conectividad</h4>
            <div style="margin: 10px 0;">
                <strong>Total de grupos:</strong> {num_componentes}
            </div>
            <div style="margin: 10px 0;">
                <strong>Grupo más grande:</strong> {tamano_mayor_fmt} autores
            </div>
            <div style="margin: 10px 0;">
                <strong>Cobertura del grupo principal:</strong> {(tamano_mayor/num_nodos*100):.1f}% de la red
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
                <h5 style="margin: 0 0 10px 0; color: #2c3e50;">Investigadores Sin Colaboraciones</h5>
                <div style="color: #2c3e50;">
                    <strong>{len(autores_sin_colab)}</strong> autores aparecen como únicos autores en sus publicaciones
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Estos investigadores representan oportunidades para fomentar futuras colaboraciones.")
    
    with col2:
        # Análisis de influencia (solo para redes de colaboración) - SIN GRÁFICO
        if tipo_red == "Red de Colaboración Autor-Autor" and top_autores:
            st.markdown("**Investigadores Conectores**")
            
            # Calcular centralidad de intermediación para los top autores
            try:
                betweenness = nx.betweenness_centrality(G)
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
                
                if top_betweenness:
                    st.markdown("**Top 5 - Investigadores que Más Conectan Grupos:**")
                    
                    for i, (autor, valor) in enumerate(top_betweenness, 1):
                        st.markdown(f"{i}. **{autor}**")
                    
                    # Explicación del resultado (sin el contenedor que se quiere eliminar)
                    if top_betweenness[0][1] > 0.1:
                        st.success(f"**{top_betweenness[0][0]}** es un conector clave en la red.")
                        
            except:
                st.warning("No se pudo calcular el análisis de conectores para esta red.")
    
    # Sección 4: Resumen ejecutivo visual (SIN el contenedor que se quiere eliminar)
    if tipo_red == "Red de Colaboración Autor-Autor":
        # Formatear números para el texto del insight
        tamano_mayor_texto = f"{tamano_mayor}"
        
        insight_text = f"""Esta red de colaboración científica conecta a **{num_nodos_fmt} investigadores** a través de **{num_aristas_fmt} colaboraciones**. La red muestra un patrón de colaboración donde cada autor colabora en promedio con **{grado_promedio:.1f} colegas**."""
        
        if top_autores:
            autor_top = top_autores[0][0]
            colabs_top = top_autores[0][1]
            insight_text += f" **{autor_top}** lidera la red con **{colabs_top} colaboraciones**, estableciéndose como un investigador central en el ecosistema científico."
        
        if num_componentes > 1:
            insight_text += f" La red se organiza en **{num_componentes} grupos independientes**, donde el grupo principal incluye **{tamano_mayor_texto} autores** ({(tamano_mayor/num_nodos*100):.1f}% de la red)."
        else:
            insight_text += " La red está **completamente conectada**, lo que facilita el flujo de información entre todos los investigadores."
        
        if autores_sin_colab:
            insight_text += f" Adicionalmente, **{len(autores_sin_colab)} autores** no han establecido colaboraciones registradas, representando oportunidades de integración futura."
    
    else:
        insight_text = f"""Esta red {tipo_red.lower()} conecta a **{num_nodos_fmt} entidades** a través de **{num_aristas_fmt} relaciones**, mostrando un patrón de conectividad promedio de **{grado_promedio:.1f} conexiones por nodo**."""
    
    # Mostrar insights como texto markdown normal sin HTML
    st.markdown("#### Conclusiones Principales")
    st.markdown(insight_text)