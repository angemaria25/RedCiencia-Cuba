"""
Interfaz Principal del Sistema de Análisis de Redes Científicas
==============================================================

Módulo principal que maneja la navegación y el flujo de la aplicación.
"""

import streamlit as st
import json
import os
import networkx as nx
import numpy as np
from pyvis.network import Network

# Importaciones locales
from ..utils.load_data import load_data_section
from .intro import show_intro
from ..visualization.graphs import (
    build_coauthor_graph,
    build_institution_institution_graph,
    build_principal_secondary_graph,
    build_author_citation_graph,
    build_paper_author_graph,
    build_institution_author_graph,
    build_field_institution_graph,
    build_field_author_graph,
    build_keyword_field_graph,
    build_paper_field_graph
)
from .exploracion import exploracion_section

def load_json(path):
    """Carga un archivo JSON desde la ruta especificada."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def show_networkx_graph(G, height=600, width=900):
    """Función auxiliar para mostrar grafos de NetworkX (mantenida por compatibilidad)."""
    try:
        if G.number_of_nodes() <= 100:
            pos = nx.kamada_kawai_layout(G, scale=1500)
        else:
            k_value = 2/np.sqrt(G.number_of_nodes())
            pos = nx.spring_layout(G, k=k_value, iterations=50, scale=1500)
    except:
        try:
            largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
            if len(largest_cc) < G.number_of_nodes():
                pos = {}
                components = list(nx.connected_components(G.to_undirected()))
                angle_step = 2 * np.pi / len(components)
                for i, component in enumerate(components):
                    subgraph = G.subgraph(component)
                    if len(component) > 3:
                        sub_pos = nx.spring_layout(subgraph, k=2, iterations=50)
                    else:
                        sub_pos = nx.circular_layout(subgraph)
                    center_x = 1200 * np.cos(i * angle_step)
                    center_y = 1200 * np.sin(i * angle_step)
                    for node, (x, y) in sub_pos.items():
                        pos[node] = (center_x + x * 500, center_y + y * 500)
            else:
                pos = nx.spring_layout(G, k=4, iterations=50, scale=1500)
        except:
            pos = nx.random_layout(G, scale=1500)
    
    for node, (x, y) in pos.items():
        G.nodes[node]['x'] = x
        G.nodes[node]['y'] = y
        G.nodes[node]['title'] = str(node)
        if 'color' not in G.nodes[node]:
            node_type = G.nodes[node].get('node_type', 'default')
            if node_type == 'paper':
                G.nodes[node]['color'] = '#FF6B6B'
            elif node_type == 'field':
                G.nodes[node]['color'] = '#FF6B6B'
            elif node_type == 'institution':
                pass

def main_interface():
    """Función principal de la interfaz de usuario."""
    
    # Configuración del sidebar
    with st.sidebar:
        st.markdown("### Menú")
        selected_tab = st.selectbox(
            "Selecciona una sección:", 
            ["Inicio", "Cargar Datos", "Análisis de Redes"], 
            index=0
        )

    # Inicializar estado de la sesión
    if 'articulos' not in st.session_state:
        st.session_state.articulos = None

    # Navegación principal
    if selected_tab == "Inicio":
        show_intro()
    elif selected_tab == "Cargar Datos":
        articulos = load_data_section()
        if articulos:
            st.session_state.articulos = articulos
    elif selected_tab == "Análisis de Redes":
        articulos = st.session_state.articulos
        if articulos:
            exploracion_section(articulos)
        else:
            st.info("Por favor, cargue un archivo JSON para comenzar la exploración.")