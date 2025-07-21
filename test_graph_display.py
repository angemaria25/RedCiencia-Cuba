#!/usr/bin/env python3
"""
Test script to verify graph display functionality
"""

import streamlit as st
import networkx as nx
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.visualization.graphs_render import show_networkx_graph

def test_simple_graph():
    """Test with a simple graph"""
    st.title("Test de Visualización de Grafos")
    
    # Create a simple test graph
    G = nx.Graph()
    G.add_edge("Nodo A", "Nodo B", weight=1)
    G.add_edge("Nodo B", "Nodo C", weight=2)
    G.add_edge("Nodo C", "Nodo A", weight=3)
    
    st.subheader("Grafo de Prueba Simple")
    st.write(f"Nodos: {G.number_of_nodes()}")
    st.write(f"Aristas: {G.number_of_edges()}")
    
    # Try to display the graph
    try:
        show_networkx_graph(G, height=400, width=600)
        st.success("✅ El grafo se mostró correctamente")
    except Exception as e:
        st.error(f"❌ Error al mostrar el grafo: {str(e)}")
        st.exception(e)

def test_with_node_attributes():
    """Test with node attributes"""
    st.subheader("Grafo con Atributos de Nodos")
    
    G = nx.Graph()
    G.add_node("Autor 1", node_type='author', color='#4A90E2')
    G.add_node("Autor 2", node_type='author', color='#4A90E2')
    G.add_node("Institución A", node_type='institution', color='#FF3C3C')
    
    G.add_edge("Autor 1", "Autor 2", weight=2)
    G.add_edge("Autor 1", "Institución A", weight=1)
    G.add_edge("Autor 2", "Institución A", weight=1)
    
    st.write(f"Nodos: {G.number_of_nodes()}")
    st.write(f"Aristas: {G.number_of_edges()}")
    
    try:
        show_networkx_graph(G, height=400, width=600)
        st.success("✅ El grafo con atributos se mostró correctamente")
    except Exception as e:
        st.error(f"❌ Error al mostrar el grafo con atributos: {str(e)}")
        st.exception(e)

def check_dependencies():
    """Check if all required dependencies are available"""
    st.subheader("Verificación de Dependencias")
    
    dependencies = {
        'streamlit': st,
        'networkx': nx,
        'pyvis': None,
        'pandas': None,
        'numpy': None
    }
    
    try:
        from pyvis.network import Network
        dependencies['pyvis'] = Network
        st.success("✅ pyvis disponible")
    except ImportError as e:
        st.error(f"❌ pyvis no disponible: {e}")
    
    try:
        import pandas as pd
        dependencies['pandas'] = pd
        st.success("✅ pandas disponible")
    except ImportError as e:
        st.error(f"❌ pandas no disponible: {e}")
    
    try:
        import numpy as np
        dependencies['numpy'] = np
        st.success("✅ numpy disponible")
    except ImportError as e:
        st.error(f"❌ numpy no disponible: {e}")
    
    # Check if webapp directory exists
    webapp_dir = os.path.join(os.getcwd(), 'webapp', 'data')
    if os.path.exists(webapp_dir):
        st.success(f"✅ Directorio webapp existe: {webapp_dir}")
    else:
        st.warning(f"⚠️ Directorio webapp no existe: {webapp_dir}")
        try:
            os.makedirs(webapp_dir, exist_ok=True)
            st.success(f"✅ Directorio webapp creado: {webapp_dir}")
        except Exception as e:
            st.error(f"❌ No se pudo crear directorio webapp: {e}")

if __name__ == "__main__":
    st.set_page_config(page_title="Test Grafos", layout="wide")
    
    check_dependencies()
    st.markdown("---")
    test_simple_graph()
    st.markdown("---")
    test_with_node_attributes()