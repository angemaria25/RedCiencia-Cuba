import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import community as community_louvain 
import numpy as np
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RedCiencia Cuba",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("RedCiencia Cuba")


@st.cache_data
def load_and_analyze_data():
    """Carga y analiza los datos de papers cubanos"""
    try:
        df = pd.read_csv('data_final_normalizado.csv')
        
        # Limpieza y preprocesamiento
        df['autores_normalizados'] = df['autores_normalizados'].fillna('').astype(str)
        df['afiliaciones_normalizadas'] = df['afiliaciones_normalizadas'].fillna('').astype(str)
        df['tematica'] = df['tematica'].fillna('Sin temática').astype(str)
        df['palabras_clave'] = df['palabras_clave'].fillna('').astype(str)
        df['titulo'] = df['titulo'].fillna('').astype(str)
        
        # Procesamiento de listas
        df['autores_list'] = df['autores_normalizados'].apply(
            lambda x: [a.strip() for a in x.split(',') if a.strip()]
        )
        df['afiliaciones_list'] = df['afiliaciones_normalizadas'].apply(
            lambda x: [i.strip() for i in x.split(',') if i.strip()]
        )
        df['palabras_clave_list'] = df['palabras_clave'].apply(
            lambda x: [k.strip() for k in x.split('|') if k.strip()]
        )
        
        # Filtrar papers sin autores
        df = df[df['autores_list'].apply(len) > 0]
        
        # Análisis exploratorio
        stats = {
            'total_papers': len(df),
            'total_autores': len(set([autor for autores_list in df['autores_list'] for autor in autores_list])),
            'total_instituciones': len(set([inst for afil_list in df['afiliaciones_list'] for inst in afil_list if inst])),
            'total_palabras_clave': len(set([kw for kw_list in df['palabras_clave_list'] for kw in kw_list if kw])),
        }
        
        return df, stats
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, None

def create_sidebar_filters(df):
    """Crea filtros funcionales en el sidebar"""
    
    st.sidebar.header("Filtros para Subgrafos")
    
    tematicas_disponibles = ['Todas'] + sorted([t for t in df['tematica'].unique() if t != 'Sin temática'])
    tematica_seleccionada = st.sidebar.selectbox(
        "📚 Filtrar por Temática:",
        tematicas_disponibles,
        help="Selecciona una temática específica para analizar"
    )
    
    todas_instituciones = set([inst for afil_list in df['afiliaciones_list'] for inst in afil_list if inst])
    # Filtrar instituciones que aparecen al menos 2 veces
    inst_counts = Counter([inst for afil_list in df['afiliaciones_list'] for inst in afil_list if inst])
    instituciones_frecuentes = [inst for inst, count in inst_counts.items() if count >= 2]
    instituciones_principales = ['Todas'] + sorted(instituciones_frecuentes)[:20]
    
    institucion_seleccionada = st.sidebar.selectbox(
        "🏫 Filtrar por Institución:",
        instituciones_principales,
        help="Selecciona una institución específica"
    )
    
    # Filtro por colaboración mínima
    min_colaboraciones = st.sidebar.slider(
        "🤝 Colaboraciones Mínimas:",
        min_value=1,
        max_value=5,
        value=1,
        help="Número mínimo de colaboraciones entre autores"
    )
    
    # Filtro por tamaño de red
    max_nodos_red = st.sidebar.slider(
        "Máximo Nodos en Visualización:",
        min_value=50,
        max_value=500,
        value=200,
        help="Limita el tamaño de la red para mejor rendimiento"
    )
    
    return {
        'tematica': tematica_seleccionada,
        'institucion': institucion_seleccionada,
        'min_colaboraciones': min_colaboraciones,
        'max_nodos': max_nodos_red
    }

def apply_filters(df, filters):
    """Aplica los filtros seleccionados al dataframe"""
    df_filtered = df.copy()
    
    if filters['tematica'] != 'Todas':
        df_filtered = df_filtered[df_filtered['tematica'] == filters['tematica']]
    
    if filters['institucion'] != 'Todas':
        mask = df_filtered['afiliaciones_list'].apply(
            lambda x: filters['institucion'] in x
        )
        df_filtered = df_filtered[mask]
    
    return df_filtered

@st.cache_data
def build_filtered_coauthorship_network(df_filtered, min_collaborations=1, max_nodes=200):
    """Construye red de coautoría con filtros aplicados"""
    G = nx.Graph()
    edge_weights = defaultdict(int)
    
    # Contar colaboraciones
    for autores_list in df_filtered['autores_list']:
        if len(autores_list) < 2:
            continue
        for i in range(len(autores_list)):
            for j in range(i + 1, len(autores_list)):
                edge = tuple(sorted([autores_list[i], autores_list[j]]))
                edge_weights[edge] += 1
    
    # Construir grafo con filtro de colaboraciones mínimas
    for (autor1, autor2), weight in edge_weights.items():
        if weight >= min_collaborations:
            G.add_edge(autor1, autor2, weight=weight)
    
    # Limitar nodos si es necesario
    if G.number_of_nodes() > max_nodes:
        # Seleccionar nodos más conectados
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        nodes_to_keep = [node for node, degree in top_nodes]
        G = G.subgraph(nodes_to_keep).copy()
    
    return G

@st.cache_data
def build_filtered_institution_network(df_filtered, min_shared_authors=1, max_nodes=100):
    """Construye red de colaboración institucional filtrada"""
    # Primero construir red bipartita
    B = nx.Graph()
    
    for _, row in df_filtered.iterrows():
        autores = row['autores_list']
        instituciones = row['afiliaciones_list']
        
        if not autores or not instituciones:
            continue
            
        for autor in autores:
            B.add_node(autor, bipartite=0, type='autor')
        for inst in instituciones:
            B.add_node(inst, bipartite=1, type='institucion')
            for autor in autores:
                B.add_edge(autor, inst)
    
    # Proyectar a red de instituciones
    instituciones = {n for n, d in B.nodes(data=True) if d.get('bipartite') == 1}
    
    if len(instituciones) < 2:
        return nx.Graph()
    
    projected = nx.bipartite.weighted_projected_graph(B, instituciones)
    
    # Filtrar por autores compartidos mínimos
    if min_shared_authors > 1:
        edges_to_remove = [(u, v) for u, v, d in projected.edges(data=True) 
                            if d.get('weight', 1) < min_shared_authors]
        projected.remove_edges_from(edges_to_remove)
        projected.remove_nodes_from(list(nx.isolates(projected)))
    
    # Limitar nodos
    if projected.number_of_nodes() > max_nodes:
        top_nodes = sorted(projected.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        nodes_to_keep = [node for node, degree in top_nodes]
        projected = projected.subgraph(nodes_to_keep).copy()
    
    return projected

@st.cache_data
def build_thematic_network(df_filtered, max_nodes=100):
    """Construye red bipartita Autor-Temática"""
    B = nx.Graph()
    
    for _, row in df_filtered.iterrows():
        autores = row['autores_list']
        tematica = row['tematica'].strip()
        
        if not autores or not tematica or tematica == 'Sin temática':
            continue
            
        # Agregar nodo de temática si no existe
        if tematica not in B:
            B.add_node(tematica, bipartite=1, type='tematica')
        
        # Agregar autores y enlaces
        for autor in autores:
            if autor not in B:
                B.add_node(autor, bipartite=0, type='autor')
            B.add_edge(autor, tematica)
    
    # Limitar nodos si es necesario
    if B.number_of_nodes() > max_nodes:
        # Mantener temáticas más conectadas y sus autores
        tematicas = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 1]
        tematica_degrees = [(t, B.degree(t)) for t in tematicas]
        top_tematicas = sorted(tematica_degrees, key=lambda x: x[1], reverse=True)[:10]
        
        nodes_to_keep = set([t for t, _ in top_tematicas])
        for tematica, _ in top_tematicas:
            neighbors = list(B.neighbors(tematica))
            nodes_to_keep.update(neighbors[:10])  # Top 10 autores por temática
        
        B = B.subgraph(nodes_to_keep).copy()
    
    return B

@st.cache_data
def build_keyword_network(df_filtered, max_nodes=150):
    """Construye red bipartita Autor-Palabra Clave"""
    B = nx.Graph()
    
    for _, row in df_filtered.iterrows():
        autores = row['autores_list']
        palabras_clave = row['palabras_clave_list']
        
        if not autores or not palabras_clave:
            continue
            
        # Agregar autores
        for autor in autores:
            if autor not in B:
                B.add_node(autor, bipartite=0, type='autor')
        
        # Agregar palabras clave y enlaces
        for kw in palabras_clave:
            if kw not in B:
                B.add_node(kw, bipartite=1, type='palabra_clave')
            for autor in autores:
                B.add_edge(autor, kw)
    
    # Limitar nodos manteniendo palabras clave más frecuentes
    if B.number_of_nodes() > max_nodes:
        keywords = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 1]
        keyword_degrees = [(kw, B.degree(kw)) for kw in keywords]
        top_keywords = sorted(keyword_degrees, key=lambda x: x[1], reverse=True)[:20]
        
        nodes_to_keep = set([kw for kw, _ in top_keywords])
        for keyword, _ in top_keywords:
            neighbors = list(B.neighbors(keyword))
            nodes_to_keep.update(neighbors[:8])  # Top 8 autores por palabra clave
        
        B = B.subgraph(nodes_to_keep).copy()
    
    return B

#newww 
@st.cache_data
def build_institution_thematic_network(df_filtered, max_nodes=150):
    """Construye red bipartita Institución-Temática"""
    B = nx.Graph()
    edge_weights = defaultdict(int)
    
    # Contar las conexiones Institución-Temática
    for _, row in df_filtered.iterrows():
        instituciones = row['afiliaciones_list']
        tematica = row['tematica'].strip()
        
        if not instituciones or not tematica or tematica == 'Sin temática':
            continue
            
        for inst in set(instituciones): # Usamos set() para no contar doble la misma institución en un solo paper
            edge_weights[(inst, tematica)] += 1
            
    # Construir el grafo con los pesos calculados
    for (inst, tematica), weight in edge_weights.items():
        # Agregar nodos con sus atributos
        if inst not in B:
            B.add_node(inst, bipartite=0, type='institucion')
        if tematica not in B:
            B.add_node(tematica, bipartite=1, type='tematica')
        
        # Agregar el enlace con su peso
        B.add_edge(inst, tematica, weight=weight)
        
    # Limitar nodos si es necesario (opcional pero recomendado)
    if B.number_of_nodes() > max_nodes:
        # Priorizar nodos con mayor peso total de sus conexiones
        node_weights = {node: B.degree(node, weight='weight') for node in B.nodes()}
        top_nodes_sorted = sorted(node_weights.items(), key=lambda item: item[1], reverse=True)
        
        nodes_to_keep = [node for node, weight in top_nodes_sorted[:max_nodes]]
        B = B.subgraph(nodes_to_keep).copy()
        # Eliminar nodos que quedaron aislados después del subgrafo
        B.remove_nodes_from(list(nx.isolates(B)))
        
    return B


def calculate_network_metrics(G, network_type="general"):
    """Calcula métricas específicas para cada red"""
    if not G.nodes():
        return {}
    
    metrics = {}
    
    # Métricas básicas
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Métricas avanzadas solo para redes no muy grandes
    if G.number_of_nodes() <= 300:
        metrics['avg_clustering'] = nx.average_clustering(G)
        
        # Componentes conectados
        components = list(nx.connected_components(G))
        metrics['num_components'] = len(components)
        metrics['largest_component_size'] = len(max(components, key=len)) if components else 0
        
        # Centralidades
        metrics['degree_centrality'] = nx.degree_centrality(G)
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        metrics['closeness_centrality'] = nx.closeness_centrality(G)
        
        # Eigenvector centrality para componente más grande
        if nx.is_connected(G):
            try:
                metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=500)
            except:
                metrics['eigenvector_centrality'] = {}
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            if len(largest_cc) > 1:
                subgraph = G.subgraph(largest_cc)
                try:
                    metrics['eigenvector_centrality'] = nx.eigenvector_centrality(subgraph, max_iter=500)
                except:
                    metrics['eigenvector_centrality'] = {}
        
        # Detección de comunidades
        if G.number_of_edges() > 0:
            try:
                partition = community_louvain.best_partition(G, random_state=42)
                metrics['communities'] = partition
                metrics['num_communities'] = len(set(partition.values()))
                metrics['modularity'] = community_louvain.modularity(partition, G)
            except:
                metrics['communities'] = {}
                metrics['num_communities'] = 0
                metrics['modularity'] = 0
    
    return metrics

def create_network_visualization(G, metrics, title="Red Científica"):
    """Crea visualización interactiva de la red"""
    if not G.nodes():
        st.info("No hay nodos para visualizar con los filtros actuales.")
        return
    
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#000000")
    net.toggle_physics(True)
    
    # Configuración de física optimizada
    net.set_options("""
    var options = {
        "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 95,
                "springConstant": 0.04,
                "damping": 0.09
            }
        }
    }
    """)
    
    # Colores para comunidades
    communities = metrics.get('communities', {})
    if communities:
        unique_communities = list(set(communities.values()))
        colors = px.colors.qualitative.Set3[:len(unique_communities)]
        color_map = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
    
    # Agregar nodos
    for node in G.nodes():
        # Tamaño basado en grado
        degree = G.degree(node)
        size = 10 + min(np.log(degree + 1) * 5, 30)
        
        # Color basado en tipo de nodo o comunidad
        node_data = G.nodes[node] if hasattr(G, 'nodes') and node in G.nodes else {}
        
        if 'bipartite' in node_data:
            # Red bipartita
            if node_data['bipartite'] == 0:
                if node_data.get('type') == 'institucion':
                    color = "#45B7D1"  # Azul para instituciones
                    label = f"{node[:25]}..." if len(node) > 25 else f"{node}"
                else:
                    color = "#FF6B6B"  # Rojo para autores
                    label = f"{node[:15]}..." if len(node) > 15 else f"{node}"
            else:
                if node_data.get('type') == 'tematica':
                    color = "#52CD4E"  # Verde  para temáticas
                    label = f"{node[:20]}..." if len(node) > 20 else f"{node}"
                else:
                    color = "#45B7D1"  # Azul para palabras clave
                    label = f"{node[:15]}..." if len(node) > 15 else f"{node}"
        else:
            # Red proyectada
            if communities and node in communities:
                color = color_map.get(communities[node], "#97C2FC")
                label = node[:20] + "..." if len(node) > 20 else node
            else:
                color = "#97C2FC"
                label = node[:20] + "..." if len(node) > 20 else node
        
        title = f"{node}<br>Grado: {degree}"
        if communities and node in communities:
            title += f"<br>Comunidad: {communities[node]}"
        
        net.add_node(node, label=label, size=size, color=color, title=title)
    
    # Agregar enlaces
    for source, target, data in G.edges(data=True):
        weight = data.get('weight', 1)
        width = min(0.5 + np.log(weight + 1) * 0.8, 8)
        title = f"Peso: {weight}" if weight > 1 else "Conexión"
        net.add_edge(source, target, width=width, title=title)
    
    # Guardar y mostrar
    try:
        path = "temp_network.html"
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()
        st.components.v1.html(html, height=600, scrolling=True)
        
        # Limpiar archivo temporal
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        st.error(f"Error en visualización: {e}")

def create_metrics_dashboard(metrics):
    """Crea dashboard de métricas de la red"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nodos", metrics.get('num_nodes', 0))
        if 'density' in metrics:
            st.metric("Densidad", f"{metrics['density']:.3f}")
    
    with col2:
        st.metric("Enlaces", metrics.get('num_edges', 0))
        if 'avg_clustering' in metrics:
            st.metric("Clustering Promedio", f"{metrics['avg_clustering']:.3f}")
    
    with col3:
        if 'num_components' in metrics:
            st.metric("Componentes", metrics['num_components'])
        if 'num_communities' in metrics:
            st.metric("Comunidades", metrics['num_communities'])
    
    with col4:
        if 'largest_component_size' in metrics:
            st.metric("Componente Mayor", metrics['largest_component_size'])
        if 'modularity' in metrics:
            st.metric("Modularidad", f"{metrics['modularity']:.3f}")

def create_single_centrality_plot(metrics, centrality_type, top_n):
    """Crea un solo gráfico de centralidad según la selección"""
    centrality_map = {
        'Grado': 'degree_centrality',
        'Intermediación': 'betweenness_centrality',
        'Cercanía': 'closeness_centrality',
        'Autovector': 'eigenvector_centrality'
    }
    
    metric_key = centrality_map.get(centrality_type)
    
    if not metric_key or metric_key not in metrics or not metrics[metric_key]:
        st.info(f"Métrica de {centrality_type} no disponible para esta red.")
        return
    
    # Obtener datos y ordenar
    centrality_data = metrics[metric_key]
    sorted_data = sorted(centrality_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not sorted_data:
        st.info(f"No hay datos de centralidad de {centrality_type}.")
        return
    
    # Crear gráfico
    fig = go.Figure(go.Bar(
        x=[item[1] for item in sorted_data],
        y=[item[0] for item in sorted_data],
        orientation='h',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} - Centralidad de {centrality_type}",
        xaxis_title="Valor de Centralidad",
        yaxis_title="Nodo",
        height=400 + top_n * 15,  # Altura dinámica
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def analyze_author_collaboration_patterns(df_filtered, G_coautor, metrics_coautor):
    """Análisis específico de patrones de colaboración entre autores"""
    st.subheader("🔍 Análisis de Datos: Patrones de Colaboración")
    
    # Crear análisis de autores más colaborativos
    autor_stats = {}
    for _, row in df_filtered.iterrows():
        autores = row['autores_list']
        if len(autores) > 1:  # Solo papers colaborativos
            for autor in autores:
                if autor not in autor_stats:
                    autor_stats[autor] = {
                        'colaboraciones': 0,
                        'coautores_unicos': set(),
                        'tematicas': set(),
                        'palabras_clave': [],
                        'instituciones': set()
                    }
                
                autor_stats[autor]['colaboraciones'] += 1
                autor_stats[autor]['coautores_unicos'].update([a for a in autores if a != autor])
                autor_stats[autor]['tematicas'].add(row['tematica'])
                autor_stats[autor]['palabras_clave'].extend(row['palabras_clave_list'])
                autor_stats[autor]['instituciones'].update(row['afiliaciones_list'])
    
    if autor_stats:
        # Convertir a DataFrame para análisis
        autor_df = []
        for autor, stats in autor_stats.items():
            autor_df.append({
                'Autor': autor,
                'Colaboraciones': stats['colaboraciones'],
                'Coautores_Unicos': len(stats['coautores_unicos']),
                'Diversidad_Tematica': len(stats['tematicas']),
                'Diversidad_Institucional': len([inst for inst in stats['instituciones'] if inst]),
                'Palabras_Clave': stats['palabras_clave'],
                'Tematicas': list(stats['tematicas'])
            })
        
        if autor_df:  # Verificar que hay datos
            autor_df = pd.DataFrame(autor_df)
            autor_df = autor_df.sort_values('Colaboraciones', ascending=False)
            
            col1, col2 = st.columns(2)
        else:
            st.warning("No hay suficientes datos de colaboración para realizar el análisis.")
            return
        
        with col1:
            st.write("**🏆 Top 10 Autores Más Colaborativos**")
            top_colaborativos = autor_df.head(10)
            
            fig = px.bar(
                top_colaborativos,
                x='Colaboraciones',
                y='Autor',
                orientation='h',
                title="Autores con Más Papers Colaborativos"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**🌐 Diversidad vs Colaboración**")
            
            fig_scatter = px.scatter(
                autor_df.head(20),
                x='Colaboraciones',
                y='Diversidad_Tematica',
                size='Coautores_Unicos',
                hover_name='Autor',
                title="Colaboración vs Diversidad Temática",
                labels={
                    'Colaboraciones': 'Número de Colaboraciones',
                    'Diversidad_Tematica': 'Número de Temáticas',
                    'Coautores_Unicos': 'Coautores Únicos'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Análisis detallado de autor seleccionado
        st.subheader("🔍 Análisis Detallado de Autor")
        autor_seleccionado = st.selectbox(
            "Selecciona un autor para análisis detallado:",
            [''] + list(autor_df['Autor'].values),
            key='autor_colab_analysis'
        )
        
        if autor_seleccionado:
            autor_info = autor_df[autor_df['Autor'] == autor_seleccionado].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🤝 Colaboraciones", autor_info['Colaboraciones'])
            with col2:
                st.metric("👥 Coautores Únicos", autor_info['Coautores_Unicos'])
            with col3:
                st.metric("📚 Temáticas", autor_info['Diversidad_Tematica'])
            with col4:
                st.metric("🏛️ Instituciones", autor_info['Diversidad_Institucional'])
            
            # Nube de palabras de especialización
            if autor_info['Palabras_Clave']:
                st.subheader(f"☁️ Especialización de {autor_seleccionado}")
                
                palabra_freq = Counter(autor_info['Palabras_Clave'])
                if len(palabra_freq) > 0:
                    try:
                        wordcloud = WordCloud(
                            width=800, 
                            height=300, 
                            background_color='white',
                            max_words=30,
                            colormap='viridis'
                        ).generate_from_frequencies(palabra_freq)
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    except:
                        st.write("**Principales áreas de especialización:**")
                        for palabra, freq in palabra_freq.most_common(15):
                            st.write(f"• {palabra} ({freq} veces)")
            
            # Temáticas del autor
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Temáticas de Investigación:**")
                for tema in autor_info['Tematicas']:
                    st.write(f"• {tema}")

def analyze_institutional_collaboration(df_filtered, G_institucional, metrics_inst):
    """Análisis específico de colaboración institucional"""
    st.subheader("🔍 Análisis de Datos: Colaboración Institucional")
    
    # Análisis de instituciones
    inst_stats = {}
    for _, row in df_filtered.iterrows():
        instituciones = list(set(row['afiliaciones_list']))  # Eliminar duplicados
        instituciones = [inst for inst in instituciones if inst]  # Filtrar vacíos
        
        for inst in instituciones:
            if inst not in inst_stats:
                inst_stats[inst] = {
                    'papers': 0,
                    'autores': set(),
                    'tematicas': set(),
                    'palabras_clave': [],
                    'colaboraciones_inst': set()
                }
            
            inst_stats[inst]['papers'] += 1
            inst_stats[inst]['autores'].update(row['autores_list'])
            inst_stats[inst]['tematicas'].add(row['tematica'])
            inst_stats[inst]['palabras_clave'].extend(row['palabras_clave_list'])
            
            # Colaboraciones con otras instituciones en el mismo paper
            otras_inst = [i for i in instituciones if i != inst]
            inst_stats[inst]['colaboraciones_inst'].update(otras_inst)
    
    if inst_stats:
        # Convertir a DataFrame
        inst_df = []
        for inst, stats in inst_stats.items():
            inst_df.append({
                'Institución': inst,
                'Papers': stats['papers'],
                'Autores': len(stats['autores']),
                'Diversidad_Tematica': len(stats['tematicas']),
                'Colaboraciones_Inst': len(stats['colaboraciones_inst']),
                'Palabras_Clave': stats['palabras_clave'],
                'Tematicas': list(stats['tematicas']),
                'Especialización': len(stats['tematicas']) / stats['papers'] if stats['papers'] > 0 else 0
            })
        
        if inst_df:  # Verificar que hay datos
            inst_df = pd.DataFrame(inst_df)
            inst_df = inst_df[inst_df['Papers'] >= 2]  # Filtrar instituciones con al menos 2 papers
            
            if len(inst_df) == 0:
                st.warning("No hay instituciones con suficientes papers para realizar el análisis.")
                return
                
            inst_df = inst_df.sort_values('Papers', ascending=False)
            
            col1, col2 = st.columns(2)
        else:
            st.warning("No hay suficientes datos institucionales para realizar el análisis.")
            return
        
        with col1:
            st.write("**🏆 Top Instituciones por Productividad**")
            top_inst = inst_df.head(10)
            
            fig = px.bar(
                top_inst,
                x='Papers',
                y='Institución',
                orientation='h',
                title="Instituciones Más Productivas"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**🎯 Especialización vs Diversidad**")
            
            # Crear categorías de especialización
            inst_df['Tipo_Especialización'] = inst_df['Especialización'].apply(
                lambda x: 'Muy Especializada' if x <= 0.3 else 
                         'Especializada' if x <= 0.6 else 
                         'Diversificada'
            )
            
            fig_scatter = px.scatter(
                inst_df.head(15),
                x='Papers',
                y='Diversidad_Tematica',
                size='Autores',
                color='Tipo_Especialización',
                hover_name='Institución',
                title="Productividad vs Diversidad Temática",
                labels={
                    'Papers': 'Número de Papers',
                    'Diversidad_Tematica': 'Número de Temáticas Diferentes'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Análisis de colaboración entre instituciones
        st.subheader("🤝 Análisis de Colaboración Inter-Institucional")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Instituciones Más Colaborativas**")
            top_colab = inst_df.nlargest(10, 'Colaboraciones_Inst')
            
            fig = px.bar(
                top_colab,
                x='Colaboraciones_Inst',
                y='Institución',
                orientation='h',
                title="Instituciones con Más Colaboraciones"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Análisis de especialización
            especialización_counts = inst_df['Tipo_Especialización'].value_counts()
            
            fig_pie = px.pie(
                values=especialización_counts.values,
                names=especialización_counts.index,
                title="Distribución de Tipos de Especialización"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Análisis detallado de institución
        st.subheader("🔍 Análisis Detallado de Institución")
        inst_seleccionada = st.selectbox(
            "Selecciona una institución para análisis detallado:",
            [''] + list(inst_df['Institución'].values),
            key='inst_analysis'
        )
        
        if inst_seleccionada:
            inst_info = inst_df[inst_df['Institución'] == inst_seleccionada].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Papers", inst_info['Papers'])
            with col2:
                st.metric("👥 Autores", inst_info['Autores'])
            with col3:
                st.metric("📚 Temáticas", inst_info['Diversidad_Tematica'])
            with col4:
                st.metric("🤝 Colaboraciones", inst_info['Colaboraciones_Inst'])
            
            # Nube de palabras de especialización institucional
            if inst_info['Palabras_Clave']:
                st.subheader(f"☁️ Áreas de Especialización - {inst_seleccionada}")
                
                palabra_freq = Counter(inst_info['Palabras_Clave'])
                if len(palabra_freq) > 0:
                    try:
                        wordcloud = WordCloud(
                            width=800, 
                            height=300, 
                            background_color='white',
                            max_words=40,
                            colormap='plasma'
                        ).generate_from_frequencies(palabra_freq)
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    except:
                        st.write("**Principales áreas de investigación:**")
                        for palabra, freq in palabra_freq.most_common(20):
                            st.write(f"• {palabra} ({freq} veces)")
            
            # Información adicional
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Temáticas de Investigación:**")
                for tema in inst_info['Tematicas'][:10]:
                    st.write(f"• {tema}")
            
            with col2:
                tipo_esp = inst_info['Tipo_Especialización']
                if tipo_esp == 'Muy Especializada':
                    st.success("🎯 **Institución Muy Especializada**: Se enfoca en pocas áreas específicas")
                elif tipo_esp == 'Especializada':
                    st.info("📊 **Instituci��n Especializada**: Tiene un enfoque moderadamente diverso")
                else:
                    st.warning("🌐 **Institución Diversificada**: Abarca muchas áreas diferentes")

def analyze_thematic_patterns(df_filtered, G_tematica, metrics_tema):
    """Análisis específico de patrones temáticos"""
    st.subheader("🔍 Análisis de Datos: Patrones Temáticos")
    
    # Análisis de temáticas
    tema_stats = {}
    for _, row in df_filtered.iterrows():
        tematica = row['tematica']
        if tematica not in tema_stats:
            tema_stats[tematica] = {
                'papers': 0,
                'autores': set(),
                'instituciones': set(),
                'palabras_clave': [],
                'colaboraciones': 0
            }
        
        tema_stats[tematica]['papers'] += 1
        tema_stats[tematica]['autores'].update(row['autores_list'])
        tema_stats[tematica]['instituciones'].update([inst for inst in row['afiliaciones_list'] if inst])
        tema_stats[tematica]['palabras_clave'].extend(row['palabras_clave_list'])
        
        # Contar colaboraciones (papers con más de un autor)
        if len(row['autores_list']) > 1:
            tema_stats[tematica]['colaboraciones'] += 1
    
    if tema_stats:
        # Convertir a DataFrame
        tema_df = []
        for tema, stats in tema_stats.items():
            tema_df.append({
                'Temática': tema,
                'Papers': stats['papers'],
                'Autores': len(stats['autores']),
                'Instituciones': len(stats['instituciones']),
                'Colaboraciones': stats['colaboraciones'],
                'Tasa_Colaboracion': stats['colaboraciones'] / stats['papers'] if stats['papers'] > 0 else 0,
                'Palabras_Clave': stats['palabras_clave']
            })
        
        if tema_df:  # Verificar que hay datos
            tema_df = pd.DataFrame(tema_df)
            tema_df = tema_df[tema_df['Papers'] >= 2]
            
            if len(tema_df) == 0:
                st.warning("No hay temáticas con suficientes papers para realizar el análisis.")
                return
                
            tema_df = tema_df.sort_values('Papers', ascending=False)
            
            col1, col2 = st.columns(2)
        else:
            st.warning("No hay suficientes datos temáticos para realizar el análisis.")
            return
        
        with col1:
            st.write("**🏆 Temáticas Más Investigadas**")
            top_temas = tema_df.head(10)
            
            fig = px.bar(
                top_temas,
                x='Papers',
                y='Temática',
                orientation='h',
                title="Temáticas con Más Publicaciones"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**🤝 Colaboración por Temática**")
            
            fig_scatter = px.scatter(
                tema_df.head(15),
                x='Papers',
                y='Tasa_Colaboracion',
                size='Autores',
                hover_name='Temática',
                title="Productividad vs Tasa de Colaboración",
                labels={
                    'Papers': 'Número de Papers',
                    'Tasa_Colaboracion': 'Tasa de Colaboración',
                    'Autores': 'Número de Autores'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Análisis detallado de temática
        st.subheader("🔍 Análisis Detallado de Temática")
        tema_seleccionada = st.selectbox(
            "Selecciona una temática para análisis detallado:",
            [''] + list(tema_df['Temática'].values),
            key='tema_analysis'
        )
        
        if tema_seleccionada:
            tema_info = tema_df[tema_df['Temática'] == tema_seleccionada].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Papers", tema_info['Papers'])
            with col2:
                st.metric("👥 Autores", tema_info['Autores'])
            with col3:
                st.metric("🏛️ Instituciones", tema_info['Instituciones'])
            with col4:
                st.metric("🤝 Tasa Colaboración", f"{tema_info['Tasa_Colaboracion']:.2%}")
            
            # Nube de palabras de la temática
            if tema_info['Palabras_Clave']:
                st.subheader(f"☁️ Palabras Clave - {tema_seleccionada}")
                
                palabra_freq = Counter(tema_info['Palabras_Clave'])
                if len(palabra_freq) > 0:
                    try:
                        wordcloud = WordCloud(
                            width=800, 
                            height=300, 
                            background_color='white',
                            max_words=50,
                            colormap='coolwarm'
                        ).generate_from_frequencies(palabra_freq)
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    except:
                        st.write("**Principales palabras clave:**")
                        for palabra, freq in palabra_freq.most_common(25):
                            st.write(f"• {palabra} ({freq} veces)")

def analyze_keyword_patterns(df_filtered, G_keywords, metrics_kw):
    """Análisis específico de patrones de palabras clave"""
    st.subheader("🔍 Análisis de Datos: Patrones de Palabras Clave")
    
    # Análisis de palabras clave más frecuentes
    all_keywords = []
    keyword_authors = {}
    keyword_tematicas = {}
    
    for _, row in df_filtered.iterrows():
        palabras_clave = row['palabras_clave_list']
        autores = row['autores_list']
        tematica = row['tematica']
        
        for kw in palabras_clave:
            if kw:
                all_keywords.append(kw)
                
                if kw not in keyword_authors:
                    keyword_authors[kw] = set()
                    keyword_tematicas[kw] = set()
                
                keyword_authors[kw].update(autores)
                keyword_tematicas[kw].add(tematica)
    
    if all_keywords:
        # Análisis de frecuencia de palabras clave
        keyword_freq = Counter(all_keywords)
        
        # Crear DataFrame para análisis
        kw_df = []
        for kw, freq in keyword_freq.items():
            if freq >= 2:  # Solo palabras clave que aparecen al menos 2 veces
                kw_df.append({
                    'Palabra_Clave': kw,
                    'Frecuencia': freq,
                    'Num_Autores': len(keyword_authors[kw]),
                    'Num_Tematicas': len(keyword_tematicas[kw]),
                    'Autores': list(keyword_authors[kw]),
                    'Tematicas': list(keyword_tematicas[kw])
                })
        
        if kw_df:  # Verificar que hay datos
            kw_df = pd.DataFrame(kw_df)
            kw_df = kw_df.sort_values('Frecuencia', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🏆 Palabras Clave Más Frecuentes**")
                top_keywords = kw_df.head(15)
                
                fig = px.bar(
                    top_keywords,
                    x='Frecuencia',
                    y='Palabra_Clave',
                    orientation='h',
                    title="Palabras Clave Más Utilizadas"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**🌐 Diversidad de Palabras Clave**")
                
                fig_scatter = px.scatter(
                    kw_df.head(20),
                    x='Frecuencia',
                    y='Num_Autores',
                    size='Num_Tematicas',
                    hover_name='Palabra_Clave',
                    title="Frecuencia vs Número de Autores",
                    labels={
                        'Frecuencia': 'Frecuencia de Uso',
                        'Num_Autores': 'Número de Autores',
                        'Num_Tematicas': 'Número de Temáticas'
                    }
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Análisis de especialización vs generalización
            st.subheader("🎯 Especialización de Palabras Clave")
            
            # Clasificar palabras clave por especialización
            kw_df['Especialización'] = kw_df['Num_Tematicas'] / kw_df['Frecuencia']
            kw_df['Tipo_Palabra'] = kw_df['Especialización'].apply(
                lambda x: 'Muy Específica' if x <= 0.3 else 
                         'Específica' if x <= 0.6 else 
                         'General'
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribución de tipos de palabras
                tipo_counts = kw_df['Tipo_Palabra'].value_counts()
                
                fig_pie = px.pie(
                    values=tipo_counts.values,
                    names=tipo_counts.index,
                    title="Distribución de Tipos de Palabras Clave"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.write("**Palabras Clave Más Específicas:**")
                especificas = kw_df[kw_df['Tipo_Palabra'] == 'Muy Específica'].head(10)
                for _, row in especificas.iterrows():
                    st.write(f"• **{row['Palabra_Clave']}** ({row['Frecuencia']} usos, {row['Num_Tematicas']} temáticas)")
            
            # Análisis detallado de palabra clave
            st.subheader("🔍 Análisis Detallado de Palabra Clave")
            kw_seleccionada = st.selectbox(
                "Selecciona una palabra clave para análisis detallado:",
                [''] + list(kw_df['Palabra_Clave'].values),
                key='kw_analysis'
            )
            
            if kw_seleccionada:
                kw_info = kw_df[kw_df['Palabra_Clave'] == kw_seleccionada].iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Frecuencia", kw_info['Frecuencia'])
                with col2:
                    st.metric("👥 Autores", kw_info['Num_Autores'])
                with col3:
                    st.metric("📚 Temáticas", kw_info['Num_Tematicas'])
                with col4:
                    tipo = kw_info['Tipo_Palabra']
                    if tipo == 'Muy Específica':
                        st.success("🎯 Muy Específica")
                    elif tipo == 'Específica':
                        st.info("📊 Específica")
                    else:
                        st.warning("🌐 General")
                
                # Mostrar información detallada
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Principales Autores:**")
                    for autor in kw_info['Autores'][:10]:
                        st.write(f"• {autor}")
                
                with col2:
                    st.write("**Temáticas Relacionadas:**")
                    for tema in kw_info['Tematicas']:
                        st.write(f"• {tema}")
        else:
            st.warning("No hay suficientes palabras clave que aparezcan múltiples veces para realizar el análisis.")

def analyze_institution_thematic_patterns(df_filtered, G_inst_tema, metrics_inst_tema):
    """Análisis específico de patrones institución-temática"""
    st.subheader("🔍 Análisis de Datos: Especialización Institucional")
    
    # Análisis de especialización institucional por temática
    inst_tema_stats = {}
    
    for _, row in df_filtered.iterrows():
        instituciones = [inst for inst in row['afiliaciones_list'] if inst]
        tematica = row['tematica']
        palabras_clave = row['palabras_clave_list']
        autores = row['autores_list']
        
        for inst in set(instituciones):
            if inst not in inst_tema_stats:
                inst_tema_stats[inst] = {}
            
            if tematica not in inst_tema_stats[inst]:
                inst_tema_stats[inst][tematica] = {
                    'papers': 0,
                    'autores': set(),
                    'palabras_clave': []
                }
            
            inst_tema_stats[inst][tematica]['papers'] += 1
            inst_tema_stats[inst][tematica]['autores'].update(autores)
            inst_tema_stats[inst][tematica]['palabras_clave'].extend(palabras_clave)
    
    if inst_tema_stats:
        # Crear análisis de especialización
        especialización_data = []
        
        for inst, temas in inst_tema_stats.items():
            total_papers = sum(tema_data['papers'] for tema_data in temas.values())
            if total_papers >= 2:  # Solo instituciones con al menos 2 papers
                
                # Encontrar temática principal
                tema_principal = max(temas.items(), key=lambda x: x[1]['papers'])
                
                especialización_data.append({
                    'Institución': inst,
                    'Total_Papers': total_papers,
                    'Num_Tematicas': len(temas),
                    'Tema_Principal': tema_principal[0],
                    'Papers_Tema_Principal': tema_principal[1]['papers'],
                    'Porcentaje_Especialización': (tema_principal[1]['papers'] / total_papers) * 100,
                    'Autores_Tema_Principal': len(tema_principal[1]['autores']),
                    'Palabras_Clave_Principal': tema_principal[1]['palabras_clave']
                })
        
        if especialización_data:  # Verificar que hay datos
            esp_df = pd.DataFrame(especialización_data)
            esp_df = esp_df.sort_values('Total_Papers', ascending=False)
            
            # Clasificar instituciones por nivel de especialización
            esp_df['Nivel_Especialización'] = esp_df['Porcentaje_Especialización'].apply(
                lambda x: 'Muy Especializada' if x >= 70 else 
                         'Especializada' if x >= 50 else 
                         'Diversificada'
            )
        else:
            st.warning("No hay suficientes datos institucionales para realizar el análisis de especialización.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🏆 Instituciones Más Especializadas**")
            top_esp = esp_df.nlargest(10, 'Porcentaje_Especialización')
            
            fig = px.bar(
                top_esp,
                x='Porcentaje_Especialización',
                y='Institución',
                orientation='h',
                title="Instituciones por Nivel de Especialización (%)",
                color='Nivel_Especialización'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**📊 Productividad vs Especialización**")
            
            fig_scatter = px.scatter(
                esp_df,
                x='Total_Papers',
                y='Porcentaje_Especialización',
                size='Num_Tematicas',
                color='Nivel_Especialización',
                hover_name='Institución',
                title="Productividad vs Especialización",
                labels={
                    'Total_Papers': 'Número Total de Papers',
                    'Porcentaje_Especialización': 'Porcentaje de Especialización (%)',
                    'Num_Tematicas': 'Número de Temáticas'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # An��lisis de fortalezas institucionales
        st.subheader("🎯 Fortalezas de Investigación por Institución")
        
        # Agrupar por temática principal
        fortalezas = esp_df.groupby('Tema_Principal').agg({
            'Institución': 'count',
            'Papers_Tema_Principal': 'sum',
            'Autores_Tema_Principal': 'sum'
        }).reset_index()
        
        fortalezas.columns = ['Temática', 'Num_Instituciones', 'Total_Papers', 'Total_Autores']
        fortalezas = fortalezas.sort_values('Total_Papers', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Temáticas con Más Instituciones Especializadas:**")
            top_fortalezas = fortalezas.head(10)
            
            fig = px.bar(
                top_fortalezas,
                x='Num_Instituciones',
                y='Temática',
                orientation='h',
                title="Temáticas por Número de Instituciones"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribución de niveles de especialización
            nivel_counts = esp_df['Nivel_Especialización'].value_counts()
            
            fig_pie = px.pie(
                values=nivel_counts.values,
                names=nivel_counts.index,
                title="Distribución de Niveles de Especialización"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Análisis detallado de institución
        st.subheader("🔍 Análisis Detallado de Especialización Institucional")
        inst_seleccionada = st.selectbox(
            "Selecciona una institución para análisis detallado:",
            [''] + list(esp_df['Institución'].values),
            key='inst_tema_analysis'
        )
        
        if inst_seleccionada:
            inst_info = esp_df[esp_df['Institución'] == inst_seleccionada].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Total Papers", inst_info['Total_Papers'])
            with col2:
                st.metric("📚 Temáticas", inst_info['Num_Tematicas'])
            with col3:
                st.metric("🎯 Especialización", f"{inst_info['Porcentaje_Especialización']:.1f}%")
            with col4:
                nivel = inst_info['Nivel_Especialización']
                if nivel == 'Muy Especializada':
                    st.success("🎯 Muy Especializada")
                elif nivel == 'Especializada':
                    st.info("��� Especializada")
                else:
                    st.warning("🌐 Diversificada")
            
            # Información de la temática principal
            st.subheader(f"🏆 Fortaleza Principal: {inst_info['Tema_Principal']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Papers en Tema Principal", inst_info['Papers_Tema_Principal'])
                st.metric("👥 Autores en Tema Principal", inst_info['Autores_Tema_Principal'])
            
            with col2:
                # Nube de palabras de la especialización
                if inst_info['Palabras_Clave_Principal']:
                    palabra_freq = Counter(inst_info['Palabras_Clave_Principal'])
                    if len(palabra_freq) > 0:
                        try:
                            wordcloud = WordCloud(
                                width=600, 
                                height=300, 
                                background_color='white',
                                max_words=30,
                                colormap='viridis'
                            ).generate_from_frequencies(palabra_freq)
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f"Especialización en {inst_info['Tema_Principal']}")
                            st.pyplot(fig)
                            plt.close()
                        except:
                            st.write("**Palabras clave principales:**")
                            for palabra, freq in palabra_freq.most_common(15):
                                st.write(f"• {palabra} ({freq})")
            
            # Mostrar todas las temáticas de la institución
            if inst_seleccionada in inst_tema_stats:
                st.subheader("📊 Distribución Completa de Temáticas")
                
                temas_inst = []
                for tema, data in inst_tema_stats[inst_seleccionada].items():
                    temas_inst.append({
                        'Temática': tema,
                        'Papers': data['papers'],
                        'Autores': len(data['autores']),
                        'Porcentaje': (data['papers'] / inst_info['Total_Papers']) * 100
                    })
                
                temas_df = pd.DataFrame(temas_inst)
                temas_df = temas_df.sort_values('Papers', ascending=False)
                
                fig = px.bar(
                    temas_df,
                    x='Papers',
                    y='Temática',
                    orientation='h',
                    title=f"Distribución de Papers por Temática - {inst_seleccionada}"
                )
                fig.update_layout(height=300 + len(temas_df) * 20)
                st.plotly_chart(fig, use_container_width=True)

##INTERFAZ PRINCIPAL
with st.spinner("🔄 Cargando datos de papers cubanos..."):
    df, stats = load_and_analyze_data()

if df is None:
    st.stop()

filters = create_sidebar_filters(df)

df_filtered = apply_filters(df, filters)

tabs = st.tabs([
    "📊 General", 
    "🤝 Red de Coautoría", 
    "🏢 Red Institucional", 
    "📚 Red Temática",
    "🔑 Red de Palabras Clave",
    "🏛️ Red Institución-Temática"
])

with tabs[0]:
    st.subheader("📊 Estadísticas Generales")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", f"{stats['total_papers']:,}")
    with col2:
        st.metric("Total Autores", f"{stats['total_autores']:,}")
    with col3:
        st.metric("Total Instituciones", f"{stats['total_instituciones']:,}")
    with col4:
        st.metric("Total Palabras Clave", f"{stats['total_palabras_clave']:,}")
    
    st.subheader("🛠️ Características de la Herramienta")

    st.markdown("""
    #### **Funcionalidades Principales:**
    - **Filtros**: Filtra por temática e institución para crear subgrafos específicos
    - **Múltiples Tipos de Redes**: Coautoría, institucional, temática, palabras clave, institución-temática.
    - **Visualizaciones Interactivas**: Redes interactivas con información detallada al pasar el mouse
    - **Métricas de Centralidad**: Identifica autores e instituciones más importantes
    - **Detección de Comunidades**: Encuentra grupos de investigación colaborativa
    """)
        
    st.info("💡Métricas")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ##### **Métricas de Centralidad:**
        - **Centralidad de Grado**:  Qué tan conectado está un nodo. Útil para identificar autores con muchas colaboraciones.
        - **Centralidad de Intermediación (Betweenness)**: Qué tan frecuentemente un nodo está en el camino más corto entre otros dos. Revela autores puente.
        - **Centralidad de Cercanía (Closeness)**: Qué tan rápido puede acceder un nodo al resto de la red. Ideal para detectar nodos con buena difusión.
        - **Centralidad de Eigenvector:**: Evalúa la influencia de un nodo tomando en cuenta la importancia de sus vecinos
        """)
    
    with col2:
        st.markdown("""
        ##### **Métricas de agrupamiento y cohesión:**
        - **Coeficiente de agrupamiento (Clustering coefficient)**: Mide qué tan conectados están los vecinos de un nodo entre sí. Perfecto para detectar comunidades científicas.
        - **Modularidad**: Ayuda a identificar comunidades dentro de la red (ej. grupos de autores que suelen colaborar entre sí).
        """)
        
    with col3:
        st.markdown("""
        ##### **Métricas de alcance e influencia:**
        - **Densidad**: Cuántos enlaces existen en relación a todos los posibles. Una red muy densa podría indicar un campo de investigación muy interconectado.
        """)
        
with tabs[1]:
    st.header("🤝 Red de Coautoría")
    st.markdown("""
    - Identifica patrones de colaboración entre investigadores cubanos.
    - Revela quiénes son los autores más conectados e influyentes.
    - Muestra comunidades de investigación que trabajan juntas frecuentemente.
    """)
    
    if len(df_filtered) == 0:
        st.warning("⚠️ No hay papers que coincidan con los filtros seleccionados.")
    else:
        with st.spinner("Construyendo red de coautoría..."):
            G_coautor = build_filtered_coauthorship_network(
                df_filtered, 
                filters['min_colaboraciones'], 
                filters['max_nodos']
            )
        
        if G_coautor.number_of_nodes() > 0:
            # Calcular métricas específicas para esta red
            metrics_coautor = calculate_network_metrics(G_coautor, "coautoria")
            
            create_network_visualization(G_coautor, metrics_coautor, "Red de Coautoría")
            
            if 'degree_centrality' in metrics_coautor:
                st.subheader("Análisis de Centralidad")
                
                col1, col2 = st.columns(2)
                with col1:
                    centrality_type = st.selectbox(
                        "Selecciona la métrica de centralidad:",
                        ['Grado', 'Intermediación', 'Cercanía', 'Eigenvector'],
                        key='centrality_coautor'
                    )
                with col2:
                    top_n = st.selectbox(
                        "Top N a mostrar:",
                        [5, 10, 15, 20, 25],
                        index=1,
                        key='top_n_coautor'
                    )
                
                create_single_centrality_plot(metrics_coautor, centrality_type, top_n)
            
            # Análisis específico de colaboración de autores
            analyze_author_collaboration_patterns(df_filtered, G_coautor, metrics_coautor)
        else:
            st.warning("No hay suficientes datos para construir la red de coautoría con los filtros actuales.")

with tabs[2]:
    st.header("🏢 Red de Colaboración Institucional")
    st.markdown("""
    - Analiza colaboraciones entre instituciones cubanas.
    - Identifica instituciones que actúan como puentes entre diferentes grupos.
    """)
    
    if len(df_filtered) == 0:
        st.warning("⚠️ No hay papers que coincidan con los filtros seleccionados.")
    else:
        min_shared = st.slider(
            "Autores compartidos mínimos entre instituciones:", 
            1, 5, 1, 
            key='inst_shared',
            help="Dos instituciones se conectan si comparten al menos este número de autores"
        )
        
        with st.spinner("Construyendo red institucional..."):
            G_institucional = build_filtered_institution_network(
                df_filtered, 
                min_shared, 
                filters['max_nodos']//2
            )
        
        if G_institucional.number_of_nodes() > 0:
            # Calcular métricas específicas para esta red
            metrics_inst = calculate_network_metrics(G_institucional, "institucional")
            
            create_network_visualization(G_institucional, metrics_inst, "Red Institucional")
            
            if 'degree_centrality' in metrics_inst:
                st.subheader("Análisis de Centralidad")
                
                col1, col2 = st.columns(2)
                with col1:
                    centrality_type = st.selectbox(
                        "Selecciona métrica de centralidad:",
                        ['Grado', 'Intermediación', 'Cercanía', 'Eigenvector'],
                        key='centrality_inst'
                    )
                with col2:
                    top_n = st.selectbox(
                        "Top N a mostrar:",
                        [5, 10, 15, 20],
                        index=1,
                        key='top_n_inst'
                    )
                
                create_single_centrality_plot(metrics_inst, centrality_type, top_n)
            
            # Análisis específico de colaboración institucional
            analyze_institutional_collaboration(df_filtered, G_institucional, metrics_inst)
        else:
            st.warning("No hay suficientes datos para construir la red institucional con los filtros actuales.")

with tabs[3]:
    st.header("📚 Red Temática (Autor-Temática)")
    st.markdown("""
    - Conecta autores con las temáticas que investigan.
    - Identifica especialistas en cada área.
    """)
    
    if len(df_filtered) == 0:
        st.warning("⚠️ No hay papers que coincidan con los filtros seleccionados.")
    else:
        with st.spinner("Construyendo red temática..."):
            G_tematica = build_thematic_network(df_filtered, filters['max_nodos'])
        
        if G_tematica.number_of_nodes() > 0:
            # Calcular métricas específicas para esta red
            metrics_tema = calculate_network_metrics(G_tematica, "tematica")
            
            st.info("🔴 Nodos rojos = Autores | 🟢 Nodos verdes = Temáticas")
            create_network_visualization(G_tematica, metrics_tema, "Red Temática")
            
            # Métricas de centralidad para red bipartita
            if 'degree_centrality' in metrics_tema:
                st.subheader("Análisis de Centralidad")
                
                col1, col2 = st.columns(2)
                with col1:
                    centrality_type = st.selectbox(
                        "Selecciona métrica de centralidad:",
                        ['Grado', 'Intermediación', 'Cercanía'],
                        key='centrality_tema'
                    )
                with col2:
                    top_n = st.selectbox(
                        "Top N a mostrar:",
                        [5, 10, 15, 20],
                        index=1,
                        key='top_n_tema'
                    )
                
                create_single_centrality_plot(metrics_tema, centrality_type, top_n)
            
            # Análisis específico de patrones temáticos
            analyze_thematic_patterns(df_filtered, G_tematica, metrics_tema)
        else:
            st.warning("No hay suficientes datos para construir la red temática con los filtros actuales.")

with tabs[4]:
    st.header("🔑 Red de Palabras Clave (Autor-Palabra Clave)")
    st.markdown("""
    - Análisis de los intereses de investigación.
    - Conecta investigadores con intereses específicos similares.
    """)
    
    if len(df_filtered) == 0:
        st.warning("⚠️ No hay papers que coincidan con los filtros seleccionados.")
    else:
        with st.spinner("Construyendo red de palabras clave..."):
            G_keywords = build_keyword_network(df_filtered, filters['max_nodos'])
        
        if G_keywords.number_of_nodes() > 0:
            # Calcular métricas específicas para esta red
            metrics_kw = calculate_network_metrics(G_keywords, "keywords")
            
            st.info("🔴 Nodos rojos = Autores | 🔵 Nodos azules = Palabras Clave")
            create_network_visualization(G_keywords, metrics_kw, "Red de Palabras Clave")
            
            if 'degree_centrality' in metrics_kw:
                st.subheader("Análisis de Centralidad")
                
                col1, col2 = st.columns(2)
                with col1:
                    centrality_type = st.selectbox(
                        "Selecciona métrica de centralidad:",
                        ['Grado', 'Intermediación', 'Cercanía'],
                        key='centrality_kw'
                    )
                with col2:
                    top_n = st.selectbox(
                        "Top N a mostrar:",
                        [5, 10, 15, 20, 25],
                        index=2,
                        key='top_n_kw'
                    )
                
                create_single_centrality_plot(metrics_kw, centrality_type, top_n)
            
            # Análisis específico de patrones de palabras clave
            analyze_keyword_patterns(df_filtered, G_keywords, metrics_kw)
        else:
            st.warning("No hay suficientes datos para construir la red de palabras clave con los filtros actuales.")

with tabs[5]:
    st.header("🏛️ Red Institución-Temática")
    st.markdown("""
    - Identifica fortalezas de investigación, conecta instituciones con las temáticas que publican.
    - Revela qué instituciones se especializan en qué áreas.
    - Muestra instituciones con mayor diversidad de investigación.
    - Identifica instituciones que comparten intereses de investigación.
    """)
    
    if len(df_filtered) == 0:
        st.warning("⚠️ No hay papers que coincidan con los filtros seleccionados.")
    else:
        with st.spinner("Construyendo red Institución-Temática..."):
            G_inst_tema = build_institution_thematic_network(df_filtered, filters['max_nodos'])
        
        if G_inst_tema.number_of_nodes() > 0:
            # Calcular métricas específicas para esta red
            metrics_inst_tema = calculate_network_metrics(G_inst_tema, "institucion_tematica")
            
            st.info("🔵 Nodos azules = Instituciones | 🟢 Nodos verdes = Temáticas")
            create_network_visualization(G_inst_tema, metrics_inst_tema, "Red Institución-Temática")
            
            # Crear análisis de especialización por institución
            instituciones = [n for n, d in G_inst_tema.nodes(data=True) if d.get('bipartite') == 0]
            tematicas = [n for n, d in G_inst_tema.nodes(data=True) if d.get('bipartite') == 1]
            
            if instituciones and tematicas:
                # Análisis de diversidad temática por institución
                inst_diversidad = {}
                inst_fortalezas = {}
                
                for inst in instituciones:
                    # Obtener temáticas conectadas y sus pesos
                    temas_conectadas = []
                    pesos_temas = []
                    
                    for tema in G_inst_tema.neighbors(inst):
                        peso = G_inst_tema[inst][tema].get('weight', 1)
                        temas_conectadas.append(tema)
                        pesos_temas.append(peso)
                    
                    inst_diversidad[inst] = len(temas_conectadas)
                    
                    # Identificar fortaleza principal (temática con mayor peso)
                    if pesos_temas:
                        idx_max = pesos_temas.index(max(pesos_temas))
                        inst_fortalezas[inst] = {
                            'tema_principal': temas_conectadas[idx_max],
                            'num_papers': max(pesos_temas),
                            'diversidad': len(temas_conectadas),
                            'total_papers': sum(pesos_temas)
                        }
                
            # Métricas de centralidad
            if 'degree_centrality' in metrics_inst_tema:
                st.subheader("Análisis de Centralidad")
                
                col1, col2 = st.columns(2)
                with col1:
                    centrality_type = st.selectbox(
                        "Selecciona métrica de centralidad:",
                        ['Grado', 'Intermediación', 'Cercanía'],
                        key='centrality_inst_tema'
                    )
                with col2:
                    top_n = st.selectbox(
                        "Top N a mostrar:",
                        [5, 10, 15, 20],
                        index=1,
                        key='top_n_inst_tema'
                    )

                create_single_centrality_plot(metrics_inst_tema, centrality_type, top_n)
            
            # Análisis específico de especialización institucional
            analyze_institution_thematic_patterns(df_filtered, G_inst_tema, metrics_inst_tema)
        else:
            st.warning("No hay suficientes datos para construir la red Institución-Temática con los filtros actuales.")
    st.markdown("""
    Esta sección te permite explorar los datos de manera interactiva para responder preguntas específicas sobre:
    - **Autores**: ¿Quiénes son los más productivos? ¿En qué instituciones trabajan?
    - **Instituciones**: ¿Cuáles son las más activas? ¿En qué áreas se especializan?
    - **Colaboraciones**: ¿Qué patrones de colaboración existen?
    - **Temáticas**: ¿Cuáles son las áreas de investigación más populares?
    """)
    
    # Selector de tipo de análisis
    analysis_type = st.selectbox(
        "¿Qué quieres explorar?",
        [
            "🧑‍🔬 Análisis de Autores",
            "🏛️ Análisis de Instituciones", 
            "Análisis de Colaboraciones",
            "📚 Análisis de Temáticas",
            "🔍 Búsqueda Específica"
        ]
    )
    
    if analysis_type == "🧑‍🔬 Análisis de Autores":
        st.subheader("Análisis Detallado de Autores")
        
        # Crear análisis de autores
        autor_stats = []
        for _, row in df_filtered.iterrows():
            autores = row['autores_list']
            instituciones = row['afiliaciones_list']
            tematica = row['tematica']
            palabras_clave = row['palabras_clave_list']
            
            for autor in autores:
                autor_stats.append({
                    'autor': autor,
                    'instituciones': instituciones,
                    'tematica': tematica,
                    'palabras_clave': palabras_clave,
                    'titulo': row['titulo']
                })
        
        if autor_stats:
            autor_df = pd.DataFrame(autor_stats)
            
            # Estadísticas por autor
            autor_summary = autor_df.groupby('autor').agg({
                'titulo': 'count',
                'tematica': lambda x: list(set(x)),
                'instituciones': lambda x: list(set([inst for sublist in x for inst in sublist if inst])),
            }).reset_index()
            
            autor_summary.columns = ['Autor', 'Num_Papers', 'Temáticas', 'Instituciones']
            autor_summary = autor_summary.sort_values('Num_Papers', ascending=False)
            
            # Top autores más productivos
            st.subheader("🏆 Top Autores Más Productivos")
            top_n_autores = st.slider("Número de autores a mostrar:", 5, 50, 20)
            
            top_autores = autor_summary.head(top_n_autores)
            
            # Gráfico de barras
            fig = px.bar(
                top_autores, 
                x='Num_Papers', 
                y='Autor',
                orientation='h',
                title=f"Top {top_n_autores} Autores por Número de Publicaciones",
                labels={'Num_Papers': 'Número de Papers', 'Autor': 'Autor'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Búsqueda de autor específico
            st.subheader("🔍 Información Detallada de Autor")
            autor_seleccionado = st.selectbox(
                "Selecciona un autor para ver detalles:",
                [''] + list(autor_summary['Autor'].values)
            )
            
            if autor_seleccionado:
                autor_info = autor_summary[autor_summary['Autor'] == autor_seleccionado].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("📄 Número de Papers", autor_info['Num_Papers'])
                    st.metric("🏛️ Número de Instituciones", len(autor_info['Instituciones']))
                    
                    st.write("**Instituciones:**")
                    for inst in autor_info['Instituciones']:
                        st.write(f"• {inst}")
                
                with col2:
                    st.metric("📚 Número de Temáticas", len(autor_info['Temáticas']))
                    
                    st.write("**Temáticas de Investigación:**")
                    for tema in autor_info['Temáticas']:
                        st.write(f"• {tema}")
                
                # Papers del autor
                papers_autor = autor_df[autor_df['autor'] == autor_seleccionado]
                st.subheader(f"📄 Papers de {autor_seleccionado}")
                
                for i, (_, paper) in enumerate(papers_autor.iterrows(), 1):
                    with st.expander(f"Paper {i}: {paper['titulo'][:100]}..."):
                        st.write(f"**Título:** {paper['titulo']}")
                        st.write(f"**Temática:** {paper['tematica']}")
                        if paper['palabras_clave']:
                            st.write(f"**Palabras Clave:** {', '.join(paper['palabras_clave'])}")
    
    elif analysis_type == "🏛️ Análisis de Instituciones":
        st.subheader("Análisis Detallado de Instituciones")
        
        # Crear análisis de instituciones
        inst_stats = []
        for _, row in df_filtered.iterrows():
            autores = row['autores_list']
            instituciones = row['afiliaciones_list']
            tematica = row['tematica']
            
            for inst in set(instituciones):
                if inst:
                    inst_stats.append({
                        'institucion': inst,
                        'autores': autores,
                        'tematica': tematica,
                        'titulo': row['titulo']
                    })
        
        if inst_stats:
            inst_df = pd.DataFrame(inst_stats)
            
            # Estadísticas por institución
            inst_summary = inst_df.groupby('institucion').agg({
                'titulo': 'count',
                'tematica': lambda x: list(set(x)),
                'autores': lambda x: list(set([autor for sublist in x for autor in sublist])),
            }).reset_index()
            
            inst_summary.columns = ['Institución', 'Num_Papers', 'Temáticas', 'Autores']
            inst_summary['Num_Autores'] = inst_summary['Autores'].apply(len)
            inst_summary['Num_Temáticas'] = inst_summary['Temáticas'].apply(len)
            inst_summary = inst_summary.sort_values('Num_Papers', ascending=False)
            
            # Top instituciones
            st.subheader("🏆 Top Instituciones Más Productivas")
            top_n_inst = st.slider("Número de instituciones a mostrar:", 5, 30, 15)
            
            top_instituciones = inst_summary.head(top_n_inst)
            
            # Gráfico de barras
            fig = px.bar(
                top_instituciones, 
                x='Num_Papers', 
                y='Institución',
                orientation='h',
                title=f"Top {top_n_inst} Instituciones por Número de Publicaciones"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de diversidad institucional
            st.subheader("📊 Diversidad de Investigación por Institución")
            
            # Gráfico de dispersión: Papers vs Diversidad Temática
            fig_scatter = px.scatter(
                inst_summary.head(20),
                x='Num_Papers',
                y='Num_Temáticas',
                size='Num_Autores',
                hover_name='Institución',
                title="Productividad vs Diversidad Temática",
                labels={
                    'Num_Papers': 'Número de Papers',
                    'Num_Temáticas': 'Número de Temáticas',
                    'Num_Autores': 'Número de Autores'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Información detallada de institución
            st.subheader("🔍 Información Detallada de Institución")
            inst_seleccionada = st.selectbox(
                "Selecciona una institución:",
                [''] + list(inst_summary['Institución'].values)
            )
            
            if inst_seleccionada:
                inst_info = inst_summary[inst_summary['Institución'] == inst_seleccionada].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📄 Papers", inst_info['Num_Papers'])
                with col2:
                    st.metric("👥 Autores", inst_info['Num_Autores'])
                with col3:
                    st.metric("📚 Temáticas", inst_info['Num_Temáticas'])
                
                # Mostrar temáticas y autores principales
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Principales Temáticas:**")
                    for tema in inst_info['Temáticas'][:10]:
                        st.write(f"• {tema}")
                
                with col2:
                    st.write("**Principales Autores:**")
                    for autor in inst_info['Autores'][:10]:
                        st.write(f"• {autor}")
    
    elif analysis_type == "🤝 Análisis de Colaboraciones":
        st.subheader("Análisis de Patrones de Colaboración")
        
        # Análisis de colaboraciones entre autores
        colaboraciones = []
        for _, row in df_filtered.iterrows():
            autores = row['autores_list']
            if len(autores) > 1:
                for i in range(len(autores)):
                    for j in range(i+1, len(autores)):
                        colaboraciones.append({
                            'autor1': autores[i],
                            'autor2': autores[j],
                            'tematica': row['tematica'],
                            'titulo': row['titulo']
                        })
        
        if colaboraciones:
            colab_df = pd.DataFrame(colaboraciones)
            
            # Top colaboraciones
            colab_counts = colab_df.groupby(['autor1', 'autor2']).size().reset_index(name='num_colaboraciones')
            colab_counts = colab_counts.sort_values('num_colaboraciones', ascending=False)
            
            st.subheader("🏆 Top Colaboraciones Entre Autores")
            top_n_colab = st.slider("Número de colaboraciones a mostrar:", 5, 30, 15)
            
            top_colaboraciones = colab_counts.head(top_n_colab)
            top_colaboraciones['colaboracion'] = top_colaboraciones['autor1'] + ' ↔ ' + top_colaboraciones['autor2']
            
            fig = px.bar(
                top_colaboraciones,
                x='num_colaboraciones',
                y='colaboracion',
                orientation='h',
                title=f"Top {top_n_colab} Colaboraciones Más Frecuentes"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de colaboración por temática
            st.subheader("📚 Colaboraciones por Temática")
            
            tematica_colab = colab_df.groupby('tematica').size().reset_index(name='num_colaboraciones')
            tematica_colab = tematica_colab.sort_values('num_colaboraciones', ascending=False).head(15)
            
            fig_tema = px.bar(
                tematica_colab,
                x='num_colaboraciones',
                y='tematica',
                orientation='h',
                title="Colaboraciones por Temática de Investigación"
            )
            fig_tema.update_layout(height=500)
            st.plotly_chart(fig_tema, use_container_width=True)
    
    elif analysis_type == "📚 Análisis de Temáticas":
        st.subheader("Análisis Detallado de Temáticas de Investigación")
        
        # Análisis de temáticas
        tematica_stats = df_filtered.groupby('tematica').agg({
            'titulo': 'count',
            'autores_list': lambda x: list(set([autor for sublist in x for autor in sublist])),
            'afiliaciones_list': lambda x: list(set([inst for sublist in x for inst in sublist if inst])),
            'palabras_clave_list': lambda x: [kw for sublist in x for kw in sublist if kw]
        }).reset_index()
        
        tematica_stats.columns = ['Temática', 'Num_Papers', 'Autores', 'Instituciones', 'Palabras_Clave']
        tematica_stats['Num_Autores'] = tematica_stats['Autores'].apply(len)
        tematica_stats['Num_Instituciones'] = tematica_stats['Instituciones'].apply(len)
        tematica_stats = tematica_stats.sort_values('Num_Papers', ascending=False)
        
        # Top temáticas
        st.subheader("🏆 Temáticas Más Investigadas")
        top_n_temas = st.slider("Número de temáticas a mostrar:", 5, 25, 15)
        
        top_tematicas = tematica_stats.head(top_n_temas)
        
        fig = px.bar(
            top_tematicas,
            x='Num_Papers',
            y='Temática',
            orientation='h',
            title=f"Top {top_n_temas} Temáticas por Número de Papers"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis detallado de temática
        st.subheader("🔍 Análisis Detallado de Temática")
        tema_seleccionada = st.selectbox(
            "Selecciona una temática:",
            [''] + list(tematica_stats['Temática'].values)
        )
        
        if tema_seleccionada:
            tema_info = tematica_stats[tematica_stats['Temática'] == tema_seleccionada].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📄 Papers", tema_info['Num_Papers'])
            with col2:
                st.metric("👥 Autores", tema_info['Num_Autores'])
            with col3:
                st.metric("🏛️ Instituciones", tema_info['Num_Instituciones'])
            
            # Mostrar palabras clave principales
            if tema_info['Palabras_Clave']:
                st.subheader("🔑 Principales Palabras Clave")
                
                # Contar frecuencia de palabras clave
                palabra_freq = Counter(tema_info['Palabras_Clave'])
                
                if len(palabra_freq) > 0:
                    st.write("**Palabras Clave Más Frecuentes:**")
                    for palabra, freq in palabra_freq.most_common(20):
                        st.write(f"• {palabra} ({freq})")
            
            # Top autores e instituciones en esta temática
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Autores:**")
                for autor in tema_info['Autores'][:10]:
                    st.write(f"• {autor}")
            
            with col2:
                st.write("**Top Instituciones:**")
                for inst in tema_info['Instituciones'][:10]:
                    st.write(f"• {inst}")
    
    elif analysis_type == "🔍 Búsqueda Específica":
        st.subheader("Búsqueda Específica en la Base de Datos")
        
        # Opciones de búsqueda
        search_type = st.selectbox(
            "¿Qué quieres buscar?",
            [
                "Buscar por Autor",
                "Buscar por Institución", 
                "Buscar por Palabra Clave",
                "Buscar por Título"
            ]
        )
        
        search_term = st.text_input("Introduce el término de búsqueda:")
        
        if search_term:
            if search_type == "Buscar por Autor":
                # Buscar papers que contengan el autor
                mask = df_filtered['autores_normalizados'].str.contains(search_term, case=False, na=False)
                resultados = df_filtered[mask]
                
            elif search_type == "Buscar por Institución":
                mask = df_filtered['afiliaciones_normalizadas'].str.contains(search_term, case=False, na=False)
                resultados = df_filtered[mask]
                
            elif search_type == "Buscar por Palabra Clave":
                mask = df_filtered['palabras_clave'].str.contains(search_term, case=False, na=False)
                resultados = df_filtered[mask]
                
            elif search_type == "Buscar por Título":
                mask = df_filtered['titulo'].str.contains(search_term, case=False, na=False)
                resultados = df_filtered[mask]
            
            if len(resultados) > 0:
                st.success(f"Se encontraron {len(resultados)} resultados para '{search_term}'")
                
                # Mostrar resultados
                for i, (_, row) in enumerate(resultados.iterrows(), 1):
                    with st.expander(f"Resultado {i}: {row['titulo'][:80]}..."):
                        st.write(f"**Título:** {row['titulo']}")
                        st.write(f"**Autores:** {row['autores_normalizados']}")
                        st.write(f"**Instituciones:** {row['afiliaciones_normalizadas']}")
                        st.write(f"**Temática:** {row['tematica']}")
                        if row['palabras_clave']:
                            st.write(f"**Palabras Clave:** {row['palabras_clave']}")
            else:
                st.warning(f"No se encontraron resultados para '{search_term}'")
