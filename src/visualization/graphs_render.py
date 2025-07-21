import streamlit as st
import os
import networkx as nx

# Try to import pyvis, fallback to plotly if not available
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    st.warning("⚠️ pyvis no está disponible. Usando visualización alternativa con plotly.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def show_networkx_graph_plotly(G, height=600, width=900):
    """
    Visualización alternativa usando plotly
    """
    import numpy as np
    
    if not PLOTLY_AVAILABLE:
        st.error("Ni pyvis ni plotly están disponibles para la visualización.")
        return
    
    # Calcular layout
    try:
        if G.number_of_nodes() <= 100:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), iterations=50)
    except:
        pos = nx.circular_layout(G)
    
    # Preparar datos para plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Crear trace para aristas
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=0.5, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    # Preparar datos para nodos
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Información del nodo
        degree = G.degree(node)
        node_text.append(f"{node}<br>Conexiones: {degree}")
        
        # Color del nodo
        node_data = G.nodes[node]
        if 'color' in node_data:
            node_colors.append(node_data['color'])
        else:
            node_colors.append('#4A90E2')
    
    # Crear trace para nodos
    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers',
                           hoverinfo='text',
                           text=node_text,
                           marker=dict(size=10,
                                     color=node_colors,
                                     line=dict(width=2)))
    
    # Crear figura
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                        title='Red de Colaboración',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Visualización interactiva con plotly",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color="#888", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=height))
    
    st.plotly_chart(fig, use_container_width=True)

def show_networkx_graph(G, height=600, width=900, show_info_expander=True):
    """
    Muestra un grafo de NetworkX usando pyvis en Streamlit, con fallback a plotly
    """
    import numpy as np
    
    # Verificar que el grafo no esté vacío
    if G.number_of_nodes() == 0:
        st.warning("El grafo no tiene nodos para mostrar.")
        return
    
    # Mostrar información básica del grafo
    if show_info_expander:
        with st.expander("ℹ️ Información del Grafo", expanded=False):
            st.write(f"**Nodos:** {G.number_of_nodes()}")
            st.write(f"**Aristas:** {G.number_of_edges()}")
            st.write(f"**Tipo:** {'Dirigido' if G.is_directed() else 'No dirigido'}")
            
            # Mostrar tipos de nodos si existen
            node_types = set()
            for _, data in G.nodes(data=True):
                if 'node_type' in data:
                    node_types.add(data['node_type'])
            if node_types:
                st.write(f"**Tipos de nodos:** {', '.join(node_types)}")
    else:
        # Mostrar información básica sin expander
        st.write(f"**Nodos:** {G.number_of_nodes()}")
        st.write(f"**Aristas:** {G.number_of_edges()}")
        st.write(f"**Tipo:** {'Dirigido' if G.is_directed() else 'No dirigido'}")
        
        # Mostrar tipos de nodos si existen
        node_types = set()
        for _, data in G.nodes(data=True):
            if 'node_type' in data:
                node_types.add(data['node_type'])
        if node_types:
            st.write(f"**Tipos de nodos:** {', '.join(node_types)}")
    
    # Si pyvis no está disponible, usar plotly
    if not PYVIS_AVAILABLE:
        show_networkx_graph_plotly(G, height, width)
        return
    
    # Layout robusto y seguro
    pos = None
    try:
        if G.number_of_nodes() <= 100:
            try:
                pos = nx.kamada_kawai_layout(G, scale=1500)
            except Exception:
                pos = None
        
        if pos is None:
            try:
                k_value = 2/np.sqrt(G.number_of_nodes()) if G.number_of_nodes() > 1 else 1
                pos = nx.spring_layout(G, k=k_value, iterations=50, scale=1500)
            except Exception:
                pos = None
        
        if pos is None:
            try:
                # Para grafos desconectados
                if not G.is_directed():
                    components = list(nx.connected_components(G))
                else:
                    components = list(nx.weakly_connected_components(G))
                
                if len(components) > 1:
                    pos = {}
                    angle_step = 2 * np.pi / len(components)
                    for i, component in enumerate(components):
                        subgraph = G.subgraph(component)
                        if len(component) > 3:
                            try:
                                sub_pos = nx.spring_layout(subgraph, k=2, iterations=50)
                            except:
                                sub_pos = nx.circular_layout(subgraph)
                        else:
                            sub_pos = nx.circular_layout(subgraph)
                        
                        center_x = 1200 * np.cos(i * angle_step)
                        center_y = 1200 * np.sin(i * angle_step)
                        for node, (x, y) in sub_pos.items():
                            pos[node] = (center_x + x * 500, center_y + y * 500)
                else:
                    pos = nx.spring_layout(G, k=4, iterations=50, scale=1500)
            except Exception:
                pos = None
        
        if pos is None:
            pos = nx.random_layout(G, scale=1500)
            
    except Exception as e:
        st.error(f"Error al calcular el layout del grafo: {str(e)}")
        # Layout de emergencia
        pos = {}
        nodes = list(G.nodes())
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            pos[node] = (1000 * np.cos(angle), 1000 * np.sin(angle))

    # Asignar posiciones y atributos visuales y hovers personalizados
    # Detectar bipartito: autor-campo o autor-institucion
    node_types = set(d.get('node_type') for n, d in G.nodes(data=True))
    is_bipartito = (('author' in node_types) and (('field' in node_types) or ('institution' in node_types)))
    bipartito_central = None
    if is_bipartito:
        if 'field' in node_types:
            bipartito_central = 'field'
        elif 'institution' in node_types:
            bipartito_central = 'institution'

    for node, (x, y) in pos.items():
        G.nodes[node]['x'] = x
        G.nodes[node]['y'] = y
        # Hover personalizado según tipo de red (texto plano)
        if not G.is_directed():
            degree = G.degree(node)
            G.nodes[node]['title'] = f"{node} ({degree} conexiones)"
        else:
            is_principal_secundario = any('principal_secundaria' in data for _, _, data in G.edges(data=True))
            if is_principal_secundario:
                outd = G.out_degree(node)
                ind = G.in_degree(node)
                secundarios = set([u for u, v in G.in_edges(node)])
                principales = set([v for u, v in G.out_edges(node)])
                G.nodes[node]['title'] = f"{node}\nPrincipal: {outd} (a {len(principales)} personas)\nSecundario: {ind} (por {len(secundarios)} personas)"
            else:
                ind = G.in_degree(node)
                citadores = set([u for u, v in G.in_edges(node)])
                G.nodes[node]['title'] = f"{node}\n{ind} veces citado\n{len(citadores)} personas distintas lo han citado"
        # Coloreado bipartito: central en rojo, resto azul
        if is_bipartito:
            node_type = G.nodes[node].get('node_type', 'default')
            if node_type == bipartito_central:
                G.nodes[node]['color'] = '#FF3C3C'  # rojo
            else:
                G.nodes[node]['color'] = '#4A90E2'  # azul
        else:
            if 'color' not in G.nodes[node]:
                node_type = G.nodes[node].get('node_type', 'default')
                if node_type == 'paper':
                    G.nodes[node]['color'] = '#FF6B6B'
                elif node_type == 'field':
                    G.nodes[node]['color'] = '#FF6B6B'
                elif node_type == 'institution':
                    pass
                elif node_type == 'keyword':
                    G.nodes[node]['color'] = '#FF6B6B'
                elif node_type == 'author':
                    G.nodes[node]['color'] = '#4A90E2'
                else:
                    G.nodes[node]['color'] = '#A8E6CF'

    # Gradiente de color en aristas y tooltips
    if G.is_directed():
        edge_weights = []
        for u, v, data in G.edges(data=True):
            w1 = data.get('weight', 1)
            w2 = G[v][u].get('weight', 0) if G.has_edge(v, u) else 0
            edge_weights.append(w1 + w2)
        max_weight = max(edge_weights) if edge_weights else 1
    else:
        edge_weights = [data.get('weight', 1) for u, v, data in G.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1

    # Calcular min y max de pesos reales para el gradiente
    if G.is_directed():
        edge_weights = []
        for u, v, data in G.edges(data=True):
            w1 = data.get('weight', 1)
            w2 = G[v][u].get('weight', 0) if G.has_edge(v, u) else 0
            edge_weights.append(w1 + w2)
        min_weight = min(edge_weights) if edge_weights else 1
        max_weight = max(edge_weights) if edge_weights else 1
    else:
        edge_weights = [data.get('weight', 1) for u, v, data in G.edges(data=True)]
        min_weight = min(edge_weights) if edge_weights else 1
        max_weight = max(edge_weights) if edge_weights else 1

    for u, v, data in G.edges(data=True):
        data['width'] = 1
        if G.is_directed():
            peso_uv = data.get('weight', 1)
            peso_vu = G[v][u].get('weight', 0) if G.has_edge(v, u) else 0
            total = peso_uv + peso_vu
            ps_uv = data.get('principal_secundaria', None)
            ps_vu = G[v][u].get('principal_secundaria', None) if G.has_edge(v, u) else None
            if ps_uv is not None or ps_vu is not None:
                title = f"{u} → {v}: {ps_uv or 0} (Total: {peso_uv}) | {v} → {u}: {ps_vu or 0} (Total: {peso_vu})"
            else:
                title = f"{u} → {v}: {peso_uv} | {v} → {u}: {peso_vu}"
            G[u][v]['title'] = title
            # Usar min y max reales para el gradiente
            if max_weight > min_weight:
                ratio = (total - min_weight) / (max_weight - min_weight)
            else:
                ratio = 0
            # Gradiente de azul a rojo (no verde)
            r1, g1, b1 = (0x4A, 0x90, 0xE2)
            r2, g2, b2 = (0xFF, 0x3C, 0x3C)  # rojo claro
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            data['color'] = f'rgb({r},{g},{b})'
        else:
            peso = data.get('weight', 1)
            G[u][v]['title'] = f"{u} — {v}: {peso}"
            if max_weight > min_weight:
                ratio = (peso - min_weight) / (max_weight - min_weight)
            else:
                ratio = 0
            r1, g1, b1 = (0x4A, 0x90, 0xE2)
            r2, g2, b2 = (0xFF, 0x3C, 0x3C)
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            data['color'] = f'rgb({r},{g},{b})'
    # COLOREADO DE NODOS SEGÚN GRADO O IN/OUT DEGREE (solo si no es bipartito)
    if not is_bipartito:
        if not G.is_directed():
            degrees = dict(G.degree())
            if degrees:
                min_deg = min(degrees.values())
                max_deg = max(degrees.values())
                for n in G.nodes():
                    if max_deg > min_deg:
                        ratio = (degrees[n] - min_deg) / (max_deg - min_deg)
                    else:
                        ratio = 0
                    r1, g1, b1 = (0x4A, 0x90, 0xE2)
                    r2, g2, b2 = (0xFF, 0x3C, 0x3C)
                    r = int(r1 + (r2 - r1) * ratio)
                    g = int(g1 + (g2 - g1) * ratio)
                    b = int(b1 + (b2 - b1) * ratio)
                    G.nodes[n]['color'] = f'rgb({r},{g},{b})'
        else:
            in_deg = dict(G.in_degree())
            out_deg = dict(G.out_degree())
            if in_deg:
                min_in = min(in_deg.values())
                max_in = max(in_deg.values())
            else:
                min_in = max_in = 0
            if out_deg:
                min_out = min(out_deg.values())
                max_out = max(out_deg.values())
            else:
                min_out = max_out = 0
            for n in G.nodes():
                if max_in > min_in:
                    r_ratio = (in_deg.get(n, 0) - min_in) / (max_in - min_in)
                else:
                    r_ratio = 0
                if max_out > min_out:
                    o_ratio = (out_deg.get(n, 0) - min_out) / (max_out - min_out)
                else:
                    o_ratio = 0
                r = int(0x4A + (0xFF - 0x4A) * max(r_ratio, o_ratio))
                g = int(0x90 + (0xB3 - 0x90) * o_ratio)
                b = int(0xE2 * (1 - max(r_ratio, o_ratio)))
                G.nodes[n]['color'] = f'rgb({r},{g},{b})'

    net = Network(height=f"{height}px", width=f"{width}px", notebook=False, directed=G.is_directed())
    net.from_nx(G)

    if G.is_directed():
        net.set_options('''{
          "physics": {"enabled": false, "stabilization": {"enabled": false}},
          "interaction": {"dragNodes": true, "dragView": true, "zoomView": true, "selectConnectedEdges": true, "hover": true},
          "edges": {"smooth": {"enabled": true, "type": "continuous", "roundness": 0.3}, "width": 1, "color": {"color": "#848484", "highlight": "#ff6b6b", "hover": "#333333"}, "arrows": {"to": {"enabled": true, "scaleFactor": 0.4, "type": "arrow"}}},
          "nodes": {"font": {"size": 0, "color": "transparent"}, "borderWidth": 1, "borderWidthSelected": 2, "size": 6, "fixed": {"x": true, "y": true}, "physics": false, "color": {"border": "#4ECDC4", "background": "#A8E6CF", "highlight": {"border": "#FF6B6B", "background": "#FFD93D"}, "hover": {"border": "#4ECDC4", "background": "#6BCF7F"}}, "shadow": {"enabled": false}},
          "layout": {"improvedLayout": false, "randomSeed": 42},
          "configure": {"enabled": false}
        }''')
    else:
        net.set_options('''{
          "physics": {"enabled": false, "stabilization": {"enabled": false}},
          "interaction": {"dragNodes": true, "dragView": true, "zoomView": true, "selectConnectedEdges": true, "hover": true},
          "edges": {"smooth": {"enabled": true, "type": "continuous", "roundness": 0.3}, "width": 1, "color": {"color": "#848484", "highlight": "#ff6b6b", "hover": "#333333"}},
          "nodes": {"font": {"size": 0, "color": "transparent"}, "borderWidth": 1, "borderWidthSelected": 2, "size": 6, "fixed": {"x": true, "y": true}, "physics": false, "color": {"border": "#4ECDC4", "background": "#A8E6CF", "highlight": {"border": "#FF6B6B", "background": "#FFD93D"}, "hover": {"border": "#4ECDC4", "background": "#6BCF7F"}}, "shadow": {"enabled": false}},
          "layout": {"improvedLayout": false, "randomSeed": 42},
          "configure": {"enabled": false}
        }''')

    # Create a temporary directory in the current working directory
    temp_dir = os.path.join(os.getcwd(), 'webapp', 'data')
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, 'tmp_graph.html')
    
    try:
        # Save the graph to HTML file
        net.save_graph(temp_file)
        
        # Read the HTML content
        with open(temp_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Display the graph using Streamlit components
        st.components.v1.html(html_content, height=height+50, scrolling=True)
        
    except Exception as e:
        st.error(f"Error al mostrar el grafo: {str(e)}")
        st.info("Intentando mostrar información básica del grafo...")
        
        # Fallback: show basic graph information
        st.write(f"**Nodos:** {G.number_of_nodes()}")
        st.write(f"**Aristas:** {G.number_of_edges()}")
        
        if G.number_of_nodes() > 0:
            # Show a simple table of nodes
            nodes_data = []
            for node in list(G.nodes())[:10]:  # Show first 10 nodes
                degree = G.degree(node) if not G.is_directed() else f"In: {G.in_degree(node)}, Out: {G.out_degree(node)}"
                nodes_data.append({"Nodo": str(node), "Grado": str(degree)})
            
            if nodes_data:
                import pandas as pd
                df = pd.DataFrame(nodes_data)
                st.dataframe(df)
                
                if G.number_of_nodes() > 10:
                    st.info(f"Mostrando los primeros 10 nodos de {G.number_of_nodes()} totales.")
