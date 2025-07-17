import networkx as nx

# ======================
# GRAFOS DE COLABORACIÓN
# ======================
def build_coauthor_graph(articulos):
    """Grafo no dirigido de coautorías. 
        - Nodos: autores (color azul #4A90E2)
        - Enlaces: colaboraciones en artículos (con peso por frecuencia)"""
    G = nx.Graph()
    for art in articulos:
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        autores = list(set([a for a in autores if a]))  # Eliminar duplicados y vacíos
        # Añadir enlaces entre todos los pares de autores
        for i in range(len(autores)):
            for j in range(i+1, len(autores)):
                a1, a2 = autores[i], autores[j]
                if G.has_edge(a1, a2):
                    G[a1][a2]['weight'] += 1
                else:
                    G.add_edge(a1, a2, weight=1)
    # Atributos de nodos
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'author'
        G.nodes[node]['color'] = '#4A90E2'
    return G

def build_principal_secondary_graph(articulos):
    """Grafo DIRIGIDO de relaciones principal-secundario. 
        - Nodos: autores (color azul #4A90E2)
        - Enlaces: principal -> secundario (con peso por frecuencia)"""
    G = nx.DiGraph()
    for art in articulos:
        principales = art.get('Autores Principales', [])
        secundarios = art.get('Autores Secundarios', [])
        # Añadir enlaces de principal a secundario
        for principal in principales:
            if principal:
                for secundario in secundarios:
                    if secundario and principal != secundario:
                        if G.has_edge(principal, secundario):
                            G[principal][secundario]['weight'] += 1
                        else:
                            G.add_edge(principal, secundario, weight=1)
    # Atributos de nodos
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'author'
        G.nodes[node]['color'] = '#4A90E2'
    return G

# ======================
# GRAFOS DE INSTITUCIONES
# ======================
def build_institution_institution_graph(articulos):
    """Grafo no dirigido de colaboraciones entre instituciones. 
        - Nodos: instituciones (color verde #50C878)
        - Enlaces: coaparición en artículos (con peso)"""
    G = nx.Graph()
    for art in articulos:
        instituciones = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
        instituciones = [i for i in instituciones if i]  # Filtrar vacíos
        # Añadir enlaces entre todas las instituciones del artículo
        for i in range(len(instituciones)):
            for j in range(i+1, len(instituciones)):
                inst1, inst2 = instituciones[i], instituciones[j]
                if G.has_edge(inst1, inst2):
                    G[inst1][inst2]['weight'] += 1
                else:
                    G.add_edge(inst1, inst2, weight=1)
    # Atributos de nodos
    for node in G.nodes():
        G.nodes[node]['node_type'] = 'institution'
        G.nodes[node]['color'] = '#50C878'
    return G

def build_institution_author_graph(articulos):
    """Grafo bipartito institución-autor. 
        - Nodos: instituciones (verde #50C878) y autores (azul #4A90E2)
        - Enlaces: autor pertenece a institución"""
    G = nx.Graph()
    for art in articulos:
        instituciones = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        # Añadir nodos y enlaces
        for inst in instituciones:
            if inst:
                G.add_node(inst, node_type='institution', color='#50C878')
                for autor in autores:
                    if autor:
                        G.add_node(autor, node_type='author', color='#4A90E2')
                        G.add_edge(inst, autor)
    return G

# ======================
# GRAFOS TEMÁTICOS
# ======================

def build_field_author_graph(articulos):
    """Grafo bipartito campo de estudio-autor. 
        - Nodos: campos (rojo #FF6B6B) y autores (azul #4A90E2)
        - Enlaces: autor publica en campo"""
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
        if campo:
            G.add_node(campo, node_type='field', color='#FF6B6B')
            for autor in autores:
                if autor:
                    G.add_node(autor, node_type='author', color='#4A90E2')
                    G.add_edge(campo, autor)
    return G

def build_field_institution_graph(articulos):
    """Grafo bipartito campo de estudio-institución. 
        - Nodos: campos (rojo #FF6B6B) e instituciones (verde #50C878)
        - Enlaces: institución publica en campo"""
    G = nx.Graph()
    for art in articulos:
        campo = art.get('Campo de Estudio', '')
        instituciones = [art.get('Institucion Principal', '')] + art.get('Instituciones Secundarias', [])
        if campo:
            G.add_node(campo, node_type='field', color='#FF6B6B')
            for inst in instituciones:
                if inst:
                    G.add_node(inst, node_type='institution', color='#50C878')
                    G.add_edge(campo, inst)
    return G