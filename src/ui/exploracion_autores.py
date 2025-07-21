import streamlit as st
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from ..visualization.graphs import build_coauthor_graph, build_author_citation_graph, build_principal_secondary_graph
from ..visualization.graphs_render import show_networkx_graph
from ..analysis.DataScience import (
    resumen_narrativo_autor_autor,
    resumen_narrativo_citaciones,
    resumen_narrativo_principal_secundario,
    resumen_narrativo_autor_campo,
    resumen_narrativo_autor_institucion
)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from .metricas_visuales import crear_dashboard_metricas_grafo
from .metricas_narrativas_visuales import crear_seccion_metricas_visuales
import math

def autores_tab(articulos):
    # --- Type check to prevent 'int' object is not iterable error ---
    if not isinstance(articulos, list):
        st.error(f"Error: 'articulos' is not a list. Type: {type(articulos)}, Value: {articulos}")
        return
    def build_community_author_institution_graph(comm_nodes, autor_info, color):
        import networkx as nx
        G_bip = nx.Graph()
        # Añadir autores
        for n in comm_nodes:
            G_bip.add_node(n, color=color, node_type='author')
        # Añadir instituciones y aristas autor-institución
        for n in comm_nodes:
            insts = autor_info.get(n, {}).get('instituciones', [])
            for inst in insts:
                if inst not in G_bip:
                    G_bip.add_node(inst, color="#FFD700", node_type='institution')
                G_bip.add_edge(n, inst)
        return G_bip
    # st.set_page_config(layout="wide")  # Removed: should only be called once at the top level
    # Verificar si hay una red preseleccionada desde la nueva interfaz
    if 'red_seleccionada' in st.session_state:
        tipo_red = st.session_state.red_seleccionada
        st.subheader(f"Análisis: {tipo_red}")
        # Mostrar información sobre la red seleccionada
        if tipo_red == "Red de Colaboración Autor-Autor":
            pass
        elif tipo_red == "Red de Autores Principales-Secundarios":
            pass
        elif tipo_red == "Red Autor-Campo de Estudio":
            pass
        elif tipo_red == "Red Autor-Institución":
            pass
    else:
        # Interfaz original como fallback
        st.subheader("Exploración de Autores")
        opciones = [
            "Red de Colaboración Autor-Autor",
            "Red Autor-Institución", 
            "Red Autor-Campo de Estudio",
            "Red de Autores Principales-Secundarios"
        ]
        tipo_red = st.selectbox("Selecciona el tipo de red de autores a visualizar:", opciones)

    # Explicación contextual según el tipo de red
    def clean_edge_titles_plaintext(G):
        # For all edges, ensure 'title' is plain text (no HTML, no <b>, no <br>, etc.)
        import re
        for data in (edata for _, _, edata in G.edges(data=True)):
            if 'title' in data:
                txt = str(data['title'])
                txt = re.sub(r'<[^>]+>', '', txt)
                txt = txt.replace('<', '').replace('>', '')
                txt = txt.replace('&lt;', '').replace('&gt;', '')
                txt = txt.replace('  ', ' ').strip()
                data['title'] = txt
        return G

    # --- Obtener todos los autores presentes en el JSON, aunque no tengan colaboraciones ---
    autores_json = set()
    for art in articulos:
        for autor in art.get('Autores Principales', []):
            autores_json.add(autor)
        for autor in art.get('Autores Secundarios', []):
            autores_json.add(autor)
    if tipo_red == "Red de Colaboración Autor-Autor":
        st.markdown("""
        **Composición de la red:** Los **nodos** representan investigadores individuales, mientras que las **aristas** conectan 
        autores que han colaborado en al menos una publicación conjunta. El **grosor y color** de las conexiones indica la 
        intensidad de la colaboración: conexiones rojizas muestran colaboraciones frecuentes, mientras que las azules representan 
        colaboraciones ocasionales. Explora la red para identificar comunidades de investigación y autores centrales en el ecosistema científico.
        """)
        G = build_coauthor_graph(articulos)
        # Añadir explícitamente los nodos huérfanos
        for autor in autores_json:
            if autor not in G:
                G.add_node(autor, node_type='author')
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red de Autores Principales-Secundarios":
        st.markdown("""
        **Composición de la red:** Los **nodos** representan investigadores, y las **aristas dirigidas** (flechas) van desde 
        autores principales hacia autores secundarios en cada publicación. El **color y grosor** de las flechas indica la 
        frecuencia de estas relaciones jerárquicas. Esta red revela patrones de mentoría, liderazgo académico y estructuras 
        de poder en la colaboración científica, permitiendo identificar investigadores influyentes y emergentes.
        """)
        G = build_principal_secondary_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red Autor-Campo de Estudio":
        st.markdown("""
        **Composición de la red:** Esta red **bipartita** conecta dos tipos de nodos: **investigadores** (círculos) y 
        **campos de estudio** (cuadrados). Las **aristas** unen autores con las disciplinas en las que han publicado. 
        El **grosor de las conexiones** refleja la productividad en cada área. Analiza la versatilidad temática de los 
        investigadores, identifica especialistas vs. generalistas, y descubre áreas de conocimiento emergentes o consolidadas.
        """)
        from ..visualization.graphs import build_author_field_graph
        G = build_author_field_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red Autor-Institución":
        st.markdown("""
        **Composición de la red:** Red **bipartita** que conecta **investigadores** (círculos) con **instituciones** (cuadrados) 
        donde han publicado. Las **aristas** representan afiliaciones académicas, y su **grosor** indica la intensidad de la 
        relación (número de publicaciones). Explora la movilidad académica, identifica instituciones centrales en el ecosistema 
        científico, y analiza patrones de colaboración interinstitucional a través de investigadores compartidos.
        """)
        from ..visualization.graphs import build_author_institution_graph
        G = build_author_institution_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    else:
        G = None

    if G is not None:

        # --- Sanitize edge tooltips (hover) for all edges ---
        import re
        def sanitize_hover(text):
            # Remove all HTML tags and escape angle brackets
            text = re.sub(r'<.*?>', '', str(text))
            text = text.replace('<', '').replace('>', '')
            return text
        for u, v, data in G.edges(data=True):
            if 'title' in data:
                data['title'] = sanitize_hover(data['title'])
        for n, data in G.nodes(data=True):
            if 'title' in data:
                data['title'] = sanitize_hover(data['title'])
        from ..visualization.graphs_render import show_networkx_graph
        # --- Center the graph visualization (pass center=True if supported, else default) ---
        show_networkx_graph(G, show_info_expander=False)
        # Resumen narrativo debajo del grafo
        from ..analysis.DataScience import (
            resumen_narrativo_autor_autor,
            resumen_narrativo_citaciones,
            resumen_narrativo_principal_secundario,
            resumen_narrativo_autor_campo,
            resumen_narrativo_autor_institucion
        )
        resumen = None
        if tipo_red == "Red de Colaboración Autor-Autor":
            # Detectar autores sin colaboraciones según el JSON, no solo el grafo
            autores_sin_colab = [autor for autor in autores_json if G.degree(autor) == 0]
            resumen = resumen_narrativo_autor_autor(G)
            if autores_sin_colab:
                resumen += f"<br><b>Autores sin colaboraciones:</b> Hay {len(autores_sin_colab)} autores que no han colaborado con ningún otro en la red."
            else:
                resumen += "<br><b>Autores sin colaboraciones:</b> Todos los autores han colaborado al menos una vez."
        elif tipo_red == "Red de Autores Principales-Secundarios":
            resumen = resumen_narrativo_principal_secundario(G)
        elif tipo_red == "Red Autor-Campo de Estudio":
            resumen = resumen_narrativo_autor_campo(G)
        elif tipo_red == "Red Autor-Institución":
            resumen = resumen_narrativo_autor_institucion(G)
        if resumen:
            # Crear visualización interactiva de métricas en lugar del texto plano
            crear_seccion_metricas_visuales(G, tipo_red, autores_sin_colab if tipo_red == "Red de Colaboración Autor-Autor" else [])

        # --- Contenedor: Perfiles de Investigadores ---
        with st.expander("👥 **Perfiles de Investigadores**", expanded=False):
            # Recopilar todos los autores y su info
            autor_info = defaultdict(lambda: {
                'instituciones': set(),
                'campos': set(),
                'articulos': [],
                'principal': 0,
                'secundario': 0,
                'palabras': [],
                'citas': 0
            })
            
            for art in articulos:
                campo = art.get('Campo de Estudio')
                inst_princ = art.get('Institucion Principal')
                inst_secs = art.get('Instituciones Secundarias', [])
                palabras = art.get('Palabras Clave', [])
                # Autores principales
                for autor in art.get('Autores Principales', []):
                    autor_info[autor]['campos'].add(campo)
                    if inst_princ:
                        autor_info[autor]['instituciones'].add(inst_princ)
                    autor_info[autor]['instituciones'].update(inst_secs)
                    autor_info[autor]['articulos'].append((art, 'Principal'))
                    autor_info[autor]['principal'] += 1
                    autor_info[autor]['palabras'] += palabras
                # Autores secundarios
                for autor in art.get('Autores Secundarios', []):
                    autor_info[autor]['campos'].add(campo)
                    if inst_princ:
                        autor_info[autor]['instituciones'].add(inst_princ)
                    autor_info[autor]['instituciones'].update(inst_secs)
                    autor_info[autor]['articulos'].append((art, 'Secundario'))
                    autor_info[autor]['secundario'] += 1
                    autor_info[autor]['palabras'] += palabras
            
            # Campo principal de cada autor
            campo_principal_autor = {}
            for autor, info in autor_info.items():
                campos_autor = [art[0].get('Campo de Estudio') for art in info['articulos']]
                if campos_autor:
                    campo_principal_autor[autor] = Counter(campos_autor).most_common(1)[0][0]
                else:
                    campo_principal_autor[autor] = None
            
            # Colaboradores con campo principal distinto
            G_coaut = build_coauthor_graph(articulos)
            colaboradores_distinto_campo = {}
            for autor in autor_info:
                vecinos = set(G_coaut.neighbors(autor)) if autor in G_coaut else set()
                mi_campo = campo_principal_autor.get(autor)
                count = 0
                for v in vecinos:
                    if v in campo_principal_autor and campo_principal_autor[v] and campo_principal_autor[v] != mi_campo:
                        count += 1
                colaboradores_distinto_campo[autor] = count
            
            # Tabla de ranking
            datos = []
            for autor, info in autor_info.items():
                total_colab = G_coaut.degree(autor) if autor in G_coaut else 0
                datos.append({
                    'Total de colaboraciones': total_colab,
                    'Autor': autor,
                    'Instituciones': len(info['instituciones']),
                    'Campos de estudio': len(info['campos']),
                    'Colaboradores con campo principal distinto': colaboradores_distinto_campo.get(autor, 0),
                    'Artículos como principal': info['principal'],
                    'Artículos como secundario': info['secundario'],
                })
            
            df = pd.DataFrame(datos)
            cols = ['Autor', 'Total de colaboraciones'] + [c for c in df.columns if c not in ['Autor', 'Total de colaboraciones']]
            df = df[cols].sort_values(['Total de colaboraciones', 'Artículos como principal', 'Artículos como secundario'], ascending=False)
            
            # Métricas generales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Investigadores", len(df))
            with col2:
                st.metric("Colaboraciones Promedio", f"{df['Total de colaboraciones'].mean():.1f}")
            with col3:
                st.metric("Máximo de Colaboraciones", df['Total de colaboraciones'].max())
            
            # Mostrar tabla con estilo mejorado
            st.markdown("**Ranking de Investigadores:**")
            st.dataframe(
                df.style.background_gradient(subset=['Total de colaboraciones'], cmap='Blues'),
                hide_index=True,
                use_container_width=True
            )

            # --- Búsqueda de autores ---
            autores_lista = sorted(list(autor_info.keys()))
            autor_sel = st.selectbox("Seleccionar autor", autores_lista, key=f"busqueda_autor_{hash(str(st.session_state.get('red_seleccionada', 'default'))) % 1000}")
            if autor_sel:
                col1, col2 = st.columns([2, 1], gap="large")
                info = autor_info[autor_sel]
                vecinos = list(G_coaut.neighbors(autor_sel)) if autor_sel in G_coaut else []
                campo_princ = campo_principal_autor.get(autor_sel)
                
                # --- Perfil resumido ---
                with col1:
                    st.markdown(f"### {autor_sel}")
                    total_colab = len(vecinos)
                    inst_count = len(info['instituciones'])
                    campos_count = len(info['campos'])
                    art_count = len(info['articulos'])
                    
                    # --- Resumen conciso en formato de puntos ---
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("📄 Artículos", art_count)
                        st.metric("🤝 Colaboraciones", total_colab)
                    with col_b:
                        st.metric("🏛️ Instituciones", inst_count)
                        st.metric("🎓 Campos", campos_count)
                    
                    # Campo principal
                    if campo_princ:
                        st.markdown(f"**🎯 Campo Principal:** {campo_princ}")
                    
                    # --- Secciones con toggles ---
                    # Artículos del autor
                    if info['articulos']:
                        mostrar_articulos_autor = st.toggle(f"📚 Mostrar artículos del autor ({len(info['articulos'])})", key=f"toggle_articulos_autor_{autor_sel}_{tipo_red}")
                        if mostrar_articulos_autor:
                            st.markdown("**Artículos:**")
                            for art, _ in info['articulos']:
                                titulo = art.get('Nombre de Articulo', '-')
                                st.markdown(f"• {titulo}")
                    
                    # Instituciones del autor
                    insts = list(info['instituciones'])
                    if insts:
                        mostrar_instituciones_autor = st.toggle(f"🏛️ Mostrar instituciones del autor ({len(insts)})", key=f"toggle_instituciones_autor_{autor_sel}_{tipo_red}")
                        if mostrar_instituciones_autor:
                            st.markdown("**Instituciones:**")
                            for inst in sorted(insts):
                                count = 0
                                for art, _ in info['articulos']:
                                    if art.get('Institucion Principal') == inst or inst in art.get('Instituciones Secundarias', []):
                                        count += 1
                                st.markdown(f"• **{inst}** ({count} colaboraciones)")
                    
                    # Colaboradores del autor
                    if vecinos:
                        mostrar_colaboradores_autor = st.toggle(f"👥 Mostrar colaboradores del autor ({len(vecinos)})", key=f"toggle_colaboradores_autor_{autor_sel}_{tipo_red}")
                        if mostrar_colaboradores_autor:
                            st.markdown("**Colaboradores:**")
                            for v in vecinos:
                                campo_colab = campo_principal_autor.get(v, '-')
                                st.markdown(f"• **{v}** ({campo_colab})")
                
                with col2:
                    # Red de colaboraciones mejorada
                    if vecinos:
                        try:
                            subnodos = [autor_sel] + vecinos
                            subG = G_coaut.subgraph(subnodos).copy()
                            
                            # Limpiar tooltips
                            for u, v, data in subG.edges(data=True):
                                if 'title' in data:
                                    data['title'] = sanitize_hover(data['title'])
                            for n, data in subG.nodes(data=True):
                                if 'title' in data:
                                    data['title'] = sanitize_hover(data['title'])
                            
                            st.markdown("**🕸️ Red de Colaboraciones:**")
                            show_networkx_graph(subG, height=350, width=350, show_info_expander=False)
                        except Exception as e:
                            st.warning(f"Error al mostrar el subgrafo: {str(e)}")
                    else:
                        st.info("Este autor no tiene colaboraciones registradas.")

                # --- Nube de palabras compacta ---
                corpus = []
                for art in articulos:
                    palabras_art = art.get('Palabras Clave', [])
                    if palabras_art:
                        corpus.append([p.strip() for p in palabras_art if p.strip()])
                
                # TF-IDF para palabras del autor
                all_phrases = set()
                for frases in corpus:
                    all_phrases.update(frases)
                
                df_counter = Counter()
                for frases in corpus:
                    for f in set(frases):
                        df_counter[f] += 1
                N = len(corpus)
                
                autor_phrases = [p.strip() for p in info['palabras'] if p.strip()]
                if len(set(autor_phrases)) < 10:
                    extra = []
                    for v in vecinos:
                        v_info = autor_info.get(v, {})
                        v_phrases = [p.strip() for p in v_info.get('palabras', []) if p.strip()]
                        for p in v_phrases:
                            if p not in autor_phrases and p not in extra:
                                extra.append(p)
                            if len(set(autor_phrases + extra)) >= 10:
                                break
                        if len(set(autor_phrases + extra)) >= 10:
                            break
                    autor_phrases = list(set(autor_phrases + extra))
                
                tfidf = {}
                for phrase in autor_phrases:
                    tf = autor_phrases.count(phrase)
                    idf = math.log((N + 1) / (1 + df_counter.get(phrase, 0))) + 1
                    tfidf[phrase] = tf * idf
                
                top_phrases = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:10]
                
                if top_phrases:
                    wc = WordCloud(width=800, height=300, background_color='white', collocations=False, prefer_horizontal=0.5)
                    wc.generate_from_frequencies(dict(top_phrases))
                    st.markdown("<div style='text-align:center'><b>🔬 Áreas de Investigación:</b></div>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

        # --- Contenedor: Comunidades ---
        with st.expander("🌐 **Comunidades**", expanded=False):
            # Descripción adaptativa según el tipo de red
            if tipo_red == "Red de Colaboración Autor-Autor":
                st.markdown("""
                El grafo muestra grupos de autores que colaboran más entre sí que con el resto de la red. Cada color representa una comunidad distinta, permitiendo identificar agrupamientos naturales de colaboración.
                """)
            elif tipo_red == "Red Autor-Institución":
                st.markdown("""
                El análisis identifica grupos de autores que comparten afiliaciones institucionales similares. Cada comunidad representa un cluster de investigadores con vínculos institucionales comunes.
                """)
            elif tipo_red == "Red Autor-Campo de Estudio":
                st.markdown("""
                Las comunidades agrupan autores que publican en campos de estudio relacionados. Cada cluster representa una especialización temática dentro de la red académica.
                """)
            elif tipo_red == "Red de Autores Principales-Secundarios":
                st.markdown("""
                El análisis revela grupos de autores con patrones similares de liderazgo y colaboración jerárquica en las publicaciones científicas.
                """)
            
            # Análisis de comunidades para todos los tipos de red
            if G is not None:
                import networkx.algorithms.community as nx_comm
                # Usar Greedy Modularity (Louvain-like)
                G_undirected = G.to_undirected() if G.is_directed() else G
                Gc = G_undirected.copy()
                for n in Gc.nodes():
                    Gc.nodes[n].clear()
                for u, v in Gc.edges():
                    Gc[u][v].clear()
                try:
                    communities = list(nx_comm.greedy_modularity_communities(Gc))
                except Exception:
                    communities = []
                
                # Paleta de colores
                community_colors = [
                    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                    '#ffb300', '#803e75', '#ff6800', '#a6bdd7', '#c10020', '#cea262', '#817066', '#007d34', '#f6768e', '#00538a',
                    '#ff7a5c', '#53377a', '#ff8e00', '#b32851', '#f4c800', '#7f180d', '#93aa00', '#593315', '#f13a13', '#232c16',
                    '#005c31', '#b2babb', '#d35400', '#7d3c98', '#229954', '#d5dbdb', '#f9e79f', '#1abc9c', '#2e4053', '#f7cac9',
                    '#92a8d1', '#034f84', '#f7786b', '#b565a7', '#dd4132', '#6b5b95', '#feb236', '#d64161', '#ffef96', '#50394c',
                    '#c94c4c', '#4b3832', '#ff6f69', '#88d8b0', '#b2ad7f', '#6b4226', '#fff4e6', '#c1c1c1', '#ffb347', '#ff6961',
                    '#aec6cf', '#77dd77', '#836953', '#cb99c9', '#e97451', '#fdfd96', '#c23b22', '#ffb7ce', '#b39eb5', '#ffdac1',
                    '#b0e0e6', '#ffef00', '#e0b0ff', '#b284be', '#72a0c1', '#f5e6e8', '#cfcfc4', '#bdb76b', '#483d8b', '#2e8b57',
                    '#fa8072', '#f0e68c', '#dda0dd', '#b0c4de', '#ff1493', '#00ced1', '#ff4500', '#da70d6'
                ]
                
                # Asignar color a cada comunidad
                node_community = {}
                community_color_map = {}
                G_colored = G.copy()
                for i, comm in enumerate(communities):
                    color = community_colors[i % len(community_colors)]
                    community_color_map[i] = color
                    for n in comm:
                        if n in G_colored.nodes:
                            G_colored.nodes[n]["color"] = color
                            G_colored.nodes[n]["group"] = i
                        node_community[n] = i
                
                # Visualización
                show_networkx_graph(G_colored, height=500, width=900, show_info_expander=False)
                
                # Resumen visual de comunidades
                num_com = len(communities)
                sizes = [len(c) for c in communities]
                if sizes:
                    avg_size = sum(sizes) / len(sizes)
                    max_size = max(sizes)
                    min_size = min(sizes)
                else:
                    avg_size = max_size = min_size = 0
                grandes = [i for i, s in enumerate(sizes) if s >= avg_size]
                pequenas = [i for i, s in enumerate(sizes) if s <= 3]
                
                # Métricas principales en columnas (SIN "Pequeñas (≤3)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Comunidades", num_com)
                with col2:
                    st.metric("Tamaño Promedio", f"{avg_size:.1f}")
                with col3:
                    st.metric("Más Grande", f"{max_size} autores")
                
                # Campos de estudio de las comunidades más grandes
                campo_com = []
                for i, comm in enumerate(communities):
                    campos = Counter()
                    for n in comm:
                        campos.update(autor_info.get(n, {}).get('campos', []))
                    if campos:
                        campo_com.append((i, campos.most_common(1)[0][0], campos.most_common(1)[0][1]))
                
                # Instituciones con más presencia en comunidades
                inst_count = Counter()
                for comm in communities:
                    insts = set()
                    for n in comm:
                        insts.update(autor_info.get(n, {}).get('instituciones', []))
                    for inst in insts:
                        inst_count[inst] += 1
                
                # Mostrar top campos e instituciones en columnas
                if campo_com or inst_count:
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        if campo_com:
                            st.markdown("**🎓 Top Campos por Comunidad:**")
                            top_campo = sorted(campo_com, key=lambda x: sizes[x[0]], reverse=True)[:3]
                            for idx, campo, freq in top_campo:
                                st.markdown(f"• **{campo}** ({freq} autores)")
                    
                    with col_right:
                        if inst_count:
                            st.markdown("**🏛️ Instituciones Más Presentes:**")
                            top_insts = inst_count.most_common(3)
                            for inst, freq in top_insts:
                                # Acortar nombres muy largos
                                inst_short = inst if len(inst) <= 40 else inst[:37] + "..."
                                st.markdown(f"• **{inst_short}** ({freq} comunidades)")
                
                # Tabla de comunidades
                comm_data = []
                for i, comm in enumerate(communities):
                    campos = Counter()
                    for n in comm:
                        campos.update(autor_info.get(n, {}).get('campos', []))
                    campo_princ = campos.most_common(1)[0][0] if campos else "-"
                    color = community_color_map[i]
                    comm_data.append({
                        'Comunidad': f"Comunidad {i+1}",
                        'Autores': len(comm),
                        'Campo de Estudio': campo_princ,
                        'Color': color,
                        'Miembros': ', '.join(list(comm)[:5]) + (f" (+{len(comm)-5} más)" if len(comm) > 5 else "")
                    })
                df_comm = pd.DataFrame(comm_data).sort_values('Autores', ascending=False)
                
                # Mostrar color como cuadrado visual
                def color_square_html(color):
                    return '■'
                df_comm['Color'] = df_comm['Color'].apply(color_square_html)
                st.dataframe(df_comm.style.apply(lambda col: [f'color: {comm_data[i]["Color"]}; font-size:22px;' if col.name=="Color" else '' for i in range(len(col))], axis=0), hide_index=True)
                
                # Selector de comunidad con key único
                comm_options = [f"Comunidad {i+1}: {len(comm)} autores" for i, comm in sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)]
                comm_sel = st.selectbox("Selecciona una comunidad para explorar", comm_options, key=f"selector_comunidad_autores_{tipo_red}_{hash(str(st.session_state.get('red_seleccionada', 'default'))) % 1000}")
                if comm_sel:
                    idx = int(comm_sel.split()[1].replace(":", "")) - 1
                    comm_nodes = list(communities[idx])
                    color = community_color_map[idx]
                    
                    # Construir subgrafo
                    subG = G.subgraph(comm_nodes).copy()
                    # Colorear nodos igual que en el grafo principal
                    for n in subG.nodes:
                        subG.nodes[n]["color"] = color
                    
                    # Visualización, tabla y narrativa en columnas
                    col_left, col_right = st.columns(2, gap="large")
                    with col_left:
                        st.markdown(f"**Subgrafo de la {comm_sel}:**")
                        # Sanitize node and edge tooltips for subgraph
                        for n in subG.nodes:
                            if 'title' in subG.nodes[n]:
                                subG.nodes[n]['title'] = sanitize_hover(subG.nodes[n]['title'])
                        for u, v, data in subG.edges(data=True):
                            if 'title' in data:
                                data['title'] = sanitize_hover(data['title'])
                        show_networkx_graph(subG, height=400, width=600, show_info_expander=False)
                        
                        # Preparar datos de autores
                        tabla_autores = []
                        for n in comm_nodes:
                            num_articulos = len(autor_info.get(n, {}).get('articulos', []))
                            vecinos = set(G.neighbors(n)) if n in G else set()
                            fuera = len([v for v in vecinos if v not in comm_nodes])
                            dentro = len([v for v in vecinos if v in comm_nodes])
                            tabla_autores.append({
                                'Autor': n,
                                'Artículos': num_articulos,
                                'Colaboraciones dentro': dentro,
                                'Colaboraciones fuera': fuera
                            })
                        
                        df_autores = pd.DataFrame(tabla_autores)
                        
                        # Sección de autores de la comunidad con key único
                        mostrar_autores = st.toggle("👥 Mostrar autores de la comunidad", key=f"toggle_autores_{tipo_red}_{idx}")
                        if mostrar_autores:
                            st.markdown("**Autores:**")
                            for _, row in df_autores.iterrows():
                                st.markdown(f"• **{row['Autor']}** - {row['Artículos']} artículos | {row['Colaboraciones dentro']} colab. internas | {row['Colaboraciones fuera']} colab. externas")
                        
                        # Sección específica según el tipo de red
                        if tipo_red == "Red Autor-Institución":
                            # Mostrar instituciones de la comunidad
                            instituciones_comunidad = set()
                            for n in comm_nodes:
                                instituciones_comunidad.update(autor_info.get(n, {}).get('instituciones', []))
                            
                            if instituciones_comunidad:
                                mostrar_instituciones = st.toggle(f"🏛️ Mostrar instituciones de la comunidad ({len(instituciones_comunidad)})", key=f"toggle_instituciones_{tipo_red}_{idx}")
                                if mostrar_instituciones:
                                    st.markdown("**Instituciones:**")
                                    for inst in sorted(instituciones_comunidad):
                                        # Contar autores por institución
                                        autores_inst = [n for n in comm_nodes if inst in autor_info.get(n, {}).get('instituciones', [])]
                                        st.markdown(f"• **{inst}** ({len(autores_inst)} autores)")
                        
                        elif tipo_red == "Red Autor-Campo de Estudio":
                            # Mostrar campos de estudio de la comunidad
                            campos_comunidad = set()
                            for n in comm_nodes:
                                campos_comunidad.update(autor_info.get(n, {}).get('campos', []))
                            
                            if campos_comunidad:
                                mostrar_campos = st.toggle(f"🎓 Mostrar campos de estudio de la comunidad ({len(campos_comunidad)})", key=f"toggle_campos_{tipo_red}_{idx}")
                                if mostrar_campos:
                                    st.markdown("**Campos de Estudio:**")
                                    for campo in sorted(campos_comunidad):
                                        # Contar autores por campo
                                        autores_campo = [n for n in comm_nodes if campo in autor_info.get(n, {}).get('campos', [])]
                                        st.markdown(f"• **{campo}** ({len(autores_campo)} autores)")
                        
                        # Sección de artículos de la comunidad
                        papers_comunidad = set()
                        for n in comm_nodes:
                            for art, _ in autor_info.get(n, {}).get('articulos', []):
                                id_art = art.get('Nombre de Articulo') or art.get('Archivo')
                                if id_art:
                                    papers_comunidad.add(id_art)
                        
                        if papers_comunidad:
                            mostrar_articulos = st.toggle(f"📄 Mostrar artículos de la comunidad ({len(papers_comunidad)})", key=f"toggle_articulos_{tipo_red}_{idx}")
                            if mostrar_articulos:
                                st.markdown("**Artículos:**")
                                for articulo in sorted(papers_comunidad):
                                    st.markdown(f"• {articulo}")

                    with col_right:
                        # Nube de palabras de la comunidad
                        palabras_com = []
                        for n in comm_nodes:
                            palabras_com += autor_info.get(n, {}).get('palabras', [])
                        palabras_com = [p.strip() for p in palabras_com if p.strip()]
                        if palabras_com:
                            wc = WordCloud(width=900, height=600, background_color='white', collocations=False, prefer_horizontal=1.0, max_words=50)
                            wc.generate_from_frequencies(Counter(palabras_com))
                            st.markdown("<div style='text-align:center'><b>Áreas de investigación de la comunidad:</b></div>", unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(9,6))
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)

                        # --- Resumen conciso de la comunidad (SIN números de ranking en paréntesis) ---
                        df_autores_sorted = df_autores.sort_values('Colaboraciones dentro', ascending=False)
                        top_autores = df_autores_sorted.head(2)['Autor'].tolist()
                        
                        total_papers = sum(df_autores['Artículos'])
                        promedio_papers = total_papers / len(comm_nodes) if len(comm_nodes) > 0 else 0
                        
                        # Campos e instituciones
                        instituciones = set()
                        campos_dentro = set()
                        for n in comm_nodes:
                            instituciones.update(autor_info.get(n, {}).get('instituciones', []))
                            campos_dentro.update(autor_info.get(n, {}).get('campos', []))
                        
                        campo_principal = list(campos_dentro)[0] if campos_dentro else "No definido"
                        
                        # Métricas de conectividad
                        densidad = nx.density(G.subgraph(comm_nodes))
                        clustering = nx.average_clustering(G.subgraph(comm_nodes)) if len(comm_nodes) > 2 else 0
                        
                        # Resumen en formato de puntos clave (SIN rankings en paréntesis)
                        st.markdown("**📊 Resumen de la Comunidad**")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"• **Tamaño:** {len(comm_nodes)} autores")
                            st.markdown(f"• **Productividad:** {total_papers} artículos")
                            st.markdown(f"• **Promedio:** {promedio_papers:.1f} artículos/autor")
                            
                        with col_b:
                            st.markdown(f"• **Campo principal:** {campo_principal}")
                            st.markdown(f"• **Instituciones:** {len(instituciones)}")
                            st.markdown(f"• **Densidad:** {densidad:.2f}")
                        
                        if top_autores:
                            st.markdown(f"• **Autores destacados:** {', '.join(top_autores)}")