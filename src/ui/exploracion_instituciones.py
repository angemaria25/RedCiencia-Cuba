import streamlit as st
import pandas as pd
import networkx as nx
from ..visualization.graphs import (
    build_institution_institution_graph,
    build_field_institution_graph,
    build_institution_author_graph
)
import matplotlib.pyplot as plt
from ..visualization.graphs_render import show_networkx_graph
from ..analysis.DataScience import (
    resumen_narrativo_institucion_institucion,
    resumen_narrativo_institucion_campo,
    resumen_narrativo_institucion_autor,
    resumen_narrativo_institucion_autor_autor,
    resumen_narrativo_institucion_institucion_dirigido
)
from collections import Counter, defaultdict

def crear_seccion_metricas_visuales_instituciones(G, tipo_red, instituciones_sin_colab=[]):
    """Crea una sección visual interactiva para mostrar las métricas de la red institucional"""
    
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
    
    # Obtener top instituciones por grado
    grados = dict(G.degree())
    top_instituciones = sorted(grados.items(), key=lambda x: x[1], reverse=True)[:3]
    
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
    Esta sección muestra cómo las instituciones se conectan y colaboran entre sí. 
    Cada número representa una característica importante de la red de colaboración institucional.
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
                Total de Instituciones
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Número total de instituciones que aparecen en las publicaciones analizadas")
    
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
        st.caption("Número de veces que dos instituciones han trabajado juntas en una publicación")
    
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
        st.caption("En promedio, cada institución colabora con este número de otras instituciones")
    
    # Sección 2: Top instituciones más activas
    if top_instituciones:
        st.subheader("Instituciones Más Activas")
        st.info("Estas son las instituciones que más colaboraciones han establecido con otras organizaciones.")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (institucion, colaboraciones) in enumerate(top_instituciones):
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
                        {institucion}
                    </div>
                    <div style="font-size: 12px; color: #888; margin-top: 5px;">
                        colaboraciones ({posiciones[i]})
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Sección 3: Estructura de la red
    st.subheader("Grupos de Colaboración")
    
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
                <strong>Grupo más grande:</strong> {tamano_mayor_fmt} instituciones
            </div>
            <div style="margin: 10px 0;">
                <strong>Cobertura del grupo principal:</strong> {(tamano_mayor/num_nodos*100):.1f}% de la red
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Información sobre instituciones sin colaboraciones
        if instituciones_sin_colab:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ffeaa7, #fab1a0);
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border-left: 4px solid #e17055;
            ">
                <h5 style="margin: 0 0 10px 0; color: #2c3e50;">Instituciones Sin Colaboraciones</h5>
                <div style="color: #2c3e50;">
                    <strong>{len(instituciones_sin_colab)}</strong> instituciones aparecen sin colaboraciones registradas
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Estas instituciones representan oportunidades para fomentar futuras colaboraciones.")
    
    with col2:
        # Análisis de influencia (solo para redes de colaboración) - SIN GRÁFICO
        if tipo_red == "Red Institución-Institución" and top_instituciones:
            st.markdown("**Instituciones Conectoras**")
            
            # Calcular centralidad de intermediación para las top instituciones
            try:
                betweenness = nx.betweenness_centrality(G)
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
                
                if top_betweenness:
                    st.markdown("**Top 5 - Instituciones que Más Conectan Grupos:**")
                    
                    for i, (institucion, valor) in enumerate(top_betweenness, 1):
                        st.markdown(f"{i}. **{institucion}** (nivel: {valor:.3f})")
                    
                    # Explicación del resultado
                    if top_betweenness[0][1] > 0.1:
                        st.success(f"**{top_betweenness[0][0]}** es una institución conectora clave en la red.")
                        
            except:
                st.warning("No se pudo calcular el análisis de conectores para esta red.")
    
    # Sección 4: Resumen ejecutivo visual
    if tipo_red == "Red Institución-Institución":
        # Formatear números para el texto del insight
        tamano_mayor_texto = f"{tamano_mayor}"
        
        insight_text = f"""Esta red de colaboración institucional conecta a **{num_nodos_fmt} instituciones** a través de **{num_aristas_fmt} colaboraciones**. La red muestra un patrón de colaboración donde cada institución colabora en promedio con **{grado_promedio:.1f} otras organizaciones**."""
        
        if top_instituciones:
            institucion_top = top_instituciones[0][0]
            colabs_top = top_instituciones[0][1]
            insight_text += f" **{institucion_top}** lidera la red con **{colabs_top} colaboraciones**, estableciéndose como una institución central en el ecosistema científico."
        
        if num_componentes > 1:
            insight_text += f" La red se organiza en **{num_componentes} grupos independientes**, donde el grupo principal incluye **{tamano_mayor_texto} instituciones** ({(tamano_mayor/num_nodos*100):.1f}% de la red)."
        else:
            insight_text += " La red está **completamente conectada**, lo que facilita el flujo de información entre todas las instituciones."
        
        if instituciones_sin_colab:
            insight_text += f" Adicionalmente, **{len(instituciones_sin_colab)} instituciones** no han establecido colaboraciones registradas, representando oportunidades de integración futura."
    
    else:
        insight_text = f"""Esta red {tipo_red.lower()} conecta a **{num_nodos_fmt} entidades** a través de **{num_aristas_fmt} relaciones**, mostrando un patrón de conectividad promedio de **{grado_promedio:.1f} conexiones por nodo**."""
    
    # Mostrar insights como texto markdown normal sin HTML
    st.markdown("#### Conclusiones Principales")
    st.markdown(insight_text)

def instituciones_tab(articulos):
    # Verificar si hay una red preseleccionada desde la nueva interfaz
    if 'red_seleccionada' in st.session_state:
        tipo_red = st.session_state.red_seleccionada
        st.subheader(f"Análisis: {tipo_red}")
        # Mostrar información sobre la red seleccionada
        if tipo_red == "Red Institución-Institución":
            pass
        elif tipo_red == "Red Institución-Campo de Estudio":
            pass
        elif tipo_red == "Red Institución-Autor":
            pass
    else:
        # Interfaz original como fallback
        st.subheader("Exploración de Instituciones")
        opciones = [
            "Red Institución-Institución",
            "Red Institución-Campo de Estudio",
            "Red Institución-Autor"
        ]
        tipo_red = st.selectbox("Selecciona el tipo de red institucional a visualizar:", opciones)

    def clean_edge_titles_plaintext(G):
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

    G = None
    # --- Obtener todas las instituciones presentes en el JSON, aunque no tengan colaboraciones ---
    instituciones_json = set()
    for art in articulos:
        inst_princ = art.get('Institucion Principal', None)
        if inst_princ:
            instituciones_json.add(inst_princ)
        for inst_sec in art.get('Instituciones Secundarias', []):
            if inst_sec:
                instituciones_json.add(inst_sec)
    if tipo_red == "Red Institución-Institución":
        st.markdown("""
        **Composición de la red:** Los **nodos** representan instituciones académicas, y las **aristas** conectan organizaciones 
        que han colaborado en publicaciones conjuntas. El **grosor y color** de las conexiones refleja la intensidad de la 
        colaboración interinstitucional. Analiza el ecosistema organizacional, identifica alianzas estratégicas, clusters 
        institucionales y organizaciones que actúan como puentes entre diferentes grupos académicos.
        """)
        G = build_institution_institution_graph(articulos)
        # Añadir explícitamente los nodos huérfanos
        for inst in instituciones_json:
            if inst not in G:
                G.add_node(inst, node_type='institution', color='#4A90E2')
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red Institución-Campo de Estudio":
        st.markdown("""
        **Composición de la red:** Red **bipartita** que conecta **instituciones** (cuadrados) con **campos de estudio** (círculos) 
        en los que publican. Las **aristas** indican actividad de investigación institucional en cada disciplina, y su **grosor** 
        refleja la productividad. Descubre el perfil temático de las organizaciones, identifica instituciones especializadas vs. 
        multidisciplinarias, y mapea las fortalezas académicas del ecosistema institucional.
        """)
        G = build_field_institution_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    elif tipo_red == "Red Institución-Autor":
        st.markdown("""
        **Composición de la red:** Red **bipartita** que vincula **instituciones** (cuadrados) con **investigadores** (círculos) 
        afiliados. Las **aristas** representan relaciones de afiliación académica, y su **grosor** indica la intensidad de la 
        colaboración. Analiza la capacidad de atracción de talento de las instituciones, identifica investigadores con múltiples 
        afiliaciones, y explora patrones de movilidad y concentración del capital humano científico.
        """)
        G = build_institution_author_graph(articulos)
        G = clean_edge_titles_plaintext(G)
    
    if G is not None:

        def sanitize_hover(text):
            import re
            text = re.sub(r'<.*?>', '', str(text))
            text = text.replace('<', '').replace('>', '')
            return text
        for u, v, data in G.edges(data=True):
            if 'title' in data:
                data['title'] = sanitize_hover(data['title'])
        for n, data in G.nodes(data=True):
            if 'title' in data:
                data['title'] = sanitize_hover(data['title'])
        show_networkx_graph(G, show_info_expander=False)
        
        # Resumen narrativo debajo del grafo
        resumen = None
        if tipo_red == "Red Institución-Institución":
            # Detectar instituciones sin colaboraciones según el JSON, no solo el grafo
            instituciones_sin_colab = [inst for inst in instituciones_json if G.degree(inst) == 0]
            resumen = resumen_narrativo_institucion_institucion(G)
            if instituciones_sin_colab:
                resumen += f"<br><b>Instituciones sin colaboraciones:</b> Hay {len(instituciones_sin_colab)} instituciones que no han colaborado con ninguna otra en la red."
            else:
                resumen += "<br><b>Instituciones sin colaboraciones:</b> Todas las instituciones han colaborado al menos una vez."
        elif tipo_red == "Red Institución-Campo de Estudio":
            resumen = resumen_narrativo_institucion_campo(G)
        elif tipo_red == "Red Institución-Autor":
            resumen = resumen_narrativo_institucion_autor(G)
        
        if resumen:
            # Para Red Institución-Institución, usar métricas visuales como en autores
            if tipo_red == "Red Institución-Institución":
                crear_seccion_metricas_visuales_instituciones(G, tipo_red, instituciones_sin_colab)
            else:
                st.subheader("📊 Resultados de la Red")
                
                # Para la red Institución-Campo de Estudio, mostrar resumen específico con tarjetas
                if tipo_red == "Red Institución-Campo de Estudio":
                    # Obtener métricas específicas
                    num_instituciones = len([n for n in G.nodes() if G.nodes[n].get('node_type') == 'institution'])
                    num_campos = len([n for n in G.nodes() if G.nodes[n].get('node_type') == 'field'])
                    num_vinculos = G.number_of_edges()
                    
                    # Calcular promedio de campos por institución
                    grados_inst = [G.degree(n) for n in G.nodes() if G.nodes[n].get('node_type') == 'institution']
                    promedio_campos = sum(grados_inst) / len(grados_inst) if grados_inst else 0
                    
                    # Encontrar institución más versátil
                    inst_mas_versatil = None
                    max_campos = 0
                    for n in G.nodes():
                        if G.nodes[n].get('node_type') == 'institution' and G.degree(n) > max_campos:
                            max_campos = G.degree(n)
                            inst_mas_versatil = n
                    
                    # Encontrar campo más concurrido
                    campo_mas_concurrido = None
                    max_instituciones = 0
                    for n in G.nodes():
                        if G.nodes[n].get('node_type') == 'field' and G.degree(n) > max_instituciones:
                            max_instituciones = G.degree(n)
                            campo_mas_concurrido = n
                    
                    # Tarjetas visuales para métricas principales
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea, #764ba2);
                            color: white;
                            padding: 20px;
                            border-radius: 12px;
                            text-align: center;
                            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
                            margin: 10px 0;
                        ">
                            <div style="font-size: 36px; font-weight: bold; margin-bottom: 8px;">
                                {num_instituciones}
                            </div>
                            <div style="font-size: 16px; opacity: 0.9;">
                                Instituciones
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #f093fb, #f5576c);
                            color: white;
                            padding: 20px;
                            border-radius: 12px;
                            text-align: center;
                            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
                            margin: 10px 0;
                        ">
                            <div style="font-size: 36px; font-weight: bold; margin-bottom: 8px;">
                                {num_campos}
                            </div>
                            <div style="font-size: 16px; opacity: 0.9;">
                                Campos de Estudio
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #4facfe, #00f2fe);
                            color: white;
                            padding: 20px;
                            border-radius: 12px;
                            text-align: center;
                            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
                            margin: 10px 0;
                        ">
                            <div style="font-size: 36px; font-weight: bold; margin-bottom: 8px;">
                                {num_vinculos}
                            </div>
                            <div style="font-size: 16px; opacity: 0.9;">
                                Vínculos Totales
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Tarjetas para destacados
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #a8edea, #fed6e3);
                            padding: 15px;
                            border-radius: 10px;
                            text-align: center;
                            margin: 10px 0;
                        ">
                            <div style="font-size: 24px; font-weight: bold; color: #2c3e50; margin-bottom: 5px;">
                                {promedio_campos:.1f}
                            </div>
                            <div style="font-size: 14px; color: #2c3e50;">
                                Campos promedio por institución
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if inst_mas_versatil:
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #ffeaa7, #fab1a0);
                                padding: 15px;
                                border-radius: 10px;
                                text-align: center;
                                margin: 10px 0;
                            ">
                                <div style="font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 5px;">
                                    🏆 Más Versátil
                                </div>
                                <div style="font-size: 12px; color: #2c3e50; line-height: 1.3;">
                                    <strong>{inst_mas_versatil}</strong><br>({max_campos} áreas)
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col3:
                        if campo_mas_concurrido:
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #d1f2eb, #a3e4d7);
                                padding: 15px;
                                border-radius: 10px;
                                text-align: center;
                                margin: 10px 0;
                            ">
                                <div style="font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 5px;">
                                    📈 Campo Popular
                                </div>
                                <div style="font-size: 12px; color: #2c3e50; line-height: 1.3;">
                                    <strong>{campo_mas_concurrido}</strong><br>({max_instituciones} instituciones)
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # Para otras redes, usar el resumen original
                    resumen = resumen.replace('desconectad', '').replace('aislad', '').replace('fragmentar', '').replace('impacto', '').replace('Impacto', '')
                    st.markdown(resumen, unsafe_allow_html=True)

        # --- Tabla resumen de instituciones ---
        import pandas as pd
        # Dashboard institucional
        institucion_info = defaultdict(lambda: {
            'articulos': set(),
            'campos': set(),
            'autores': set(),
            'primaria': 0,
            'secundaria': 0
        })
        for art in articulos:
            inst_princ = art.get('Institucion Principal', None)
            inst_secs = art.get('Instituciones Secundarias', [])
            campo = art.get('Campo de Estudio', None)
            autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
            nombre_art = art.get('Nombre de Articulo', None)
            # Institución principal
            if inst_princ:
                institucion_info[inst_princ]['primaria'] += 1
                if nombre_art:
                    institucion_info[inst_princ]['articulos'].add(nombre_art)
                if campo:
                    institucion_info[inst_princ]['campos'].add(campo)
                institucion_info[inst_princ]['autores'].update(autores)
            # Instituciones secundarias
            for inst_sec in inst_secs:
                if inst_sec:
                    institucion_info[inst_sec]['secundaria'] += 1
                    if nombre_art:
                        institucion_info[inst_sec]['articulos'].add(nombre_art)
                    if campo:
                        institucion_info[inst_sec]['campos'].add(campo)
                    institucion_info[inst_sec]['autores'].update(autores)
        datos = []
        for inst, info in institucion_info.items():
            datos.append({
                'Nombre de Institucion': inst,
                'Cantidad de Articulos': len(info['articulos']),
                'Cantidad de Campos de Estudio': len(info['campos']),
                'Cantidad de Autores': len(info['autores']),
                'Institucion Primaria': info['primaria'],
                'Institucion Secundaria': info['secundaria']
            })
        df = pd.DataFrame(datos)
        
        # Para la red Institución-Campo de Estudio, quitar las columnas de Primaria/Secundaria
        if tipo_red == "Red Institución-Campo de Estudio":
            cols = ['Nombre de Institucion', 'Cantidad de Articulos', 'Cantidad de Campos de Estudio', 'Cantidad de Autores']
            df = df[cols].sort_values(['Cantidad de Articulos', 'Cantidad de Autores'], ascending=False)
        else:
            cols = ['Nombre de Institucion', 'Cantidad de Articulos', 'Cantidad de Campos de Estudio', 'Cantidad de Autores', 'Institucion Primaria', 'Institucion Secundaria']
            df = df[cols].sort_values(['Cantidad de Articulos', 'Cantidad de Autores', 'Institucion Primaria'], ascending=False)
        
        st.dataframe(df, hide_index=True)

        # --- Búsqueda de instituciones ---
        instituciones_lista = sorted(list(institucion_info.keys()))
        institucion_sel = st.selectbox("Seleccionar institución", instituciones_lista, key=f"busqueda_institucion_{hash(str(st.session_state.get('red_seleccionada', 'default'))) % 1000}")
        if institucion_sel:
            info = institucion_info[institucion_sel]
            
            # --- Perfil resumido de la institución ---
            st.markdown(f"### {institucion_sel}")
            
            # Métricas principales
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📄 Artículos", len(info['articulos']))
                st.metric("🎓 Campos", len(info['campos']))
            with col_b:
                st.metric("👥 Autores", len(info['autores']))
                st.metric("🏛️ Como Principal", info['primaria'])
            
            # --- Preparar datos ---
            articulos_pub = sorted(info['articulos'])
            colaboraciones_inst = Counter()
            for art in articulos:
                insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                if institucion_sel in insts:
                    for inst in insts:
                        if inst and inst != institucion_sel:
                            colaboraciones_inst[inst] += 1
            colaboraciones_aut = Counter()
            for art in articulos:
                insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                if institucion_sel in insts:
                    autores = art.get('Autores Principales', []) + art.get('Autores Secundarios', [])
                    for autor in autores:
                        if autor:
                            colaboraciones_aut[autor] += 1
            
            # --- Secciones con toggles ---
            # Artículos de la institución
            if articulos_pub:
                mostrar_articulos_inst = st.toggle(f"📚 Mostrar artículos de la institución ({len(articulos_pub)})", key=f"toggle_articulos_inst_{institucion_sel}_{tipo_red}")
                if mostrar_articulos_inst:
                    st.markdown("**Artículos:**")
                    for articulo in articulos_pub:
                        st.markdown(f"• {articulo}")
            
            # Instituciones colaboradoras
            if colaboraciones_inst:
                mostrar_instituciones_colab = st.toggle(f"🏛️ Mostrar instituciones colaboradoras ({len(colaboraciones_inst)})", key=f"toggle_instituciones_colab_{institucion_sel}_{tipo_red}")
                if mostrar_instituciones_colab:
                    st.markdown("**Instituciones Colaboradoras:**")
                    for inst, count in colaboraciones_inst.most_common():
                        st.markdown(f"• **{inst}** ({count} colaboraciones)")
            
            # Autores colaboradores
            if colaboraciones_aut:
                mostrar_autores_colab = st.toggle(f"👥 Mostrar autores colaboradores ({len(colaboraciones_aut)})", key=f"toggle_autores_colab_{institucion_sel}_{tipo_red}")
                if mostrar_autores_colab:
                    st.markdown("**Autores Colaboradores:**")
                    for autor, count in colaboraciones_aut.most_common():
                        st.markdown(f"• **{autor}** ({count} colaboraciones)")
            
            # --- Subgrafo de colaboraciones institucionales ---
            if colaboraciones_inst:
                col1, col2 = st.columns([1, 1])
                with col1:
                    G_inst = nx.Graph()
                    G_inst.add_node(institucion_sel)
                    for inst, w in colaboraciones_inst.items():
                        G_inst.add_node(inst)
                        G_inst.add_edge(institucion_sel, inst, weight=w)
                    if G_inst.number_of_edges() > 0:
                        st.markdown(f"**🕸️ Red de Colaboraciones Institucionales:**")
                        show_networkx_graph(G_inst, height=350, width=350, show_info_expander=False)
                    else:
                        st.info("Sin colaboraciones institucionales para mostrar.")
            
            # --- Nube de palabras/frases clave ---
            palabras_clave = []
            for art in articulos:
                insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                if institucion_sel in insts:
                    palabras_clave += art.get('Palabras Clave', [])
            freq = Counter([p.strip() for p in palabras_clave if p.strip()])
            top_palabras = freq.most_common(50 if len(freq) > 25 else 25)
            if top_palabras:
                from wordcloud import WordCloud
                wc = WordCloud(width=800, height=300, background_color='white', collocations=False, prefer_horizontal=0.5)
                wc.generate_from_frequencies(dict(top_palabras))
                st.markdown("<div style='text-align:center'><b>🔬 Áreas de Investigación:</b></div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10,4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

        # --- Contenedor: Comunidades ---
        tipos_con_comunidades = [
            "Red Institución-Institución",
            "Red Institución-Autor"
        ]
        if tipo_red in tipos_con_comunidades:
            with st.expander("🌐 **Comunidades**", expanded=False):
                # Descripción adaptativa según el tipo de red
                if tipo_red == "Red Institución-Institución":
                    st.markdown("""
                    El grafo muestra grupos de instituciones que colaboran más entre sí que con el resto de la red. Cada color representa una comunidad distinta, permitiendo identificar agrupamientos naturales de colaboración institucional.
                    """)
                elif tipo_red == "Red Institución-Autor":
                    st.markdown("""
                    El análisis identifica grupos de instituciones que comparten patrones similares de colaboración con autores. Cada comunidad representa un cluster institucional con vínculos de investigación comunes.
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
                    
                    # Métricas principales en columnas (SIN "Pequeñas (≤3)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Comunidades", num_com)
                    with col2:
                        st.metric("Tamaño Promedio", f"{avg_size:.1f}")
                    with col3:
                        st.metric("Más Grande", f"{max_size} instituciones")
                    
                    # Campos de estudio de las comunidades más grandes
                    campo_com = []
                    for i, comm in enumerate(communities):
                        campos = Counter()
                        for n in comm:
                            campos.update(institucion_info.get(n, {}).get('campos', []))
                        if campos:
                            campo_com.append((i, campos.most_common(1)[0][0], campos.most_common(1)[0][1]))
                    
                    # Mostrar top campos en columnas
                    if campo_com:
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            if campo_com:
                                st.markdown("**🎓 Top Campos por Comunidad:**")
                                top_campo = sorted(campo_com, key=lambda x: sizes[x[0]], reverse=True)[:3]
                                for idx, campo, freq in top_campo:
                                    st.markdown(f"• **{campo}** ({freq} instituciones)")
                    
                    # Tabla de comunidades
                    comm_data = []
                    for i, comm in enumerate(communities):
                        campos = Counter()
                        for n in comm:
                            campos.update(institucion_info.get(n, {}).get('campos', []))
                        campo_princ = campos.most_common(1)[0][0] if campos else "-"
                        color = community_color_map[i]
                        comm_data.append({
                            'Comunidad': f"Comunidad {i+1}",
                            'Instituciones': len(comm),
                            'Campo de Estudio': campo_princ,
                            'Color': color,
                            'Miembros': ', '.join(list(comm)[:5]) + (f" (+{len(comm)-5} más)" if len(comm) > 5 else "")
                        })
                    df_comm = pd.DataFrame(comm_data).sort_values('Instituciones', ascending=False)
                    
                    # Mostrar color como cuadrado visual
                    def color_square_html(color):
                        return '■'
                    df_comm['Color'] = df_comm['Color'].apply(color_square_html)
                    st.dataframe(df_comm.style.apply(lambda col: [f'color: {comm_data[i]["Color"]}; font-size:22px;' if col.name=="Color" else '' for i in range(len(col))], axis=0), hide_index=True)
                    
                    # Selector de comunidad con key único
                    comm_options = [f"Comunidad {i+1}: {len(comm)} instituciones" for i, comm in sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)]
                    comm_sel = st.selectbox("Selecciona una comunidad para explorar", comm_options, key=f"selector_comunidad_instituciones_{tipo_red}_{hash(str(st.session_state.get('red_seleccionada', 'default'))) % 1000}")
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
                            
                            # Preparar datos de instituciones
                            tabla_instituciones = []
                            for n in comm_nodes:
                                num_articulos = len(institucion_info.get(n, {}).get('articulos', []))
                                vecinos = set(G.neighbors(n)) if n in G else set()
                                fuera = len([v for v in vecinos if v not in comm_nodes])
                                dentro = len([v for v in vecinos if v in comm_nodes])
                                tabla_instituciones.append({
                                    'Institución': n,
                                    'Artículos': num_articulos,
                                    'Colaboraciones dentro': dentro,
                                    'Colaboraciones fuera': fuera
                                })
                            
                            df_instituciones = pd.DataFrame(tabla_instituciones)
                            
                            # Sección de instituciones de la comunidad con key único
                            mostrar_instituciones = st.toggle("🏛️ Mostrar instituciones de la comunidad", key=f"toggle_instituciones_{tipo_red}_{idx}")
                            if mostrar_instituciones:
                                st.markdown("**Instituciones:**")
                                for _, row in df_instituciones.iterrows():
                                    st.markdown(f"• **{row['Institución']}** - {row['Artículos']} artículos | {row['Colaboraciones dentro']} colab. internas | {row['Colaboraciones fuera']} colab. externas")
                            
                            # Sección de artículos de la comunidad
                            papers_comunidad = set()
                            for n in comm_nodes:
                                for art in articulos:
                                    insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                                    id_art = art.get('Nombre de Articulo') or art.get('Archivo')
                                    if id_art and n in insts:
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
                                for art in articulos:
                                    insts = [art.get('Institucion Principal', None)] + art.get('Instituciones Secundarias', [])
                                    if n in insts:
                                        palabras_com += art.get('Palabras Clave', [])
                            palabras_com = [p.strip() for p in palabras_com if p.strip()]
                            if palabras_com:
                                from wordcloud import WordCloud
                                wc = WordCloud(width=900, height=600, background_color='white', collocations=False, prefer_horizontal=1.0, max_words=50)
                                wc.generate_from_frequencies(Counter(palabras_com))
                                st.markdown("<div style='text-align:center'><b>Áreas de investigación de la comunidad:</b></div>", unsafe_allow_html=True)
                                fig, ax = plt.subplots(figsize=(9,6))
                                ax.imshow(wc, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)

                            # --- Resumen conciso de la comunidad (SIN números de ranking en paréntesis) ---
                            df_instituciones_sorted = df_instituciones.sort_values('Colaboraciones dentro', ascending=False)
                            top_instituciones = df_instituciones_sorted.head(2)['Institución'].tolist()
                            
                            total_papers = sum(df_instituciones['Artículos'])
                            promedio_papers = total_papers / len(comm_nodes) if len(comm_nodes) > 0 else 0
                            
                            # Campos e instituciones
                            campos_dentro = set()
                            for n in comm_nodes:
                                campos_dentro.update(institucion_info.get(n, {}).get('campos', []))
                            
                            campo_principal = list(campos_dentro)[0] if campos_dentro else "No definido"
                            
                            # Métricas de conectividad
                            densidad = nx.density(G.subgraph(comm_nodes))
                            clustering = nx.average_clustering(G.subgraph(comm_nodes)) if len(comm_nodes) > 2 else 0
                            
                            # Resumen en formato de puntos clave (SIN rankings en paréntesis)
                            st.markdown("**📊 Resumen de la Comunidad**")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"• **Tamaño:** {len(comm_nodes)} instituciones")
                                st.markdown(f"• **Productividad:** {total_papers} artículos")
                                st.markdown(f"• **Promedio:** {promedio_papers:.1f} artículos/institución")
                                
                            with col_b:
                                st.markdown(f"• **Campo principal:** {campo_principal}")
                                st.markdown(f"• **Campos:** {len(campos_dentro)}")
                                st.markdown(f"• **Densidad:** {densidad:.2f}")
                            
                            if top_instituciones:
                                st.markdown(f"• **Instituciones destacadas:** {', '.join(top_instituciones)}")