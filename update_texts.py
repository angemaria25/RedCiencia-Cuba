"""
Script para actualizar los textos descriptivos de los grafos
"""

import re

def update_exploracion_autores():
    """Actualiza los textos en exploracion_autores.py"""
    file_path = 'src/ui/exploracion_autores.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Texto 1: Red de Colaboración Autor-Autor
    old_text1 = '''        st.markdown("""
        <b>En esta red</b>, cada nodo representa un <b>autor</b> y dos autores están conectados si han colaborado juntos en al menos un artículo.<br>
        El <b>color de la arista</b> indica la intensidad de la colaboración: <b>más rojo</b> significa que han trabajado juntos muchas veces, <b>más azul</b> significa pocas colaboraciones.<br>
        Los <b>nodos más rojizos</b> son los autores con más conexiones (mayor grado).<br>
        <b>Al pasar el mouse</b> sobre un nodo, verás el <b>nombre del autor</b> y la <b>cantidad de colaboradores distintos</b> que tiene.<br>
        <i>Ejemplo:</i> si ves una arista entre Ana y Luis con peso <b>3</b> y color rojizo, significa que han coescrito <b>3 artículos</b> y es una de las colaboraciones más fuertes de la red. Si pasas el mouse sobre Ana y ves <b>"Ana (12 conexiones)"</b>, significa que Ana ha colaborado con <b>12 autores diferentes</b>.
        """, unsafe_allow_html=True)'''
    
    new_text1 = '''        st.markdown("""
        **Visualización de colaboraciones científicas:** Cada investigador aparece como un nodo conectado a sus colaboradores. 
        Las conexiones más intensas (más artículos compartidos) se muestran en tonos rojizos, mientras que las colaboraciones 
        ocasionales aparecen en azul. Pasa el cursor sobre cualquier nodo para ver detalles del autor.
        """)'''
    
    # Texto 2: Red de Autores Principales-Secundarios
    old_text2 = '''        st.markdown("""
        <b>En esta red</b>, cada nodo es un <b>autor</b> y una flecha va de un <b>autor principal</b> a un <b>autor secundario</b> cuando han participado juntos en un artículo.<br>
        El <b>color de la arista</b> depende de la suma de colaboraciones en ambos sentidos: <b>más rojo</b> indica mayor interacción total, <b>más azul</b> indica menos.<br>
        Los <b>nodos más rojizos</b> son los que más veces han sido secundarios (mayor in-degree), y los más anaranjados los que más veces han sido principales (mayor out-degree).<br>
        <b>Al pasar el mouse</b> sobre un nodo, verás el <b>nombre del autor</b>, la <b>cantidad de veces que fue principal</b>, la <b>cantidad de veces que fue secundario</b> y la <b>cantidad de personas distintas</b> en cada caso.<br>
        <i>Ejemplo:</i> si ves una flecha de Pedro → Laura con peso <b>4</b> y color rojizo, significa que Pedro fue autor principal y Laura secundaria en <b>4 artículos</b> y la relación total entre ambos es fuerte. Si pasas el mouse sobre Laura y ves <b>"Laura (Principal: 2 a 2 personas, Secundario: 8 por 5 personas)"</b>, significa que Laura fue principal en <b>2 artículos</b> para <b>2 personas</b> y secundaria en <b>8 artículos</b> para <b>5 personas</b> distintas.
        """, unsafe_allow_html=True)'''
    
    new_text2 = '''        st.markdown("""
        **Análisis de jerarquías académicas:** Las flechas muestran la dirección de liderazgo en las publicaciones, 
        desde autores principales hacia secundarios. Los colores indican la frecuencia de estas relaciones jerárquicas. 
        Explora cómo se distribuyen los roles de liderazgo en la comunidad científica.
        """)'''
    
    # Texto 3: Red Autor-Campo de Estudio
    old_text3 = '''        st.markdown("""
        <b>En esta red</b>, los nodos representan <b>autores</b> y <b>campos de estudio</b>.<br>
        Un autor está conectado a un campo si ha publicado en ese ámbito.<br>
        El <b>color de la arista</b> indica la intensidad de la relación (<b>más publicaciones, más rojo</b>).<br>
        Los <b>nodos de autores más rojizos</b> son los que han trabajado en más campos distintos.<br>
        <b>Al pasar el mouse</b> sobre un nodo autor, verás su <b>nombre</b> y la <b>cantidad de campos</b> en los que ha publicado.<br>
        <i>Ejemplo:</i> si ves a Ana conectada a tres campos, significa que su producción es diversa. Si un campo está conectado a muchos autores, es un área de investigación central.
        """, unsafe_allow_html=True)'''
    
    new_text3 = '''        st.markdown("""
        **Mapa de especialización temática:** Conecta investigadores con sus áreas de trabajo. Los autores más versátiles 
        aparecen vinculados a múltiples disciplinas, mientras que los especialistas se concentran en campos específicos. 
        Identifica áreas emergentes y patrones de interdisciplinariedad.
        """)'''
    
    # Texto 4: Red Autor-Institución
    old_text4 = '''        st.markdown("""
        <b>En esta red</b>, los nodos representan <b>autores</b> e <b>instituciones</b>.<br>
        Un autor está conectado a una institución si ha publicado afiliado a ella.<br>
        El <b>color de la arista</b> indica la intensidad de la relación (<b>más publicaciones, más rojo</b>).<br>
        Los <b>autores más rojizos</b> han colaborado con más instituciones.<br>
        <b>Al pasar el mouse</b> sobre un nodo autor, verás su <b>nombre</b> y la <b>cantidad de instituciones</b> con las que ha trabajado.<br>
        <i>Ejemplo:</i> si ves a Juan conectado a cinco instituciones, es un autor con amplia colaboración institucional. Si una institución tiene muchos autores conectados, es un centro de investigación relevante.
        """, unsafe_allow_html=True)'''
    
    new_text4 = '''        st.markdown("""
        **Red de afiliaciones institucionales:** Muestra las conexiones entre investigadores y las organizaciones donde publican. 
        Descubre centros de investigación influyentes, autores con alta movilidad institucional y patrones de colaboración 
        entre diferentes entidades académicas.
        """)'''
    
    # Aplicar reemplazos
    content = content.replace(old_text1, new_text1)
    content = content.replace(old_text2, new_text2)
    content = content.replace(old_text3, new_text3)
    content = content.replace(old_text4, new_text4)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Actualizado exploracion_autores.py")

def update_exploracion_instituciones():
    """Actualiza los textos en exploracion_instituciones.py"""
    file_path = 'src/ui/exploracion_instituciones.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Texto 1: Red Institución-Institución
    old_text1 = '''        st.markdown("""
        <b>En esta red</b>, cada nodo es una <b>institución</b> y dos instituciones están conectadas si han colaborado en al menos un artículo.<br>
        El <b>color de la arista</b> indica la intensidad de la colaboración: <b>más rojo</b> significa más colaboraciones, <b>más azul</b> menos.<br>
        Los <b>nodos más rojizos</b> son las instituciones con más conexiones.<br>
        <b>Al pasar el mouse</b> sobre un nodo, verás el <b>nombre de la institución</b> y la <b>cantidad de colaboraciones</b> que tiene.<br>
        """, unsafe_allow_html=True)'''
    
    new_text1 = '''        st.markdown("""
        **Ecosistema de colaboración institucional:** Visualiza las alianzas estratégicas entre organizaciones académicas. 
        Las conexiones más fuertes indican partnerships frecuentes, mientras que las débiles muestran colaboraciones ocasionales. 
        Identifica clusters institucionales y centros de articulación en el ecosistema científico.
        """)'''
    
    # Texto 2: Red Institución-Campo de Estudio
    old_text2 = '''        st.markdown("""
        <b>En esta red</b>, los nodos representan <b>instituciones</b> y <b>campos de estudio</b>.<br>
        Una institución está conectada a un campo si ha publicado en ese ámbito.<br>
        El <b>color de la arista</b> indica la intensidad de la relación.<br>
        Los <b>nodos de instituciones más rojizos</b> son los que han trabajado en más campos distintos.<br>
        <b>Al pasar el mouse</b> sobre un nodo institución, verás su <b>nombre</b> y la <b>cantidad de campos</b> en los que ha publicado.<br>
        """, unsafe_allow_html=True)'''
    
    new_text2 = '''        st.markdown("""
        **Perfil temático institucional:** Mapea las fortalezas disciplinarias de cada organización. Las instituciones 
        multidisciplinarias aparecen conectadas a diversos campos, mientras que las especializadas se concentran en áreas específicas. 
        Explora la diversidad académica del panorama institucional.
        """)'''
    
    # Texto 3: Red Institución-Autor
    old_text3 = '''        st.markdown("""
        <b>En esta red</b>, los nodos representan <b>instituciones</b> y <b>autores</b>.<br>
        Una institución está conectada a un autor si han colaborado en algún artículo.<br>
        El <b>color de la arista</b> indica la intensidad de la relación.<br>
        Los <b>nodos de instituciones más rojizos</b> han colaborado con más autores.<br>
        <b>Al pasar el mouse</b> sobre un nodo institución, verás su <b>nombre</b> y la <b>cantidad de autores</b> con los que ha colaborado.<br>
        """, unsafe_allow_html=True)'''
    
    new_text3 = '''        st.markdown("""
        **Talento y afiliaciones académicas:** Conecta instituciones con sus investigadores activos. Identifica organizaciones 
        con mayor capacidad de atracción de talento, autores con múltiples afiliaciones y patrones de movilidad académica 
        en el ecosistema científico.
        """)'''
    
    # Aplicar reemplazos
    content = content.replace(old_text1, new_text1)
    content = content.replace(old_text2, new_text2)
    content = content.replace(old_text3, new_text3)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Actualizado exploracion_instituciones.py")

if __name__ == "__main__":
    print("🔄 Actualizando textos descriptivos...")
    update_exploracion_autores()
    update_exploracion_instituciones()
    print("✨ Actualización de textos completada!")