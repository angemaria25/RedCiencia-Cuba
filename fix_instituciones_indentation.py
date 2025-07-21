"""
Script para corregir el error de indentación en exploracion_instituciones.py
"""

def fix_instituciones_indentation():
    """Corrige el error de indentación en exploracion_instituciones.py"""
    file_path = 'src/ui/exploracion_instituciones.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar y corregir las líneas problemáticas
    old_section = '''        if tipo_red == "Red Institución-Institución":
            
        elif tipo_red == "Red Institución-Campo de Estudio":
            
        elif tipo_red == "Red Institución-Autor":
            '''
    
    new_section = '''        if tipo_red == "Red Institución-Institución":
            pass
        elif tipo_red == "Red Institución-Campo de Estudio":
            pass
        elif tipo_red == "Red Institución-Autor":
            pass'''
    
    content = content.replace(old_section, new_section)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Corregido error de indentación en exploracion_instituciones.py")

if __name__ == "__main__":
    print("🔧 Corrigiendo error de indentación en instituciones...")
    fix_instituciones_indentation()
    print("✨ Error corregido!")