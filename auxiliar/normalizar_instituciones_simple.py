import csv
import re
from collections import Counter

def normalizar_instituciones():
    """
    Normaliza los nombres de instituciones en el archivo data_final.csv
    """
    print("Cargando archivo data_final.csv...")
    
    # Leer el archivo CSV
    filas = []
    with open('data_final.csv', 'r', encoding='utf-8') as archivo:
        lector = csv.DictReader(archivo)
        headers = lector.fieldnames
        for fila in lector:
            filas.append(fila)
    
    print(f"Total de filas: {len(filas)}")
    
    # Función para normalizar una institución individual
    def normalizar_institucion(institucion):
        if not institucion or institucion.strip() == '':
            return institucion
        
        # Convertir a string y limpiar espacios
        inst = str(institucion).strip()
        
        # Diccionario de normalizaciones específicas
        normalizaciones = {
            # Universidad De La Habana
            'Universidad De La Habana Cuba': 'Universidad De La Habana',
            'Universidad De La Habana La Habana Cuba': 'Universidad De La Habana',
            
            # Facultad De Matemática Y Computación
            'Facultad De Matematica Y Computacion Universidad De La Habana Cuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Facultad De Matematica Y Computacion Universidad De La Habana La Habana Cuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Facultad De Matematica Y Computacion Universidad De La Habana': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Universidad De La Habana Facultad Matematica Y Computacion Cuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Facultad De Matematica Y Computacion': 'Facultad De Matemática Y Computación Universidad De La Habana',
            
            # Departamento de Matemática
            'Departamento De Matematica Universidad De La Habana Cuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Universidad De La Habana Facultad De Matematica Y Computacion': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Departamentodematematicauniversidaddelahabanacuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Departamento De Matematica Universidad De La Habana La Habana Cuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Departamentodematematicauniversidaddelahabanalahabanacuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            
            # Departamento de Matemática Aplicada
            'Dpto Matematica Aplicada Facultad De Matematica Y Computacion Universidad De La Habana Cuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Departamento De Matematica Aplicada Facultad De Matematica Y Computacion Universidad De La Habana La Habana Cuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Departamento De Matematica Aplicada Universidad De La Habana Cuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Departamentodematematicaaplicadauniversidaddelahabanalahabanacuba': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Departamento De Matematica Aplicada Facultad De Matematica Y Computacion Universidad De La Habana': 'Facultad De Matemática Y Computación Universidad De La Habana',
            'Departamento De Matematica Aplicada Universidad De La Habana Facultad De Matematica Y Computacion': 'Facultad De Matemática Y Computación Universidad De La Habana',
            
            # Departamento de Licenciatura en Matemática Universidad de Holguín
            'Departamentodelicenciaturaenmatematicauniversidaddeholguınholguıncuba': 'Departamento De Licenciatura En Matemática Universidad De Holguín',
        }
        
        # Aplicar normalizaciones exactas primero
        if inst in normalizaciones:
            return normalizaciones[inst]
        
        # Normalizaciones con patrones más flexibles
        inst_lower = inst.lower()
        
        # Patrones para Universidad De La Habana (solo si no contiene facultad o departamento)
        if (re.search(r'universidad\s+de\s+la\s+habana.*cuba', inst_lower) and 
            'facultad' not in inst_lower and 'departamento' not in inst_lower and 
            'instituto' not in inst_lower):
            return 'Universidad De La Habana'
        
        # Patrones para Facultad De Matemática Y Computación
        if (re.search(r'facultad.*matematica.*computacion.*universidad.*habana', inst_lower) or
            re.search(r'matematica.*computacion.*universidad.*habana', inst_lower) or
            re.search(r'departamento.*matematica.*universidad.*habana', inst_lower) or
            re.search(r'dpto.*matematica.*universidad.*habana', inst_lower)):
            return 'Facultad De Matemática Y Computación Universidad De La Habana'
        
        return inst
    
    # Aplicar normalización a la columna de afiliaciones
    print("Normalizando afiliaciones...")
    
    def normalizar_afiliaciones_fila(afiliaciones_str):
        if not afiliaciones_str or afiliaciones_str.strip() == '':
            return afiliaciones_str
        
        # Dividir por comas y normalizar cada institución
        instituciones = [inst.strip() for inst in str(afiliaciones_str).split(',')]
        instituciones_normalizadas = [normalizar_institucion(inst) for inst in instituciones]
        
        return ', '.join(instituciones_normalizadas)
    
    # Crear una copia de las filas originales para comparar
    filas_originales = []
    for fila in filas:
        filas_originales.append(fila.copy())
    
    # Aplicar normalización
    for fila in filas:
        fila['afiliaciones_normalizadas'] = normalizar_afiliaciones_fila(fila['afiliaciones_normalizadas'])
    
    # Mostrar algunos ejemplos de cambios
    print("\n=== EJEMPLOS DE NORMALIZACIONES REALIZADAS ===")
    cambios_encontrados = 0
    
    for i, (fila_original, fila_nueva) in enumerate(zip(filas_originales, filas)):
        if fila_original['afiliaciones_normalizadas'] != fila_nueva['afiliaciones_normalizadas']:
            if cambios_encontrados < 10:  # Mostrar solo los primeros 10 cambios
                print(f"\nFila {i+1}:")
                print(f"ANTES: {fila_original['afiliaciones_normalizadas']}")
                print(f"DESPUÉS: {fila_nueva['afiliaciones_normalizadas']}")
            cambios_encontrados += 1
    
    print(f"\nTotal de filas con cambios: {cambios_encontrados}")
    
    # Guardar el archivo normalizado
    print("\nGuardando archivo normalizado...")
    
    # Guardar como nuevo archivo
    with open('data_final_normalizado.csv', 'w', encoding='utf-8', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=headers)
        escritor.writeheader()
        escritor.writerows(filas)
    
    # También sobrescribir el archivo original
    with open('data_final.csv', 'w', encoding='utf-8', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=headers)
        escritor.writeheader()
        escritor.writerows(filas)
    
    print("✅ Normalización completada!")
    print("📁 Archivo guardado como: data_final_normalizado.csv")
    print("📁 Archivo original actualizado: data_final.csv")
    
    # Mostrar estadísticas de instituciones más frecuentes después de la normalización
    print("\n=== INSTITUCIONES MÁS FRECUENTES DESPUÉS DE LA NORMALIZACIÓN ===")
    todas_instituciones = []
    for fila in filas:
        afiliaciones = fila['afiliaciones_normalizadas']
        if afiliaciones and afiliaciones.strip():
            instituciones = [inst.strip() for inst in str(afiliaciones).split(',')]
            todas_instituciones.extend([inst for inst in instituciones if inst])
    
    contador_inst = Counter(todas_instituciones)
    
    print("Top 15 instituciones más frecuentes:")
    for inst, count in contador_inst.most_common(15):
        if inst:  # Evitar strings vacíos
            print(f"{count:3d}: {inst}")

if __name__ == "__main__":
    normalizar_instituciones()