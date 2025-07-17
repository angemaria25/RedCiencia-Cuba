import json

input_json = "Json_normalizado.json"  
output_json = "articulos_cuba.json"    

CAMPOS_A_ELIMINAR = [
    "Numero de Palabras",
    "Referencias Bibliograficas",
    "Autores de Articulos Referenciados",
    "Instituciones de Articulos Referenciados"
]

def procesar_json(data):
    articulos_limpios = []
    for articulo in data:
        if articulo.get("Pais") == "Cuba":
            articulo_limpio = {
                key: value for key, value in articulo.items() 
                if key not in CAMPOS_A_ELIMINAR
            }
            articulos_limpios.append(articulo_limpio)
    return articulos_limpios

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)  
articulos_finales = procesar_json(data)

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(articulos_finales, f, indent=2, ensure_ascii=False)

print(f"✅ JSON filtrado y limpiado guardado en: {output_json}")