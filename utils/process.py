import pandas as pd

df = pd.read_json('Json_normalizado.json') 

df_cuba = df[df['Pais'] == 'Cuba'].copy()

df_cuba = df_cuba.drop(columns=['Numero de Palabras'])

columnas_deseadas = [col for col in df_cuba.columns if col != 'Numero de Palabras']
df_cuba = df_cuba[columnas_deseadas]

df_cuba.to_json('articulos_cuba.json', orient='records', indent=2, force_ascii=False)