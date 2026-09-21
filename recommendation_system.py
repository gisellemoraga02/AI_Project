"""Sistema simple de recomendación de productos con KNN."""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# Datos de ejemplo: las características representan precio, valoración y popularidad.
productos = pd.DataFrame(
	{
		"nombre": [
			"Laptop Pro",
			"Laptop Básica",
			"Teléfono Inteligente",
			"Tablet Familiar",
			"Auriculares Inalámbricos",
			"Monitor 24 pulgadas",
		],
		"precio": [1200, 600, 800, 350, 90, 250],
		"valoracion": [4.8, 4.2, 4.6, 4.3, 4.5, 4.4],
		"popularidad": [90, 75, 95, 80, 88, 70],
	}
)

# Seleccionamos las variables que usará el modelo y las normalizamos para
# evitar que una escala (por ejemplo, el precio) domine a las demás.
columnas_caracteristicas = ["precio", "valoracion", "popularidad"]
X = productos[columnas_caracteristicas]
y = productos["nombre"]

escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Entrenamos un clasificador KNN usando el nombre del producto como etiqueta.
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_escalado, y)


def recomendar_productos(precio, valoracion, popularidad, cantidad=3):
	"""Devuelve los productos más parecidos a las preferencias indicadas."""
	if cantidad < 1:
		raise ValueError("La cantidad debe ser mayor que cero.")

	preferencias = pd.DataFrame(
		[[precio, valoracion, popularidad]], columns=columnas_caracteristicas
	)
	preferencias_escaladas = escalador.transform(preferencias)
	cantidad = min(cantidad, len(productos))

	# Buscamos los vecinos más cercanos y conservamos su información original.
	_, indices = modelo.kneighbors(preferencias_escaladas, n_neighbors=cantidad)
	return productos.iloc[indices[0]].reset_index(drop=True)


# Ejemplo de uso: recomendar productos para alguien que busca un artículo
# de precio medio, buena valoración y alta popularidad.
if __name__ == "__main__":
	recomendaciones = recomendar_productos(
		precio=500, valoracion=4.5, popularidad=85, cantidad=3
	)
	print("Productos recomendados:")
	print(recomendaciones[["nombre", "precio", "valoracion", "popularidad"]].to_string(index=False))
