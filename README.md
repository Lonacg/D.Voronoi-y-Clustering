# Practica 2 Geometria Comptacional - Diagrama de Voronoi y Clustering
Practica 2 GCOM - D.Voronoi y Clustering

Determina el número ideal de franjas de Villa Laminera (sistema A) a partir del número óptimo de clusters o vecindades de Voronói. Para ello, utiliza el coeficiente de Silhouette (s), que puede emplearse directamente desde la librería sklearn:

i) Obtén el coeficiente s de A para diferente número de vecindades k ∈ {2, 3, ..., 15} usando el algoritmo KMeans. Muestra en una gráfica el valor de ¯s en función de k y decide con ello cuál es el número óptimo de vecindades. En una segunda gráfica, muestra la clasificación (clusters) resulante con diferentes colores y representa el diagrama de Voronói en esa misma gráfica.

ii) Obtén el coeficiente s para el mismo sistema A usando ahora el algoritmo DBSCAN con la métrica ‘euclidean’ y luego con ‘manhattan’. En este caso, el parámetro que debemos explorar es el umbral de distancia ϵ ∈ (0.1, 0.4), fijando el número de elementos mínimo en n0 = 10. Compara gráficamente con el resultado del apartado anterior.

iii) ¿De qué franja de edad diríamos que son las personas con coordenadas a := (1/2, 0) y b := (0, -3)? Comprueba tu respuesta con la función kmeans.predict.
