"""
Práctica 2. DIAGRAMA DE VORONÓI Y CLUSTERING

Alumna: Laura Cano Gómez
Subgrupo: U2 
"""


import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN


archivo1 = r"C:\Users\Usuario\Documents\0-MIS DOCUMENTOS\0-Universidad\5-2023-2024\Segundo_cuatri\Gcom\Practicas\Practica_2-D.Voronoi-y-Clustering\Personas_de_villa_laminera.txt"
archivo2 = r"C:\Users\Usuario\Documents\0-MIS DOCUMENTOS\0-Universidad\5-2023-2024\Segundo_cuatri\Gcom\Practicas\Practica_2-D.Voronoi-y-Clustering\Franjas_de_edad.txt"

X = np.loadtxt(archivo1,skiprows=1)
Y = np.loadtxt(archivo2,skiprows=1)



'''
Determina el número ideal de franjas de Villa Laminera (sistema A) a partir del número óptimo 
de clusters o vecindades de Voronói. Para ello, utiliza el coeficiente de Silhouette (s), que puede
emplearse directamente desde la librería sklearn:
'''


'''
APARTADO i) 
- Obtén el coeficiente s de A para diferente número de vecindades k ∈ {2, 3, ..., 15} usando el
algoritmo KMeans. 
- Muestra en una gráfica el valor de s en función de k y decide con ello cuál
es el número óptimo de vecindades. 
- En una segunda gráfica, muestra la clasificación (clusters) resulante con diferentes 
colores y representa el diagrama de Voronoi en esa misma gráfica.
'''


def silhouette_coeff(X):
    """
    Determina el coeficiente de Sihouette para k = 2, ..., 15 vecindades 
    
    Returns a list object (silhouette coefficients for k = 2, ..., 15)

    Arguments:
        X -> numpy.ndarray object

    """
    coef_s = []
    for k in range(2,16):
        kmeans = KMeans(n_clusters= k, random_state= 0).fit(X)  # Inicializamos k aleatoriamente
        labels = kmeans.labels_
        silhouette = metrics.silhouette_score(X, labels)        # Obtiene el coef. de Silhouette usando KMeans
        coef_s.append(silhouette)                               # Añade el coef. a la lista

    return coef_s


def show_optimal_Voronoi_cells(coef_s):
    """
    Grafica los coeficientes de Silhouette en funcion del numero (k = 2, ..., 15) vecindades y devuelve el numero optimo de vecindades de Voronoi.
    
    Returns a int object (optimal number of Voronoi cells)

    Arguments:
        coef_s -> list object

    """
    plt.xlabel("Number of Voronoi cell")      
    plt.ylabel("Silhouette value")
    plt.plot(list(range(2,16)), coef_s)      # Representamos graficamente los coeficientes obtenidos para cada k
    plt.show()

    index = coef_s.index(max(coef_s))        # Posicion del coeficiente de Silhouette mayor en la lista

    return index + 2                         # Empezamos en k = 2 asi que ajustamos el indice


def clusters_and_VoronoiD(k):
    kmeans = KMeans(n_clusters= k, random_state= 0).fit(X)
    labels = kmeans.labels_

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(8,4))

    vor=Voronoi(kmeans.cluster_centers_)
    voronoi_plot_2d(vor)

    for i, col in zip(unique_labels, colors):
        if i == -1:
            col = [0, 0, 0, 1]          # Black used for noise.

        class_member_mask = (labels == i)

        xy = X[class_member_mask]
        plt.plot(xy[ : , 0], xy[ : , 1], 'o', markerfacecolor= tuple(col), markeredgecolor= 'k', markersize= 5)


    plt.title('Fixed number of KMeans clusters: %d' % k)
    plt.show()


## PENDIENTE AJUSTAR MARGENES DE LA IMAGEN PARA QUE SE VEA ENETRA

'''
ii) Obtén el coeficiente s para el mismo sistema A usando ahora el algoritmo DBSCAN con la
métrica ‘euclidean’ y luego con ‘manhattan’. En este caso, el parámetro que debemos explorar
es el umbral de distancia ϵ ∈ (0.1, 0.4), fijando el número de elementos mínimo en n0 = 10.
Comparad gráficamente con el resultado del apartado anterior.
'''

'''
iii) ¿De qué franja de edad diríamos que son las personas con coordenadas a := (1/2, 0) y b :=
(0, -3)? Comprueba tu respuesta con la función kmeans.predict.
'''


def main():
    archivo1 = r"C:\Users\Usuario\Documents\0-MIS DOCUMENTOS\0-Universidad\5-2023-2024\Segundo_cuatri\Gcom\Practicas\Practica_2-D.Voronoi-y-Clustering\Personas_de_villa_laminera.txt"
    archivo2 = r"C:\Users\Usuario\Documents\0-MIS DOCUMENTOS\0-Universidad\5-2023-2024\Segundo_cuatri\Gcom\Practicas\Practica_2-D.Voronoi-y-Clustering\Franjas_de_edad.txt"

    X = np.loadtxt(archivo1,skiprows=1)
    Y = np.loadtxt(archivo2,skiprows=1)

    # Apartado I
    print("APARTADO I")
    coef_s = silhouette_coeff(X)
    better_k = show_optimal_Voronoi_cells(coef_s)

    print(f"El numero optimo de vecindades de Voronoi es aquel en el que el coeficiente de Silhouette es mayor, por tanto, segun la gráfica obtenida, debemos tomar {better_k} vecindades.\n")

    clusters_and_VoronoiD(better_k)




if __name__ == '__main__':
    main()


