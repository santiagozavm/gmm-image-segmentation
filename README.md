# GMM Image Segmentation

Implementación desde cero en Python de un algoritmo de **segmentación de imágenes utilizando Gaussian Mixture Models (GMM)** y el algoritmo **Expectation-Maximization (EM)**.

El proyecto utiliza los valores RGB de cada píxel como características y agrupa los píxeles en diferentes clusters de acuerdo con sus distribuciones de color.

## Descripción

La segmentación de imágenes consiste en dividir una imagen en diferentes regiones o grupos de píxeles con características similares.

En este proyecto, cada píxel se representa mediante un vector de tres dimensiones:

[
x = [R, G, B]
]

Posteriormente, se utiliza un **Gaussian Mixture Model** para modelar la distribución de los colores presentes en la imagen.

La implementación del algoritmo GMM-EM se realizó manualmente, sin utilizar funciones de clustering de librerías como `scikit-learn`.

## Funcionamiento

El algoritmo utiliza el método **Expectation-Maximization (EM)** para estimar los parámetros de las distribuciones gaussianas.

El proceso general es:

1. Cargar la imagen.
2. Convertir los píxeles a vectores RGB.
3. Inicializar las medias, covarianzas y probabilidades de mezcla.
4. Ejecutar el **E-Step** para calcular las responsabilidades.
5. Ejecutar el **M-Step** para actualizar los parámetros del modelo.
6. Calcular la log-verosimilitud.
7. Comprobar la convergencia.
8. Asignar cada píxel al cluster con mayor responsabilidad.
9. Reconstruir la imagen utilizando la media de cada cluster.

### Expectation-Maximization

#### E-Step

Para cada píxel se calcula la probabilidad de pertenecer a cada cluster:

[
\gamma_{ik} =
\frac{
\pi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)
}{
\sum_j \pi_j \mathcal{N}(x_i|\mu_j,\Sigma_j)
}
]

donde:

* (\gamma_{ik}) es la responsabilidad del cluster (k) sobre el píxel (i).
* (\pi_k) es la probabilidad de mezcla del cluster.
* (\mu_k) es la media del cluster.
* (\Sigma_k) es la matriz de covarianza.
* (\mathcal{N}) representa la distribución gaussiana multivariada.

#### M-Step

Se actualizan los parámetros del modelo utilizando las responsabilidades calculadas durante el E-Step.

Se actualizan:

* Medias (\mu)
* Matrices de covarianza (\Sigma)
* Probabilidades de mezcla (\pi)

El proceso se repite hasta alcanzar el criterio de convergencia o llegar al número máximo de iteraciones.

## Convergencia

Para determinar cuándo detener el algoritmo se utiliza la **log-verosimilitud**.

La log-verosimilitud permite observar cómo mejora el modelo durante las iteraciones. Conforme el algoritmo se acerca a una solución estable, los cambios entre iteraciones disminuyen.

La implementación también utiliza una tolerancia relativa para determinar la convergencia.

## Estructura del proyecto

```text
gmm-image-segmentation/
│
├── README.md
├── .gitignore
├── requirements.txt
├── main.py
│
├── src/
│   └── gmm.py
│
├── images/
│   └── .gitkeep
│
└── results/
    └── .gitkeep
```

Las imágenes de entrada y los resultados generados se encuentran excluidos del repositorio mediante `.gitignore`.

## Tecnologías utilizadas

* Python
* NumPy
* OpenCV
* SciPy
* Matplotlib

## Instalación

Clona el repositorio:

```bash
git clone https://github.com/santiagozavm/gmm-image-segmentation.git
cd gmm-image-segmentation
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Coloca la imagen que deseas segmentar dentro de la carpeta:

```text
images/
```

Por ejemplo:

```text
images/tu-imagen.png
```

Después, en `main.py`, modifica la ruta de la imagen si es necesario:

```python
IMAGE_PATH = "images/tu-imagen.png"
```

También puedes modificar el número de clusters:

```python
N_CLUSTERS = 5
```

y los parámetros del algoritmo:

```python
MAX_ITER = 100
TOLERANCE = 1e-5
RANDOM_SEED = 42
```

Finalmente, ejecuta:

```bash
python main.py
```

El programa mostrará:

* La imagen original.
* La imagen segmentada.
* La evolución de la log-verosimilitud durante las iteraciones.

Además, los resultados generados se guardan en:

```text
results/
```

## Implementación

Una de las principales características de este proyecto es que el algoritmo GMM-EM fue implementado desde cero.

No se utiliza una función como:

```python
GaussianMixture(...)
```

de `scikit-learn` para realizar la segmentación.

La implementación contiene directamente las principales etapas del algoritmo:

```text
GaussianMixture
│
├── Inicialización
├── E-Step
├── M-Step
├── Log-Likelihood
├── Criterio de convergencia
├── Predicción de clusters
└── Segmentación
```

Esto permite comprender y demostrar cómo funciona internamente un modelo de mezcla gaussiana aplicado a la segmentación de imágenes.

## Objetivo

El objetivo principal del proyecto es demostrar la implementación de un algoritmo de **segmentación de imágenes mediante aprendizaje no supervisado**, utilizando Python y operaciones matemáticas sobre los datos de los píxeles, en lugar de depender directamente de una implementación de clustering ya existente.

## Autor

**Santiago Zavala Maldonado**
