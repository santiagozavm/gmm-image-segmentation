import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.gmm import GaussianMixture



# CONFIGURACIÓN


IMAGE_PATH = "images/tu-imagen.png"

N_CLUSTERS = 3
MAX_ITER = 100
TOLERANCE = 1e-5
RANDOM_SEED = 42



# CARGAR IMAGEN


img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(
        f"No se pudo cargar la imagen: {IMAGE_PATH}"
    )


# OpenCV utiliza BGR.
# Convertimos a RGB para trabajar con el formato convencional.
img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

rows, cols, _ = img_rgb.shape



# PREPARAR DATOS


datos = img_rgb.reshape(
    -1,
    3
).astype(np.float64)



# CREAR Y ENTRENAR GMM


gmm = GaussianMixture(
    n_clusters=N_CLUSTERS,
    max_iter=MAX_ITER,
    tolerance=TOLERANCE,
    random_seed=RANDOM_SEED
)

gmm.fit(datos)



# SEGMENTAR IMAGEN


segmentada = gmm.segment(datos)

segmentada = segmentada.reshape(
    rows,
    cols,
    3
)



# MOSTRAR RESULTADOS DEL MODELO


print("\nResultados finales:")

for k in range(N_CLUSTERS):

    print(f"\nCluster {k}")
    print(
        f"Media RGB: "
        f"{gmm.means[k].astype(int)}"
    )

    print(
        f"Probabilidad: "
        f"{gmm.weights[k]:.4f}"
    )



# GUARDAR IMAGEN SEGMENTADA


segmentada_bgr = cv2.cvtColor(
    segmentada,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    "results/segmented.png",
    segmentada_bgr
)



# MOSTRAR IMÁGENES


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmentada)
plt.title(
    f"Segmentación GMM - K={N_CLUSTERS}"
)
plt.axis("off")

plt.tight_layout()
plt.show()



# GRÁFICA DE CONVERGENCIA


plt.figure(figsize=(8, 5))

plt.plot(
    range(
        1,
        len(gmm.log_likelihood_history) + 1
    ),
    gmm.log_likelihood_history
)

plt.xlabel("Iteración")
plt.ylabel("Log-Likelihood")
plt.title("Convergencia del algoritmo EM")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()