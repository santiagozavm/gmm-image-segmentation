import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixture:

    def __init__(
        self,
        n_clusters=3,
        max_iter=100,
        tolerance=1e-5,
        random_seed=42,
        epsilon=1e-6
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_seed = random_seed
        self.epsilon = epsilon

        self.means = None
        self.covariances = None
        self.weights = None
        self.responsibilities = None
        self.labels = None
        self.log_likelihood_history = []


    def _initialize(self, data):

        np.random.seed(self.random_seed)

        n_data = data.shape[0]

        # Seleccionar datos reales como medias iniciales
        indices = np.random.choice(
            n_data,
            self.n_clusters,
            replace=False
        )

        self.means = data[indices].copy()

        # Inicializar las matrices de covarianza
        self.covariances = np.array([
            np.eye(3) * 100.0
            for _ in range(self.n_clusters)
        ])

        # Probabilidades iniciales de mezcla
        self.weights = np.full(
            self.n_clusters,
            1 / self.n_clusters
        )


    def _e_step(self, data):

        n_data = data.shape[0]

        densities = np.zeros(
            (n_data, self.n_clusters)
        )

        # Calcular la densidad de cada distribución
        for k in range(self.n_clusters):

            distribution = multivariate_normal(
                mean=self.means[k],
                cov=self.covariances[k]
            )

            densities[:, k] = distribution.pdf(data)

        # Multiplicar por la probabilidad de mezcla
        probabilities = densities * self.weights

        # Suma de probabilidades de todos los clusters
        total_probability = np.sum(
            probabilities,
            axis=1,
            keepdims=True
        )

        # Evitar divisiones entre cero
        total_probability = np.maximum(
            total_probability,
            np.finfo(float).eps
        )

        # Calcular responsabilidades
        responsibilities = (
            probabilities / total_probability
        )

        return responsibilities


    def _m_step(self, data, responsibilities):

        n_data = data.shape[0]

        # Cantidad efectiva de datos pertenecientes
        # a cada cluster
        responsibility_sum = np.sum(
            responsibilities,
            axis=0
        )

        # ----------------------------------------------------
        # Actualizar medias
        # ----------------------------------------------------

        for k in range(self.n_clusters):

            self.means[k] = (
                np.sum(
                    responsibilities[:, k:k + 1] * data,
                    axis=0
                )
                / responsibility_sum[k]
            )

        # ----------------------------------------------------
        # Actualizar covarianzas
        # ----------------------------------------------------

        for k in range(self.n_clusters):

            difference = data - self.means[k]

            covariance = (
                responsibilities[:, k:k + 1] * difference
            ).T @ difference

            covariance /= responsibility_sum[k]

            # Regularización para evitar matrices singulares
            covariance += (
                self.epsilon * np.eye(3)
            )

            self.covariances[k] = covariance

        # ----------------------------------------------------
        # Actualizar probabilidades de mezcla
        # ----------------------------------------------------

        self.weights = (
            responsibility_sum / n_data
        )


    def _calculate_log_likelihood(self, data):

        total_probability = np.zeros(
            data.shape[0]
        )

        for k in range(self.n_clusters):

            distribution = multivariate_normal(
                mean=self.means[k],
                cov=self.covariances[k]
            )

            density = distribution.pdf(data)

            total_probability += (
                self.weights[k] * density
            )

        total_probability = np.maximum(
            total_probability,
            np.finfo(float).eps
        )

        return np.sum(
            np.log(total_probability)
        )


    def fit(self, data):

        self._initialize(data)

        previous_log_likelihood = None

        for iteration in range(self.max_iter):


            # E-STEP


            responsibilities = self._e_step(data)

            
            # M-STEP
            

            self._m_step(
                data,
                responsibilities
            )

            # Guardar responsabilidades actualizadas
            self.responsibilities = responsibilities

            
            # LOG-LIKELIHOOD
            

            log_likelihood = (
                self._calculate_log_likelihood(data)
            )

            self.log_likelihood_history.append(
                log_likelihood
            )

            print(
                f"Iteración {iteration + 1}: "
                f"Log-Likelihood = {log_likelihood:.2f}"
            )

            
            # CONVERGENCIA
            

            if previous_log_likelihood is not None:

                difference = abs(
                    log_likelihood -
                    previous_log_likelihood
                )

                relative_difference = (
                    difference /
                    abs(previous_log_likelihood)
                )

                if relative_difference < self.tolerance:

                    print(
                        f"\nConvergencia alcanzada en "
                        f"la iteración {iteration + 1}"
                    )

                    break

            previous_log_likelihood = log_likelihood

        # Calcular responsabilidades finales
        self.responsibilities = self._e_step(data)
        
        # Obtener cluster dominante
        self.labels = np.argmax(
            self.responsibilities,
            axis=1
        )

        return self


    def predict(self, data):

        responsibilities = self._e_step(data)

        return np.argmax(
            responsibilities,
            axis=1
        )


    def segment(self, data):

        labels = self.predict(data)

        segmented = np.zeros_like(
            data,
            dtype=np.float64
        )

        for k in range(self.n_clusters):

            segmented[labels == k] = self.means[k]

        segmented = np.clip(
            segmented,
            0,
            255
        ).astype(np.uint8)

        return segmented