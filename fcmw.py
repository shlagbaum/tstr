from typing import Optional

import numpy as np
from pydantic import BaseModel, Extra, Field, validate_arguments

from my_typing import ArrayLike


class FCMW(BaseModel):
    n_clusters: int = Field(5, ge=1, le=100)
    max_iter: int = Field(150, ge=1, le=1000)
    m: float = Field(2.0, ge=1.0)
    error: float = Field(1e-5, ge=1e-9)
    random_state: Optional[int] = None
    trained: bool = Field(False, const=True)

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True

    @validate_arguments
    def fit(self, X: ArrayLike, W: ArrayLike = None, C: ArrayLike = None, uinit = None, centers_init = None) -> None:
        """Train the fuzzy-c-means model..

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training instances to cluster.
        """
        n_samples = X.shape[0]
        if W is None:
            self.W = np.ones(n_samples)
        else:
            self.W = W / np.mean(W)
        if C is None:
            self.C = np.ones(n_samples)
        if uinit is None:
            self.rng = np.random.default_rng(self.random_state)
            self.u = self.rng.uniform(size=(n_samples, self.n_clusters))
        else: 
            self.u = uinit
        self.u = self.u / np.tile(
            self.u.sum(axis=1)[np.newaxis].T, self.n_clusters
        )
        if not centers_init is None:
            self._centers = centers_init
            self.u = self.soft_predict(X, self.W, self.C)
        for _ in range(self.max_iter):
            u_old = self.u.copy()
            self._centers = FCMW._next_centers(X, self.u, self.m)
            self.u = self.soft_predict(X, self.W, self.C)
            # Stopping rule
            if np.linalg.norm(self.u - u_old) < self.error:
                break
        self.trained = True

    def soft_predict(self, X: ArrayLike, W: ArrayLike = None, C: ArrayLike = None) -> ArrayLike:
        """Soft predict of FCM

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        array, shape = [n_samples, n_clusters]
            Fuzzy partition array, returned as an array with n_samples rows
            and n_clusters columns.
        """
        n_samples = X.shape[0]
        if W is None:
            W = np.ones(n_samples)
        else:
            W = W / np.mean(W)
        if C is None:
            C = np.ones(n_samples)
    
        temp = FCMW._dist(X, self._centers)
        if self.m <= 1.0:
            idx = temp.argmin(axis = 0)
            temp[:,:] = 0.0
            temp[:,idx] = 1.0
        else:
            temp = temp ** float(2 / (self.m - 1))
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(
            temp.shape[-1], axis=1
        )
        denominator_ = ((np.sqrt(W.reshape((temp.shape[0], 1))).repeat(temp.shape[-1], axis=1)) * temp)[:, :, np.newaxis] / denominator_
        return C.reshape((temp.shape[0],1)).repeat(temp.shape[-1], axis=1) / denominator_.sum(2)

    @validate_arguments
    def predict_memberships(self, X: ArrayLike, W: ArrayLike = None, C: ArrayLike = None):
        """Predict the memberships wrt each cluster each sample in X belongs to.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        memberships : array, shape = [n_samples, n_clusters]
            memberships of the cluster each sample belongs to.
        """
        if self.is_trained():
            X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
            return self.soft_predict(X, W, C)

    @validate_arguments
    def predict(self, X: ArrayLike, W: ArrayLike = None, C: ArrayLike = None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape = [n_samples,]
            Index of the cluster each sample belongs to.
        """
        if self.is_trained():
            X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
            return self.soft_predict(X, W, C).argmax(axis=-1)

    def is_trained(self) -> bool:
        if self.trained:
            return True
        return False

    @staticmethod
    def _dist(A: ArrayLike, B: ArrayLike):
        """Compute the euclidean distance two matrices"""
        return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2))

    @staticmethod
    def _next_centers(X, u, m):
        """Update cluster centers"""
        um = u**m
        return (X.T @ um / np.sum(um, axis=0)).T

    @property
    def centers(self):
        if self.is_trained():
            return self._centers
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    @property
    def partition_coefficient(self) -> float:
        """Partition coefficient
        (Equation 12a of https://doi.org/10.1016/0098-3004(84)90020-7)

        Returns
        -------
        float
            partition coefficient of clustering model
        """
        if self.is_trained():
            return np.mean(self.u**2)
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    @property
    def partition_entropy_coefficient(self):
        if self.is_trained():
            return -np.mean(self.u * np.log2(self.u))
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )
