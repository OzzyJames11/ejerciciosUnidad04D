# -*- coding: utf-8 -*-
"""
Python 3
28 / 07 / 2024
@author: OzzyLoachamin

"""

import numpy as np

def  gauss_jacobi_iteraciones(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales Ax = b mediante el método de Gauss-Jacobi.

    ## Parameters

    ``A``: Matriz de coeficientes de dimensiones n x n.
    ``b``: Vector del término independiente de dimensión n.
    ``x0``: Vector inicial de dimensión n.
    ``tol``: Tolerancia para el criterio de parada.
    ``max_iter``: Número máximo de iteraciones.

    ## Return

    ``x``: Solución aproximada del sistema Ax = b.
    """
    A = np.array(
        A, dtype=float
    )  # convertir en float, porque si no, puede convertir como entero
    
    n = len(b)
    x = x0.copy()

    for k in range(max_iter):
        x_new = np.zeros_like(x)

        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        print("Iteración",k+1,", solución aproximada:",x_new)

def gauss_jacobi(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales Ax = b mediante el método de Gauss-Jacobi.

    ## Parameters

    ``A``: Matriz de coeficientes de dimensiones n x n.
    ``b``: Vector del término independiente de dimensión n.
    ``x0``: Vector inicial de dimensión n.
    ``tol``: Tolerancia para el criterio de parada.
    ``max_iter``: Número máximo de iteraciones.

    ## Return

    ``x``: Solución aproximada del sistema Ax = b.
    """
    A = np.array(
        A, dtype=float
    )  # convertir en float, porque si no, puede convertir como entero
    
    n = len(b)
    x = x0.copy()

    for k in range(max_iter):
        x_new = np.zeros_like(x)

        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        # Verificar el criterio de parada
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new

        x = x_new
        

    raise ValueError("El método de Gauss-Jacobi no convergió.")