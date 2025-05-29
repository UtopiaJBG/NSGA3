"""
Implementação dos problemas de teste DTLZ para o NSGA-III.

Este módulo implementa os problemas de teste DTLZ1, DTLZ2, DTLZ3 e DTLZ4
conforme utilizados no artigo original do NSGA-III.

Referência:
Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
Part I: Solving Problems With Box Constraints. IEEE Transactions on
Evolutionary Computation, 18(4), 577-601.
"""

import numpy as np

class Problem:
    """Classe base para problemas de otimização multi-objetivo."""
    
    def __init__(self, n_var, n_obj):
        """
        Inicializa o problema.
        
        Args:
            n_var: Número de variáveis de decisão
            n_obj: Número de objetivos
        """
        self.n_var = n_var
        self.n_obj = n_obj
        self.bounds = [(0.0, 1.0)] * n_var
    
    def evaluate(self, x):
        """
        Avalia uma solução e retorna os valores dos objetivos.
        
        Args:
            x: Vetor de variáveis de decisão
            
        Returns:
            Vetor de valores dos objetivos
        """
        raise NotImplementedError("Método evaluate deve ser implementado nas subclasses")


class DTLZ1(Problem):
    """
    Problema de teste DTLZ1.
    
    Características:
    - Frente Pareto-ótima: fi ∈ [0, 0.5]
    - Número de variáveis: M + k - 1, onde k = 5
    """
    
    def __init__(self, n_obj, n_var=None):
        """
        Inicializa o problema DTLZ1.
        
        Args:
            n_obj: Número de objetivos
            n_var: Número de variáveis (se None, usa n_obj + 4)
        """
        if n_var is None:
            n_var = n_obj + 4  # k=5
        super().__init__(n_var, n_obj)
    
    def evaluate(self, x):
        """
        Avalia uma solução para o problema DTLZ1.
        
        Args:
            x: Vetor de variáveis de decisão
            
        Returns:
            Vetor de valores dos objetivos
        """
        k = self.n_var - self.n_obj + 1
        
        # Calcular g(xM)
        g = 100 * (k + np.sum((x[self.n_obj-1:] - 0.5)**2 - np.cos(20 * np.pi * (x[self.n_obj-1:] - 0.5))))
        
        # Calcular os objetivos
        f = np.zeros(self.n_obj)
        
        for i in range(self.n_obj):
            f[i] = 0.5 * (1 + g)
            
            for j in range(self.n_obj - i - 1):
                f[i] *= x[j]
            
            if i > 0:
                f[i] *= (1 - x[self.n_obj - i - 1])
        
        return f


class DTLZ2(Problem):
    """
    Problema de teste DTLZ2.
    
    Características:
    - Frente Pareto-ótima: fi ∈ [0, 1]
    - Número de variáveis: M + k - 1, onde k = 10
    """
    
    def __init__(self, n_obj, n_var=None):
        """
        Inicializa o problema DTLZ2.
        
        Args:
            n_obj: Número de objetivos
            n_var: Número de variáveis (se None, usa n_obj + 9)
        """
        if n_var is None:
            n_var = n_obj + 9  # k=10
        super().__init__(n_var, n_obj)
    
    def evaluate(self, x):
        """
        Avalia uma solução para o problema DTLZ2.
        
        Args:
            x: Vetor de variáveis de decisão
            
        Returns:
            Vetor de valores dos objetivos
        """
        k = self.n_var - self.n_obj + 1
        
        # Calcular g(xM)
        g = np.sum((x[self.n_obj-1:] - 0.5)**2)
        
        # Calcular os objetivos
        f = np.zeros(self.n_obj)
        
        for i in range(self.n_obj):
            f[i] = 1 + g
            
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(x[j] * np.pi / 2)
            
            if i > 0:
                f[i] *= np.sin(x[self.n_obj - i - 1] * np.pi / 2)
        
        return f


class DTLZ3(Problem):
    """
    Problema de teste DTLZ3.
    
    Características:
    - Frente Pareto-ótima: fi ∈ [0, 1]
    - Número de variáveis: M + k - 1, onde k = 10
    - Múltiplos frontes locais
    """
    
    def __init__(self, n_obj, n_var=None):
        """
        Inicializa o problema DTLZ3.
        
        Args:
            n_obj: Número de objetivos
            n_var: Número de variáveis (se None, usa n_obj + 9)
        """
        if n_var is None:
            n_var = n_obj + 9  # k=10
        super().__init__(n_var, n_obj)
    
    def evaluate(self, x):
        """
        Avalia uma solução para o problema DTLZ3.
        
        Args:
            x: Vetor de variáveis de decisão
            
        Returns:
            Vetor de valores dos objetivos
        """
        k = self.n_var - self.n_obj + 1
        
        # Calcular g(xM) - similar ao DTLZ1 mas com a forma do DTLZ2
        g = 100 * (k + np.sum((x[self.n_obj-1:] - 0.5)**2 - np.cos(20 * np.pi * (x[self.n_obj-1:] - 0.5))))
        
        # Calcular os objetivos - similar ao DTLZ2
        f = np.zeros(self.n_obj)
        
        for i in range(self.n_obj):
            f[i] = 1 + g
            
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(x[j] * np.pi / 2)
            
            if i > 0:
                f[i] *= np.sin(x[self.n_obj - i - 1] * np.pi / 2)
        
        return f


class DTLZ4(Problem):
    """
    Problema de teste DTLZ4.
    
    Características:
    - Frente Pareto-ótima: fi ∈ [0, 1]
    - Número de variáveis: M + k - 1, onde k = 10
    - Distribuição não uniforme de pontos
    """
    
    def __init__(self, n_obj, n_var=None, alpha=100):
        """
        Inicializa o problema DTLZ4.
        
        Args:
            n_obj: Número de objetivos
            n_var: Número de variáveis (se None, usa n_obj + 9)
            alpha: Parâmetro de mapeamento (controla a densidade)
        """
        if n_var is None:
            n_var = n_obj + 9  # k=10
        super().__init__(n_var, n_obj)
        self.alpha = alpha
    
    def evaluate(self, x):
        """
        Avalia uma solução para o problema DTLZ4.
        
        Args:
            x: Vetor de variáveis de decisão
            
        Returns:
            Vetor de valores dos objetivos
        """
        k = self.n_var - self.n_obj + 1
        
        # Calcular g(xM) - similar ao DTLZ2
        g = np.sum((x[self.n_obj-1:] - 0.5)**2)
        
        # Calcular os objetivos - similar ao DTLZ2 mas com mapeamento não linear
        f = np.zeros(self.n_obj)
        
        for i in range(self.n_obj):
            f[i] = 1 + g
            
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(x[j]**self.alpha * np.pi / 2)
            
            if i > 0:
                f[i] *= np.sin(x[self.n_obj - i - 1]**self.alpha * np.pi / 2)
        
        return f
