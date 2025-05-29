"""
Módulo de associação para o NSGA-III.

Este módulo implementa a operação de associação que associa cada membro
da população a um ponto de referência, conforme descrito na Seção IV.D do artigo:
"Association Operation"

Referência:
Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
Part I: Solving Problems With Box Constraints. IEEE Transactions on
Evolutionary Computation, 18(4), 577-601.
"""

import numpy as np

def associate_to_reference_points(normalized_objectives, reference_points):
    """
    Associa cada membro da população a um ponto de referência.
    
    Implementa o procedimento descrito no Algoritmo 3 do artigo original.
    
    Args:
        normalized_objectives: Matriz de objetivos normalizados (shape: [n_pop, n_obj])
        reference_points: Pontos de referência (shape: [n_ref_points, n_obj])
        
    Returns:
        associations: Array com o índice do ponto de referência associado a cada indivíduo
        perpendicular_distances: Distâncias perpendiculares de cada indivíduo ao ponto de referência associado
    """
    n_pop = normalized_objectives.shape[0]
    n_ref_points = reference_points.shape[0]
    
    # Inicializar arrays para armazenar associações e distâncias
    associations = np.zeros(n_pop, dtype=int)
    perpendicular_distances = np.zeros(n_pop)
    
    # Para cada indivíduo na população
    for i in range(n_pop):
        # Calcular a distância perpendicular a cada linha de referência
        distances = np.zeros(n_ref_points)
        
        for j in range(n_ref_points):
            # Passo 1-2: Definir a linha de referência como o vetor do ponto de referência
            ref_line = reference_points[j]
            
            # Passo 6: Calcular a distância perpendicular
            # Projeção do ponto na linha de referência
            proj = np.dot(normalized_objectives[i], ref_line) / np.dot(ref_line, ref_line) * ref_line
            
            # Distância perpendicular
            dist = np.linalg.norm(normalized_objectives[i] - proj)
            distances[j] = dist
        
        # Passo 8: Associar ao ponto de referência com menor distância perpendicular
        min_idx = np.argmin(distances)
        associations[i] = min_idx
        perpendicular_distances[i] = distances[min_idx]
    
    return associations, perpendicular_distances
