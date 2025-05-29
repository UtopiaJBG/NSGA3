"""
Script para executar um experimento de teste rápido com o NSGA-III.

Este script configura e executa o algoritmo NSGA-III em um problema DTLZ2
com 3 objetivos por um número reduzido de gerações, para verificar se a
implementação está funcionando corretamente.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Adicionar diretórios ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nsga3 import NSGA3
from src.metrics import igd, generate_reference_points_on_pareto_front
from problems.dtlz import DTLZ2

def run_test():
    """
    Executa um teste rápido do NSGA-III com DTLZ2 e 3 objetivos.
    """
    print("Iniciando teste do NSGA-III com DTLZ2 e 3 objetivos...")
    
    # Criar problema
    problem = DTLZ2(n_obj=3)
    
    # Configurar número reduzido de gerações para teste
    max_gen = 50
    
    # Inicializar e executar o NSGA-III
    start_time = time.time()
    nsga3 = NSGA3(problem, max_gen=max_gen)
    population, objectives = nsga3.run(verbose=True)
    end_time = time.time()
    
    # Calcular métricas
    reference_points = generate_reference_points_on_pareto_front('DTLZ2', 3)
    igd_value = igd(objectives, reference_points)
    
    print(f"Teste concluído em {end_time - start_time:.2f} segundos")
    print(f"IGD: {igd_value:.6f}")
    
    # Visualizar resultados
    visualize_results(objectives)
    
    return igd_value

def visualize_results(objectives):
    """
    Visualiza os resultados do teste.
    
    Args:
        objectives: Objetivos da população final
    """
    # Criar diretório para resultados
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualizar fronteira de Pareto para 3 objetivos
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotar pontos
    ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='blue', s=20)
    
    # Configurar rótulos
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    ax.set_title('DTLZ2 - 3 Objetivos (Teste)')
    
    # Ajustar limites dos eixos
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    plt.savefig(os.path.join(output_dir, "test_dtlz2_3obj.png"), dpi=300)
    print(f"Visualização salva em {os.path.join(output_dir, 'test_dtlz2_3obj.png')}")

if __name__ == "__main__":
    run_test()
