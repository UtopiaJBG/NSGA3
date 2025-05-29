"""
Script para executar um experimento específico com o NSGA-III.

Este script configura e executa o algoritmo NSGA-III em um problema DTLZ
específico com um número definido de objetivos, permitindo testes rápidos
e visualização de resultados.

Referência:
Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
Part I: Solving Problems With Box Constraints. IEEE Transactions on
Evolutionary Computation, 18(4), 577-601.
"""

import os
import sys
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Adicionar diretórios ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nsga3 import NSGA3
from src.metrics import igd, calculate_igd_with_exact_pareto
from src.pareto_front import generate_pareto_front
from problems.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4

def run_single_experiment(problem_name, n_obj, max_gen=None, visualize=True):
    """
    Executa um único experimento com o NSGA-III.
    
    Args:
        problem_name: Nome do problema ('DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4')
        n_obj: Número de objetivos
        max_gen: Número máximo de gerações (se None, usa valores do paper)
        visualize: Se True, gera visualizações dos resultados
        
    Returns:
        results: Dicionário com resultados do experimento
    """
    # Mapear nome do problema para classe
    problem_classes = {
        'DTLZ1': DTLZ1,
        'DTLZ2': DTLZ2,
        'DTLZ3': DTLZ3,
        'DTLZ4': DTLZ4
    }
    
    # Valores padrão de gerações do paper
    default_generations = {
        'DTLZ1': {3: 400, 5: 600, 8: 750, 10: 1000, 15: 1500},
        'DTLZ2': {3: 250, 5: 350, 8: 500, 10: 750, 15: 1000},
        'DTLZ3': {3: 1000, 5: 1000, 8: 1000, 10: 1500, 15: 2000},
        'DTLZ4': {3: 600, 5: 1000, 8: 1250, 10: 2000, 15: 3000}
    }
    
    # Verificar se o problema é válido
    if problem_name not in problem_classes:
        raise ValueError(f"Problema inválido: {problem_name}. Escolha entre DTLZ1, DTLZ2, DTLZ3 ou DTLZ4.")
    
    # Criar problema
    problem_class = problem_classes[problem_name]
    problem = problem_class(n_obj)
    
    # Configurar número máximo de gerações
    if max_gen is None:
        if n_obj in default_generations[problem_name]:
            max_gen = default_generations[problem_name][n_obj]
        else:
            # Valor padrão para casos não especificados
            max_gen = 500
    
    print(f"Executando {problem_name} com {n_obj} objetivos por {max_gen} gerações")
    
    # Inicializar e executar o NSGA-III
    start_time = time.time()
    nsga3 = NSGA3(problem, max_gen=max_gen)
    population, objectives = nsga3.run(verbose=True)
    end_time = time.time()
    
    # Calcular métricas
    reference_points = generate_pareto_front(problem_name, n_obj)
    igd_value = calculate_igd_with_exact_pareto(problem_name, n_obj, objectives)
    
    # Armazenar resultados
    results = {
        'problem': problem_name,
        'n_obj': n_obj,
        'population': population,
        'objectives': objectives,
        'igd': igd_value,
        'execution_time': end_time - start_time,
        'history': nsga3.history
    }
    
    print(f"Experimento concluído em {end_time - start_time:.2f} segundos")
    print(f"IGD: {igd_value:.6f}")
    
    # Visualizar resultados
    if visualize:
        visualize_results(results)
    
    return results

def visualize_results(results):
    """
    Visualiza os resultados de um experimento.
    
    Args:
        results: Dicionário com resultados do experimento
    """
    problem = results['problem']
    n_obj = results['n_obj']
    objectives = results['objectives']
    
    # Criar diretório para resultados
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualizar fronteira de Pareto para 2 ou 3 objetivos
    if n_obj == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(objectives[:, 0], objectives[:, 1], c='blue', s=20)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title(f'{problem} - 2 Objetivos')
        
        # Ajustar limites dos eixos
        if problem == 'DTLZ1':
            plt.xlim(0, 0.5)
            plt.ylim(0, 0.5)
        else:
            plt.xlim(0, 1)
            plt.ylim(0, 1)
        
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{problem}_{n_obj}obj_front.png"), dpi=300)
        plt.show()
    
    elif n_obj == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotar pontos
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='blue', s=20)
        
        # Configurar rótulos
        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        ax.set_zlabel('f3')
        ax.set_title(f'{problem} - 3 Objetivos')
        
        # Ajustar limites dos eixos
        if problem == 'DTLZ1':
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 0.5)
            ax.set_zlim(0, 0.5)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
        
        plt.savefig(os.path.join(output_dir, f"{problem}_{n_obj}obj_front.png"), dpi=300)
        plt.show()
    
    else:
        # Para mais de 3 objetivos, visualizar matriz de dispersão
        # Limitar a visualização a pares de objetivos (máximo 5 para legibilidade)
        max_vis_obj = min(5, n_obj)
        
        fig, axes = plt.subplots(max_vis_obj, max_vis_obj, figsize=(15, 15))
        fig.suptitle(f'{problem} - {n_obj} Objetivos (Matriz de Dispersão)', fontsize=16)
        
        for i in range(max_vis_obj):
            for j in range(max_vis_obj):
                if i != j:
                    axes[i, j].scatter(objectives[:, j], objectives[:, i], c='blue', s=10)
                    
                    # Ajustar limites dos eixos
                    if problem == 'DTLZ1':
                        axes[i, j].set_xlim(0, 0.5)
                        axes[i, j].set_ylim(0, 0.5)
                    else:
                        axes[i, j].set_xlim(0, 1)
                        axes[i, j].set_ylim(0, 1)
                else:
                    axes[i, j].text(0.5, 0.5, f'f{i+1}', ha='center', va='center', fontsize=12)
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                
                # Remover rótulos internos para melhor visualização
                if i < max_vis_obj - 1:
                    axes[i, j].set_xticks([])
                if j > 0:
                    axes[i, j].set_yticks([])
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f"{problem}_{n_obj}obj_scatter_matrix.png"), dpi=300)
        plt.show()
    
    # Visualizar convergência do IGD ao longo das gerações
    if 'history' in results and 'igd' in results['history']:
        plt.figure(figsize=(10, 6))
        plt.plot(results['history']['gen'], results['history']['igd'])
        plt.xlabel('Geração')
        plt.ylabel('IGD')
        plt.title(f'{problem} - {n_obj} Objetivos - Convergência do IGD')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{problem}_{n_obj}obj_igd_convergence.png"), dpi=300)
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Executar um experimento com NSGA-III')
    parser.add_argument('--problem', type=str, default='DTLZ2', choices=['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4'],
                        help='Problema de teste (DTLZ1, DTLZ2, DTLZ3, DTLZ4)')
    parser.add_argument('--objectives', type=int, default=3,
                        help='Número de objetivos')
    parser.add_argument('--generations', type=int, default=None,
                        help='Número máximo de gerações')
    parser.add_argument('--no-visualize', action='store_false', dest='visualize',
                        help='Desativar visualização dos resultados')
    
    args = parser.parse_args()
    
    run_single_experiment(args.problem, args.objectives, args.generations, args.visualize)
