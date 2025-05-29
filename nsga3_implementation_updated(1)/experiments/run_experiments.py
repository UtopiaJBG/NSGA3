"""
Script para executar experimentos com o NSGA-III em problemas DTLZ.

Este script configura e executa o algoritmo NSGA-III nos problemas de teste
DTLZ1, DTLZ2, DTLZ3 e DTLZ4 com diferentes números de objetivos, conforme
os experimentos descritos no artigo original.

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
from src.metrics import igd, generate_reference_points_on_pareto_front
from problems.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4

# Configurações dos experimentos
PROBLEMS = {
    'DTLZ1': DTLZ1,
    'DTLZ2': DTLZ2,
    'DTLZ3': DTLZ3,
    'DTLZ4': DTLZ4
}

# Número de objetivos a testar
OBJECTIVES = [3, 5, 8, 10, 15]

# Número máximo de gerações para cada problema e número de objetivos
# Conforme Tabela III do artigo
MAX_GENERATIONS = {
    'DTLZ1': {3: 400, 5: 600, 8: 750, 10: 1000, 15: 1500},
    'DTLZ2': {3: 250, 5: 350, 8: 500, 10: 750, 15: 1000},
    'DTLZ3': {3: 1000, 5: 1000, 8: 1000, 10: 1500, 15: 2000},
    'DTLZ4': {3: 600, 5: 1000, 8: 1250, 10: 2000, 15: 3000}
}

# Número de execuções para cada configuração
N_RUNS = 20

def run_experiment(problem_name, n_obj, run_id):
    """
    Executa um experimento com o NSGA-III em um problema específico.
    
    Args:
        problem_name: Nome do problema ('DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4')
        n_obj: Número de objetivos
        run_id: ID da execução
        
    Returns:
        results: Dicionário com resultados do experimento
    """
    print(f"Executando {problem_name} com {n_obj} objetivos (Execução {run_id+1}/{N_RUNS})")
    
    # Criar problema
    problem_class = PROBLEMS[problem_name]
    problem = problem_class(n_obj)
    
    # Configurar número máximo de gerações
    max_gen = MAX_GENERATIONS[problem_name][n_obj]
    
    # Inicializar e executar o NSGA-III
    start_time = time.time()
    nsga3 = NSGA3(problem, max_gen=max_gen)
    population, objectives = nsga3.run(verbose=False)
    end_time = time.time()
    
    # Calcular métricas
    reference_points = generate_reference_points_on_pareto_front(problem_name, n_obj)
    igd_value = igd(objectives, reference_points)
    
    # Armazenar resultados
    results = {
        'problem': problem_name,
        'n_obj': n_obj,
        'run_id': run_id,
        'population': population,
        'objectives': objectives,
        'igd': igd_value,
        'execution_time': end_time - start_time,
        'history': nsga3.history
    }
    
    return results

def save_results(results, output_dir):
    """
    Salva os resultados dos experimentos.
    
    Args:
        results: Lista de resultados de experimentos
        output_dir: Diretório para salvar os resultados
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Agrupar resultados por problema e número de objetivos
    grouped_results = {}
    for result in results:
        problem = result['problem']
        n_obj = result['n_obj']
        key = f"{problem}_{n_obj}"
        
        if key not in grouped_results:
            grouped_results[key] = []
        
        grouped_results[key].append(result)
    
    # Salvar resultados agrupados
    for key, group in grouped_results.items():
        filename = os.path.join(output_dir, f"{key}_results.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(group, f)
    
    # Salvar resumo dos resultados
    summary = []
    for result in results:
        summary.append({
            'problem': result['problem'],
            'n_obj': result['n_obj'],
            'run_id': result['run_id'],
            'igd': result['igd'],
            'execution_time': result['execution_time']
        })
    
    summary_file = os.path.join(output_dir, "summary.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    # Gerar tabela de IGD
    generate_igd_table(results, output_dir)
    
    # Gerar visualizações para problemas com 3 objetivos
    for problem in PROBLEMS:
        results_3obj = [r for r in results if r['problem'] == problem and r['n_obj'] == 3]
        if results_3obj:
            # Encontrar o resultado com IGD mediano
            results_3obj.sort(key=lambda x: x['igd'])
            median_result = results_3obj[len(results_3obj) // 2]
            visualize_3d_front(median_result, output_dir)

def generate_igd_table(results, output_dir):
    """
    Gera uma tabela com os valores de IGD para cada problema e número de objetivos.
    
    Args:
        results: Lista de resultados de experimentos
        output_dir: Diretório para salvar a tabela
    """
    # Agrupar resultados por problema e número de objetivos
    igd_values = {}
    for result in results:
        problem = result['problem']
        n_obj = result['n_obj']
        key = (problem, n_obj)
        
        if key not in igd_values:
            igd_values[key] = []
        
        igd_values[key].append(result['igd'])
    
    # Calcular estatísticas
    igd_stats = {}
    for key, values in igd_values.items():
        values.sort()
        igd_stats[key] = {
            'best': values[0],
            'median': values[len(values) // 2],
            'worst': values[-1]
        }
    
    # Gerar tabela
    with open(os.path.join(output_dir, "igd_table.txt"), 'w') as f:
        f.write("Problema | Objetivos | Melhor IGD | IGD Mediano | Pior IGD\n")
        f.write("---------|-----------|------------|-------------|--------\n")
        
        for problem in sorted(PROBLEMS.keys()):
            for n_obj in sorted(OBJECTIVES):
                key = (problem, n_obj)
                if key in igd_stats:
                    stats = igd_stats[key]
                    f.write(f"{problem} | {n_obj} | {stats['best']:.6f} | {stats['median']:.6f} | {stats['worst']:.6f}\n")

def visualize_3d_front(result, output_dir):
    """
    Visualiza a fronteira de Pareto para problemas com 3 objetivos.
    
    Args:
        result: Resultado de um experimento
        output_dir: Diretório para salvar a visualização
    """
    if result['n_obj'] != 3:
        return
    
    problem = result['problem']
    objectives = result['objectives']
    
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
    
    # Salvar figura
    plt.savefig(os.path.join(output_dir, f"{problem}_3obj_front.png"), dpi=300)
    plt.close()

def main():
    """Função principal para executar todos os experimentos."""
    # Criar diretório para resultados
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Lista para armazenar todos os resultados
    all_results = []
    
    # Executar experimentos para cada problema, número de objetivos e execução
    for problem_name in PROBLEMS:
        for n_obj in OBJECTIVES:
            for run_id in range(N_RUNS):
                try:
                    result = run_experiment(problem_name, n_obj, run_id)
                    all_results.append(result)
                    
                    # Salvar resultados parciais
                    if (run_id + 1) % 5 == 0:
                        save_results(all_results, output_dir)
                except Exception as e:
                    print(f"Erro ao executar {problem_name} com {n_obj} objetivos (Execução {run_id+1}): {e}")
    
    # Salvar todos os resultados
    save_results(all_results, output_dir)
    print(f"Todos os experimentos concluídos. Resultados salvos em {output_dir}")

if __name__ == "__main__":
    main()
