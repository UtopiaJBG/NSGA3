"""
Script para executar experimentos completos com o NSGA-III em problemas DTLZ.

Este script executa o algoritmo NSGA-III nos problemas de teste DTLZ1, DTLZ2, DTLZ3 e DTLZ4
com diferentes números de objetivos, conforme os experimentos descritos no artigo original,
mas com um número reduzido de execuções para demonstração.
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

def run_experiments():
    """
    Executa experimentos com o NSGA-III em problemas DTLZ.
    """
    # Configurações dos experimentos
    problems = {
        'DTLZ1': DTLZ1,
        'DTLZ2': DTLZ2,
        'DTLZ3': DTLZ3,
        'DTLZ4': DTLZ4
    }
    
    # Número de objetivos a testar (reduzido para demonstração)
    objectives = [3, 5]
    
    # Número de execuções para cada configuração (reduzido para demonstração)
    n_runs = 3
    
    # Número de gerações para cada problema (reduzido para demonstração)
    generations = {
        'DTLZ1': 200,
        'DTLZ2': 150,
        'DTLZ3': 300,
        'DTLZ4': 200
    }
    
    # Criar diretório para resultados
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Armazenar resultados
    all_results = []
    
    # Executar experimentos
    for problem_name, problem_class in problems.items():
        for n_obj in objectives:
            for run in range(n_runs):
                print(f"Executando {problem_name} com {n_obj} objetivos (Execução {run+1}/{n_runs})")
                
                # Criar problema
                problem = problem_class(n_obj)
                
                # Inicializar e executar o NSGA-III
                start_time = time.time()
                nsga3 = NSGA3(problem, max_gen=generations[problem_name])
                population, objectives_values = nsga3.run(verbose=True)
                end_time = time.time()
                
                # Calcular IGD
                reference_points = generate_reference_points_on_pareto_front(problem_name, n_obj)
                igd_value = igd(objectives_values, reference_points)
                
                # Armazenar resultados
                result = {
                    'problem': problem_name,
                    'n_obj': n_obj,
                    'run': run,
                    'population': population,
                    'objectives': objectives_values,
                    'igd': igd_value,
                    'execution_time': end_time - start_time,
                    'history': nsga3.history
                }
                
                all_results.append(result)
                
                # Salvar resultado individual
                result_file = os.path.join(results_dir, f"{problem_name}_{n_obj}obj_run{run}.pkl")
                with open(result_file, 'wb') as f:
                    pickle.dump(result, f)
                
                # Visualizar fronteira para 3 objetivos
                if n_obj == 3:
                    visualize_front(result, results_dir)
                
                # Visualizar convergência do IGD
                visualize_convergence(result, results_dir)
    
    # Salvar todos os resultados
    summary_file = os.path.join(results_dir, "all_results.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Gerar tabela de IGD
    generate_igd_table(all_results, results_dir)
    
    print(f"Todos os experimentos concluídos. Resultados salvos em {results_dir}")

def visualize_front(result, output_dir):
    """
    Visualiza a fronteira de Pareto para problemas com 3 objetivos.
    
    Args:
        result: Resultado de um experimento
        output_dir: Diretório para salvar a visualização
    """
    problem = result['problem']
    n_obj = result['n_obj']
    run = result['run']
    objectives = result['objectives']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotar pontos
    ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='blue', s=20)
    
    # Configurar rótulos
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    ax.set_title(f'{problem} - {n_obj} Objetivos (Execução {run+1})')
    
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
    plt.savefig(os.path.join(output_dir, f"{problem}_{n_obj}obj_run{run}_front.png"), dpi=300)
    plt.close()

def visualize_convergence(result, output_dir):
    """
    Visualiza a convergência do IGD ao longo das gerações.
    
    Args:
        result: Resultado de um experimento
        output_dir: Diretório para salvar a visualização
    """
    problem = result['problem']
    n_obj = result['n_obj']
    run = result['run']
    history = result['history']
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['gen'], history['igd'])
    plt.xlabel('Geração')
    plt.ylabel('IGD')
    plt.title(f'{problem} - {n_obj} Objetivos - Convergência do IGD (Execução {run+1})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{problem}_{n_obj}obj_run{run}_convergence.png"), dpi=300)
    plt.close()

def generate_igd_table(results, output_dir):
    """
    Gera uma tabela com os valores de IGD para cada problema e número de objetivos.
    
    Args:
        results: Lista de resultados de experimentos
        output_dir: Diretório para salvar a tabela
    """
    # Agrupar resultados por problema e número de objetivos
    grouped = {}
    for result in results:
        problem = result['problem']
        n_obj = result['n_obj']
        key = (problem, n_obj)
        
        if key not in grouped:
            grouped[key] = []
        
        grouped[key].append(result['igd'])
    
    # Calcular estatísticas
    stats = {}
    for key, values in grouped.items():
        values.sort()
        stats[key] = {
            'best': values[0],
            'median': values[len(values) // 2] if len(values) % 2 == 1 else 
                     (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2,
            'worst': values[-1],
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values)
        }
    
    # Gerar tabela
    with open(os.path.join(output_dir, "igd_table.md"), 'w') as f:
        f.write("# Resultados de IGD\n\n")
        f.write("## Valores de IGD (Inverted Generational Distance)\n\n")
        f.write("| Problema | Objetivos | Melhor | Mediana | Pior | Média | Desvio Padrão | Execuções |\n")
        f.write("|----------|-----------|--------|---------|------|-------|---------------|----------|\n")
        
        for problem in ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4']:
            for n_obj in [3, 5]:
                key = (problem, n_obj)
                if key in stats:
                    s = stats[key]
                    f.write(f"| {problem} | {n_obj} | {s['best']:.6f} | {s['median']:.6f} | {s['worst']:.6f} | {s['mean']:.6f} | {s['std']:.6f} | {s['count']} |\n")

if __name__ == "__main__":
    run_experiments()
