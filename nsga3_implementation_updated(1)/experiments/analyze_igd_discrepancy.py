"""
Script para executar análise detalhada das discrepâncias no cálculo do IGD.

Este script utiliza o módulo de análise de IGD para investigar as diferenças
entre os valores calculados e os reportados no artigo original.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Adicionar diretórios ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nsga3 import NSGA3
from src.igd_analysis import analyze_igd_discrepancy
from src.pareto_front import generate_pareto_front
from problems.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4

def run_igd_analysis():
    """
    Executa análise detalhada das discrepâncias no cálculo do IGD.
    """
    print("Executando análise detalhada das discrepâncias no cálculo do IGD...")
    
    # Criar diretório para resultados
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurações de gerações do artigo original
    generations = {
        'DTLZ1': {3: 400, 5: 600},
        'DTLZ2': {3: 250, 5: 350}
    }
    
    # Executar análise para DTLZ1 e DTLZ2 com 3 objetivos
    for problem_name in ['DTLZ1', 'DTLZ2']:
        for n_obj in [3]:
            print(f"\nAnalisando {problem_name} com {n_obj} objetivos...")
            
            # Criar problema
            if problem_name == 'DTLZ1':
                problem = DTLZ1(n_obj)
            else:
                problem = DTLZ2(n_obj)
            
            # Configurar número de gerações
            max_gen = generations[problem_name][n_obj]
            
            # Executar NSGA-III
            start_time = time.time()
            nsga3 = NSGA3(problem, max_gen=max_gen)
            population, objectives = nsga3.run(verbose=True)
            end_time = time.time()
            
            # Gerar pontos exatos da fronteira de Pareto
            pareto_front = generate_pareto_front(problem_name, n_obj, n_points=10000)
            
            # Analisar discrepâncias no cálculo do IGD
            report = analyze_igd_discrepancy(problem_name, n_obj, objectives, pareto_front)
            
            # Salvar relatório
            report_file = os.path.join(output_dir, f"igd_analysis_{problem_name}_{n_obj}obj.md")
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"Análise salva em {report_file}")
            
            # Visualizar fronteira de Pareto para 3 objetivos
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plotar pontos obtidos pelo NSGA-III
            ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='blue', s=20, label='NSGA-III')
            
            # Plotar amostra da fronteira de Pareto exata
            sample_size = min(500, len(pareto_front))
            sample_indices = np.random.choice(len(pareto_front), sample_size, replace=False)
            sample_front = pareto_front[sample_indices]
            ax.scatter(sample_front[:, 0], sample_front[:, 1], sample_front[:, 2], c='red', s=5, alpha=0.5, label='Pareto Exato')
            
            # Configurar rótulos
            ax.set_xlabel('f1')
            ax.set_ylabel('f2')
            ax.set_zlabel('f3')
            ax.set_title(f'{problem_name} - 3 Objetivos')
            ax.legend()
            
            # Ajustar limites dos eixos
            if problem_name == 'DTLZ1':
                ax.set_xlim(0, 0.5)
                ax.set_ylim(0, 0.5)
                ax.set_zlim(0, 0.5)
            else:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_zlim(0, 1)
            
            plt.savefig(os.path.join(output_dir, f"{problem_name}_{n_obj}obj_analysis.png"), dpi=300)
            plt.close()
    
    print("\nAnálise concluída. Relatórios salvos no diretório 'results'.")

if __name__ == "__main__":
    run_igd_analysis()
