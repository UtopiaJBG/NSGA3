"""
Script para testar a implementação corrigida do NSGA-III com foco na validação do IGD
usando pontos exatos da fronteira de Pareto.

Este script executa o algoritmo NSGA-III em problemas DTLZ específicos
e compara os valores de IGD obtidos com os valores reportados na Tabela 3
do artigo original, utilizando pontos exatos da fronteira de Pareto.
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
from src.metrics import igd
from src.pareto_front import generate_pareto_front
from problems.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4

def test_igd_with_exact_pareto():
    """
    Testa o cálculo do IGD usando pontos exatos da fronteira de Pareto
    para garantir que está alinhado com o artigo original.
    """
    print("Testando cálculo do IGD com pontos exatos da fronteira de Pareto...")
    
    # Criar diretório para resultados
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Valores de referência da Tabela 3 do artigo original para NSGA-III
    reference_values = {
        'DTLZ1': {
            3: {'best': 4.880e-4, 'median': 1.308e-3, 'worst': 4.880e-3},
            5: {'best': 8.710e-4, 'median': 7.580e-3, 'worst': 1.970e-2}
        },
        'DTLZ2': {
            3: {'best': 1.289e-3, 'median': 1.302e-3, 'worst': 2.114e-3},
            5: {'best': 4.294e-3, 'median': 4.862e-3, 'worst': 5.994e-3}
        },
        'DTLZ3': {
            3: {'best': 9.751e-4, 'median': 4.405e-3, 'worst': 6.665e-3},
            5: {'best': 3.680e-3, 'median': 5.994e-3, 'worst': 1.190e-2}
        },
        'DTLZ4': {
            3: {'best': 2.915e-4, 'median': 5.970e-4, 'worst': 7.238e-4},
            5: {'best': 9.849e-4, 'median': 1.255e-3, 'worst': 1.573e-3}
        }
    }
    
    # Configurações de gerações do artigo original
    generations = {
        'DTLZ1': {3: 400, 5: 600},
        'DTLZ2': {3: 250, 5: 350},
        'DTLZ3': {3: 1000, 5: 1000},
        'DTLZ4': {3: 600, 5: 1000}
    }
    
    # Executar testes para DTLZ1 e DTLZ2 com 3 e 5 objetivos
    results = {}
    
    for problem_name in ['DTLZ1', 'DTLZ2']:
        results[problem_name] = {}
        
        for n_obj in [3, 5]:
            print(f"\nTestando {problem_name} com {n_obj} objetivos...")
            
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
            
            # Calcular IGD com normalização
            igd_value = igd(objectives, pareto_front, normalize=True)
            
            # Armazenar resultados
            results[problem_name][n_obj] = {
                'igd': igd_value,
                'reference': reference_values[problem_name][n_obj],
                'execution_time': end_time - start_time
            }
            
            print(f"IGD com pontos exatos: {igd_value:.6e}")
            print(f"Referência (artigo): best={reference_values[problem_name][n_obj]['best']:.6e}, "
                  f"median={reference_values[problem_name][n_obj]['median']:.6e}, "
                  f"worst={reference_values[problem_name][n_obj]['worst']:.6e}")
            
            # Visualizar fronteira de Pareto para 3 objetivos
            if n_obj == 3:
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
                
                plt.savefig(os.path.join(output_dir, f"{problem_name}_{n_obj}obj_front_exact_pareto.png"), dpi=300)
                plt.close()
    
    # Gerar tabela comparativa
    print("\n\nTabela Comparativa de IGD com Pontos Exatos da Fronteira de Pareto:")
    print("=" * 80)
    print(f"{'Problema':<10} {'M':<5} {'IGD (Calculado)':<20} {'Referência (Mediana)':<20} {'Razão':<10}")
    print("-" * 80)
    
    for problem_name in results:
        for n_obj in results[problem_name]:
            ratio = results[problem_name][n_obj]['igd'] / results[problem_name][n_obj]['reference']['median']
            print(f"{problem_name:<10} {n_obj:<5} {results[problem_name][n_obj]['igd']:<20.6e} "
                  f"{results[problem_name][n_obj]['reference']['median']:<20.6e} "
                  f"{ratio:<10.2f}")
    
    print("=" * 80)
    
    # Salvar resultados em arquivo markdown
    with open(os.path.join(output_dir, "igd_exact_pareto_comparison.md"), 'w') as f:
        f.write("# Comparação de Valores de IGD com Pontos Exatos da Fronteira de Pareto\n\n")
        f.write("## Valores Obtidos vs. Valores do Artigo Original\n\n")
        f.write("| Problema | M | IGD (Calculado) | Referência (Mediana) | Razão |\n")
        f.write("|----------|---|-----------------|----------------------|-------|\n")
        
        for problem_name in results:
            for n_obj in results[problem_name]:
                ratio = results[problem_name][n_obj]['igd'] / results[problem_name][n_obj]['reference']['median']
                f.write(f"| {problem_name} | {n_obj} | {results[problem_name][n_obj]['igd']:.6e} | "
                        f"{results[problem_name][n_obj]['reference']['median']:.6e} | {ratio:.2f} |\n")
    
    print(f"\nResultados salvos em {os.path.join(output_dir, 'igd_exact_pareto_comparison.md')}")

if __name__ == "__main__":
    test_igd_with_exact_pareto()
