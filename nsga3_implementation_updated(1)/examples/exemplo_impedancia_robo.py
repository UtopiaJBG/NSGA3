"""
Script de exemplo para integração de um problema externo ao NSGA-III.

Este script demonstra como implementar e utilizar um problema personalizado
de otimização de impedância para interação física humano-robô com o NSGA-III.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Adicionar diretórios ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsga3_project.problems.dtlz import Problem
from nsga3_project.src.nsga3 import NSGA3

class ImpedanciaRobo(Problem):
    """
    Problema de otimização multiobjetivo para adaptação de impedância em interação física humano-robô.
    
    Objetivos:
    1. Minimizar erro de trajetória (estabilidade)
    2. Minimizar força de interação (conforto)
    3. Minimizar energia de controle (esforço)
    
    Variáveis de decisão:
    - Parâmetros de impedância: rigidez (K), amortecimento (D), inércia (M)
    - Parâmetros de controle adicionais
    """
    
    def __init__(self, n_var=10, n_obj=3):
        """
        Inicializa o problema de impedância.
        
        Args:
            n_var: Número de variáveis de decisão (parâmetros de impedância e controle)
            n_obj: Número de objetivos (estabilidade, conforto, esforço)
        """
        super().__init__(n_var, n_obj)
        
        # Definir limites das variáveis
        # Exemplo: limites para rigidez, amortecimento, inércia e outros parâmetros
        self.bounds = []
        
        # Parâmetros de rigidez (K) para diferentes articulações
        for i in range(3):
            self.bounds.append((10.0, 1000.0))  # Rigidez (N/m ou Nm/rad)
        
        # Parâmetros de amortecimento (D) para diferentes articulações
        for i in range(3):
            self.bounds.append((1.0, 100.0))    # Amortecimento (Ns/m ou Nms/rad)
        
        # Parâmetros de inércia (M) para diferentes articulações
        for i in range(3):
            self.bounds.append((0.1, 10.0))     # Inércia (kg ou kgm²)
        
        # Parâmetro adicional (se n_var > 9)
        for i in range(9, n_var):
            self.bounds.append((0.0, 1.0))
        
        # Parâmetros do modelo de simulação
        self.simulation_steps = 1000
        self.dt = 0.01  # Passo de tempo (s)
        
    def simulate_robot_interaction(self, K, D, M, trajectory):
        """
        Simula a interação física humano-robô com os parâmetros de impedância dados.
        
        Args:
            K: Vetor de rigidez
            D: Vetor de amortecimento
            M: Vetor de inércia
            trajectory: Trajetória desejada
            
        Returns:
            error: Erro de trajetória
            force: Força de interação
            energy: Energia de controle
        """
        # Implementação simplificada da simulação
        # Em um caso real, isso seria substituído por um modelo mais complexo
        # ou por dados experimentais
        
        n_joints = len(K)
        n_steps = len(trajectory)
        
        # Inicializar variáveis
        position = np.zeros((n_steps, n_joints))
        velocity = np.zeros((n_steps, n_joints))
        acceleration = np.zeros((n_steps, n_joints))
        force = np.zeros((n_steps, n_joints))
        energy = np.zeros((n_steps, n_joints))
        
        # Condições iniciais
        position[0] = trajectory[0]
        
        # Simulação
        for t in range(1, n_steps):
            for j in range(n_joints):
                # Modelo de impedância: M*a + D*v + K*x = F_ext
                # Aqui, simplificamos assumindo uma força externa constante
                F_ext = np.sin(t * self.dt) * 10.0  # Força externa simulada
                
                # Calcular erro de posição
                error = trajectory[t, j] - position[t-1, j]
                
                # Calcular aceleração baseada no modelo de impedância
                acceleration[t, j] = (F_ext - D[j] * velocity[t-1, j] - K[j] * error) / M[j]
                
                # Integrar para obter velocidade e posição
                velocity[t, j] = velocity[t-1, j] + acceleration[t, j] * self.dt
                position[t, j] = position[t-1, j] + velocity[t, j] * self.dt
                
                # Calcular força de interação
                force[t, j] = K[j] * error + D[j] * velocity[t, j]
                
                # Calcular energia de controle
                energy[t, j] = 0.5 * K[j] * error**2 + 0.5 * M[j] * velocity[t, j]**2
        
        # Calcular métricas
        mean_error = np.mean(np.abs(trajectory - position))
        mean_force = np.mean(np.abs(force))
        total_energy = np.sum(energy)
        
        return mean_error, mean_force, total_energy
    
    def evaluate(self, x):
        """
        Avalia uma solução e retorna os valores dos objetivos.
        
        Args:
            x: Vetor de variáveis de decisão (parâmetros de impedância e controle)
            
        Returns:
            Vetor de valores dos objetivos (estabilidade, conforto, esforço)
        """
        # Extrair parâmetros de impedância
        K = x[0:3]  # Rigidez
        D = x[3:6]  # Amortecimento
        M = x[6:9]  # Inércia
        
        # Gerar trajetória desejada para simulação
        n_joints = 3
        trajectory = np.zeros((self.simulation_steps, n_joints))
        
        # Trajetória senoidal simples para demonstração
        for t in range(self.simulation_steps):
            for j in range(n_joints):
                trajectory[t, j] = np.sin(t * self.dt * (j + 1))
        
        # Simular interação robô-humano
        error, force, energy = self.simulate_robot_interaction(K, D, M, trajectory)
        
        # Definir objetivos (todos para minimização)
        f = np.zeros(self.n_obj)
        f[0] = error      # Minimizar erro de trajetória (estabilidade)
        f[1] = force      # Minimizar força de interação (conforto)
        f[2] = energy     # Minimizar energia de controle (esforço)
        
        return f

def run_impedancia_example():
    """
    Executa o exemplo de otimização de impedância com NSGA-III.
    """
    print("Executando exemplo de otimização de impedância para interação física humano-robô...")
    
    # Criar diretório para resultados
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar instância do problema
    problema_impedancia = ImpedanciaRobo(n_var=10, n_obj=3)
    
    # Configurar e executar o NSGA-III
    # Usar um número reduzido de gerações para demonstração
    nsga3 = NSGA3(problema_impedancia, max_gen=50)
    population, objectives = nsga3.run(verbose=True)
    
    # Visualizar resultados
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotar pontos
    scatter = ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], 
                         c=np.sum(objectives, axis=1), cmap='viridis', s=30)
    
    # Adicionar barra de cores
    cbar = plt.colorbar(scatter)
    cbar.set_label('Soma dos Objetivos (menor é melhor)')
    
    # Configurar rótulos
    ax.set_xlabel('Erro de Trajetória (Estabilidade)')
    ax.set_ylabel('Força de Interação (Conforto)')
    ax.set_zlabel('Energia de Controle (Esforço)')
    ax.set_title('Fronteira de Pareto para Otimização de Impedância')
    
    plt.savefig(os.path.join(output_dir, 'fronteira_pareto_impedancia.png'), dpi=300)
    
    # Analisar soluções
    print("\nEstatísticas das soluções encontradas:")
    print(f"Número de soluções não-dominadas: {len(objectives)}")
    print(f"Mínimo Erro de Trajetória: {np.min(objectives[:, 0]):.6f}")
    print(f"Mínimo Força de Interação: {np.min(objectives[:, 1]):.6f}")
    print(f"Mínimo Energia de Controle: {np.min(objectives[:, 2]):.6f}")
    
    # Selecionar uma solução de compromisso
    # Método simples: normalizar objetivos e escolher a solução com menor soma
    normalized_obj = (objectives - np.min(objectives, axis=0)) / (np.max(objectives, axis=0) - np.min(objectives, axis=0))
    compromise_idx = np.argmin(np.sum(normalized_obj, axis=1))
    
    print("\nSolução de compromisso selecionada:")
    print(f"Parâmetros de Rigidez (K): {population[compromise_idx, 0:3]}")
    print(f"Parâmetros de Amortecimento (D): {population[compromise_idx, 3:6]}")
    print(f"Parâmetros de Inércia (M): {population[compromise_idx, 6:9]}")
    print(f"Valores dos objetivos: {objectives[compromise_idx]}")
    
    # Salvar resultados em arquivo
    results_file = os.path.join(output_dir, 'resultados_impedancia.md')
    with open(results_file, 'w') as f:
        f.write("# Resultados da Otimização de Impedância\n\n")
        f.write("## Estatísticas das Soluções\n\n")
        f.write(f"Número de soluções não-dominadas: {len(objectives)}\n")
        f.write(f"Mínimo Erro de Trajetória: {np.min(objectives[:, 0]):.6f}\n")
        f.write(f"Mínimo Força de Interação: {np.min(objectives[:, 1]):.6f}\n")
        f.write(f"Mínimo Energia de Controle: {np.min(objectives[:, 2]):.6f}\n\n")
        
        f.write("## Solução de Compromisso\n\n")
        f.write(f"Parâmetros de Rigidez (K): {population[compromise_idx, 0:3]}\n")
        f.write(f"Parâmetros de Amortecimento (D): {population[compromise_idx, 3:6]}\n")
        f.write(f"Parâmetros de Inércia (M): {population[compromise_idx, 6:9]}\n")
        f.write(f"Valores dos objetivos: {objectives[compromise_idx]}\n\n")
        
        f.write("## Todas as Soluções Não-Dominadas\n\n")
        f.write("| Índice | Erro de Trajetória | Força de Interação | Energia de Controle |\n")
        f.write("|--------|-------------------|--------------------|--------------------|")
        for i in range(min(20, len(objectives))):  # Limitar a 20 soluções para legibilidade
            f.write(f"\n| {i} | {objectives[i, 0]:.6f} | {objectives[i, 1]:.6f} | {objectives[i, 2]:.6f} |")
    
    print(f"\nResultados salvos em {results_file}")
    print(f"Visualização salva em {os.path.join(output_dir, 'fronteira_pareto_impedancia.png')}")
    
    return population, objectives

if __name__ == "__main__":
    run_impedancia_example()
