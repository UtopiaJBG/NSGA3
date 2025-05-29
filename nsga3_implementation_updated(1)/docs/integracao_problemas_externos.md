"""
Guia para Integração de Problemas Externos ao NSGA-III

Este documento explica como integrar problemas de otimização multiobjetivo externos
à implementação do NSGA-III, incluindo problemas personalizados e casos de uso reais.
"""

# Integração de Problemas Externos ao NSGA-III

## Introdução

A implementação do NSGA-III fornecida neste projeto foi projetada para ser modular e facilmente extensível. 
Este guia explica como integrar seus próprios problemas de otimização multiobjetivo à implementação, 
permitindo que você aplique o algoritmo NSGA-III a uma ampla variedade de cenários reais.

## Estrutura Básica de um Problema

Para integrar um problema externo ao NSGA-III, você precisa criar uma classe que herda da classe base `Problem` 
e implementa o método `evaluate()`. A classe deve ter a seguinte estrutura:

```python
from nsga3_project.problems.dtlz import Problem

class MeuProblema(Problem):
    """
    Implementação de um problema personalizado.
    """
    
    def __init__(self, n_var, n_obj):
        """
        Inicializa o problema.
        
        Args:
            n_var: Número de variáveis de decisão
            n_obj: Número de objetivos
        """
        super().__init__(n_var, n_obj)
        
        # Definir limites das variáveis (por padrão, [0, 1])
        self.bounds = [(lower_i, upper_i) for i in range(n_var)]
    
    def evaluate(self, x):
        """
        Avalia uma solução e retorna os valores dos objetivos.
        
        Args:
            x: Vetor de variáveis de decisão
            
        Returns:
            Vetor de valores dos objetivos
        """
        # Implementar a função de avaliação
        f = np.zeros(self.n_obj)
        
        # Calcular os valores dos objetivos
        # ...
        
        return f
```

## Exemplo: Problema de Otimização de Impedância para Interação Física Humano-Robô

Vamos implementar um exemplo específico para o problema de otimização multiobjetivo para adaptação de impedância 
durante uma interação física humano-robô, onde múltiplos critérios (conforto, estabilidade e esforço) precisam ser equilibrados.

```python
import numpy as np
from nsga3_project.problems.dtlz import Problem

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
```

## Exemplo de Uso

Aqui está um exemplo de como usar o problema personalizado com o NSGA-III:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nsga3_project.src.nsga3 import NSGA3
from nsga3_project.src.metrics import calculate_igd_with_exact_pareto

# Criar instância do problema
problema_impedancia = ImpedanciaRobo(n_var=10, n_obj=3)

# Configurar e executar o NSGA-III
nsga3 = NSGA3(problema_impedancia, max_gen=100)
population, objectives = nsga3.run(verbose=True)

# Visualizar resultados
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotar pontos
ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='blue', s=20)

# Configurar rótulos
ax.set_xlabel('Erro de Trajetória (Estabilidade)')
ax.set_ylabel('Força de Interação (Conforto)')
ax.set_zlabel('Energia de Controle (Esforço)')
ax.set_title('Fronteira de Pareto para Otimização de Impedância')

plt.savefig('fronteira_pareto_impedancia.png', dpi=300)
plt.show()

# Analisar soluções
print("Estatísticas das soluções encontradas:")
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
```

## Integração com Bibliotecas Externas

Você também pode integrar problemas de outras bibliotecas, como DEAP ou Platypus, adaptando suas funções de avaliação para o formato esperado pelo NSGA-III.

### Exemplo com DEAP

```python
import numpy as np
from deap import benchmarks
from nsga3_project.problems.dtlz import Problem

class DEAPProblem(Problem):
    """
    Adaptador para problemas da biblioteca DEAP.
    """
    
    def __init__(self, deap_function, n_var, n_obj, bounds=None):
        """
        Inicializa o adaptador para problemas DEAP.
        
        Args:
            deap_function: Função de benchmark da DEAP
            n_var: Número de variáveis de decisão
            n_obj: Número de objetivos
            bounds: Limites das variáveis (se None, usa [0, 1])
        """
        super().__init__(n_var, n_obj)
        
        self.deap_function = deap_function
        
        if bounds is None:
            self.bounds = [(0.0, 1.0) for _ in range(n_var)]
        else:
            self.bounds = bounds
    
    def evaluate(self, x):
        """
        Avalia uma solução usando a função da DEAP.
        
        Args:
            x: Vetor de variáveis de decisão
            
        Returns:
            Vetor de valores dos objetivos
        """
        return np.array(self.deap_function(x))

# Exemplo de uso com o benchmark ZDT1 da DEAP
zdt1_problem = DEAPProblem(
    lambda x: benchmarks.zdt1(x), 
    n_var=30, 
    n_obj=2
)

# Configurar e executar o NSGA-III
nsga3 = NSGA3(zdt1_problem, max_gen=100)
population, objectives = nsga3.run(verbose=True)
```

### Exemplo com Platypus

```python
import numpy as np
from platypus import DTLZ2 as PlatypusDTLZ2
from nsga3_project.problems.dtlz import Problem

class PlatypusProblem(Problem):
    """
    Adaptador para problemas da biblioteca Platypus.
    """
    
    def __init__(self, platypus_problem):
        """
        Inicializa o adaptador para problemas Platypus.
        
        Args:
            platypus_problem: Instância de problema da Platypus
        """
        n_var = platypus_problem.nvars
        n_obj = platypus_problem.nobjs
        super().__init__(n_var, n_obj)
        
        self.platypus_problem = platypus_problem
        
        # Extrair limites das variáveis
        self.bounds = [(var.lower_bound, var.upper_bound) for var in platypus_problem.variables]
    
    def evaluate(self, x):
        """
        Avalia uma solução usando o problema da Platypus.
        
        Args:
            x: Vetor de variáveis de decisão
            
        Returns:
            Vetor de valores dos objetivos
        """
        solution = self.platypus_problem.evaluate(x)
        return np.array(solution.objectives)

# Exemplo de uso com o problema DTLZ2 da Platypus
platypus_dtlz2 = PlatypusProblem(
    PlatypusDTLZ2(nvars=12, nobjs=3)
)

# Configurar e executar o NSGA-III
nsga3 = NSGA3(platypus_dtlz2, max_gen=100)
population, objectives = nsga3.run(verbose=True)
```

## Dicas para Implementação de Problemas Reais

1. **Normalização de Objetivos**: Em problemas reais, os objetivos podem ter escalas muito diferentes. O NSGA-III já inclui normalização adaptativa, mas você pode pré-processar seus objetivos para melhorar a convergência.

2. **Restrições**: Para problemas com restrições, você pode implementar um método adicional `is_feasible(x)` em sua classe de problema e modificar o método `evaluate()` para penalizar soluções inviáveis.

3. **Funções Objetivas Computacionalmente Intensivas**: Se suas funções objetivas forem computacionalmente intensivas, considere implementar cache ou paralelização na avaliação.

4. **Ajuste de Parâmetros**: Os parâmetros do NSGA-III (tamanho da população, número de gerações, operadores genéticos) podem precisar de ajustes para problemas específicos.

## Exemplo de Implementação com Restrições

```python
class ProblemaComRestricoes(Problem):
    """
    Exemplo de problema com restrições.
    """
    
    def __init__(self, n_var, n_obj):
        super().__init__(n_var, n_obj)
        self.bounds = [(0.0, 1.0) for _ in range(n_var)]
    
    def is_feasible(self, x):
        """
        Verifica se uma solução é viável.
        
        Args:
            x: Vetor de variáveis de decisão
            
        Returns:
            True se a solução for viável, False caso contrário
        """
        # Exemplo de restrição: soma das variáveis <= 0.5
        return np.sum(x) <= 0.5
    
    def evaluate(self, x):
        """
        Avalia uma solução e retorna os valores dos objetivos.
        
        Args:
            x: Vetor de variáveis de decisão
            
        Returns:
            Vetor de valores dos objetivos
        """
        f = np.zeros(self.n_obj)
        
        # Calcular objetivos
        # ...
        
        # Penalizar soluções inviáveis
        if not self.is_feasible(x):
            # Adicionar penalidade grande a todos os objetivos
            f += 1000.0
        
        return f
```

## Conclusão

A implementação do NSGA-III fornecida neste projeto é flexível e pode ser facilmente adaptada para resolver uma ampla variedade de problemas de otimização multiobjetivo. Seguindo as diretrizes deste guia, você pode integrar seus próprios problemas, incluindo casos de uso reais como a otimização de impedância para interação física humano-robô.

Para problemas mais complexos ou específicos, você pode estender a implementação base ou adaptar problemas de outras bibliotecas, como DEAP ou Platypus.
