
# Análise Detalhada das Discrepâncias no Cálculo do IGD

## Problema: DTLZ1 com 3 objetivos

### Valores de IGD Calculados

| Método de Cálculo | Valor de IGD | Razão vs. Referência |
|-------------------|--------------|----------------------|
| IGD Padrão (normalizado) | 5.970337e-01 | 456.45 |
| IGD Não Normalizado | 1.298004e+00 | 992.36 |
| IGD Normalizado por Objetivo | 7.489020e-01 | 572.56 |
| IGD com Escala Específica | 5.970337e-01 | 456.45 |
| IGD com Escala Específica (não normalizado) | 2.596008e+00 | 1984.72 |
| IGD com Escala Específica (norm. por objetivo) | 7.489020e-01 | 572.56 |
| IGD com Escala Específica (norm. conjunta) | 5.970337e-01 | 456.45 |
| IGD Dividido por Número de Objetivos | 1.990112e-01 | 152.15 |
| Raiz Quadrada do IGD | 7.726796e-01 | 590.73 |
| Raiz Quadrada do IGD com Escala Específica | 7.726796e-01 | 590.73 |
| Raiz Quadrada do IGD com Escala Específica (norm. conjunta) | 7.726796e-01 | 590.73 |

### Valor de Referência do Artigo Original

| Métrica | Valor |
|---------|-------|
| Melhor (Best) | 4.880000e-04 |
| Mediana (Median) | 1.308000e-03 |
| Pior (Worst) | 4.880000e-03 |

### Estatísticas do Conjunto de Aproximação

| Métrica | Valor |
|---------|-------|
| Número de Pontos | 20 |
| Mínimo por Objetivo | [0.         0.         1.39331309] |
| Máximo por Objetivo | [0.52920301 0.60177489 3.65084831] |
| Média por Objetivo | [0.13501921 0.27418238 2.77317787] |

### Estatísticas do Conjunto de Referência

| Métrica | Valor |
|---------|-------|
| Número de Pontos | 10201 |
| Mínimo por Objetivo | [ 0.00000000e+00  0.00000000e+00 -5.55111512e-17] |
| Máximo por Objetivo | [0.5 0.5 0.5] |
| Média por Objetivo | [0.25  0.125 0.125] |

### Estatísticas das Distâncias Mínimas

| Métrica | Valor |
|---------|-------|
| Mínimo | 2.951500e-01 |
| Máximo | 9.818361e-01 |
| Média | 5.970337e-01 |
| Mediana | 5.587265e-01 |
| Desvio Padrão | 1.964920e-01 |

### Conclusões e Recomendações

1. **Discrepância Significativa**: Existe uma discrepância significativa entre os valores de IGD calculados e os reportados no artigo original, com razões variando de 590.73 a 992.36 vezes.

2. **Melhor Aproximação**: O método que mais se aproxima dos valores do artigo é "IGD Dividido por Número de Objetivos", com uma razão de 152.15.

3. **Possíveis Causas**:
   - Diferenças na geração dos pontos de referência na fronteira Pareto-ótima
   - Diferenças na normalização dos objetivos
   - Diferenças na implementação dos problemas DTLZ
   - Diferenças nos parâmetros do algoritmo NSGA-III
   - Diferenças no número de execuções e na seleção dos resultados reportados

4. **Recomendações**:
   - Revisar a implementação dos problemas DTLZ para garantir que estão de acordo com as definições originais
   - Verificar os parâmetros do algoritmo NSGA-III, especialmente o tamanho da população e o número de gerações
   - Considerar a possibilidade de que o artigo original possa ter reportado valores selecionados ou processados de forma diferente
   - Para fins de comparação relativa entre algoritmos, usar a mesma implementação de IGD para todos
