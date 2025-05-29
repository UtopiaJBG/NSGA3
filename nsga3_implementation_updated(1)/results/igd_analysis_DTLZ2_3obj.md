
# Análise Detalhada das Discrepâncias no Cálculo do IGD

## Problema: DTLZ2 com 3 objetivos

### Valores de IGD Calculados

| Método de Cálculo | Valor de IGD | Razão vs. Referência |
|-------------------|--------------|----------------------|
| IGD Padrão (normalizado) | 8.288736e-01 | 636.62 |
| IGD Não Normalizado | 8.892360e-01 | 682.98 |
| IGD Normalizado por Objetivo | 7.532775e+00 | 5785.54 |
| IGD com Escala Específica | 8.288736e-01 | 636.62 |
| IGD com Escala Específica (não normalizado) | 8.892360e-01 | 682.98 |
| IGD com Escala Específica (norm. por objetivo) | 7.532775e+00 | 5785.54 |
| IGD com Escala Específica (norm. conjunta) | 8.288736e-01 | 636.62 |
| IGD Dividido por Número de Objetivos | 2.762912e-01 | 212.21 |
| Raiz Quadrada do IGD | 9.104249e-01 | 699.25 |
| Raiz Quadrada do IGD com Escala Específica | 9.104249e-01 | 699.25 |
| Raiz Quadrada do IGD com Escala Específica (norm. conjunta) | 9.104249e-01 | 699.25 |

### Valor de Referência do Artigo Original

| Métrica | Valor |
|---------|-------|
| Melhor (Best) | 1.289000e-03 |
| Mediana (Median) | 1.302000e-03 |
| Pior (Worst) | 2.114000e-03 |

### Estatísticas do Conjunto de Aproximação

| Métrica | Valor |
|---------|-------|
| Número de Pontos | 20 |
| Mínimo por Objetivo | [9.28931241e-20 7.34159656e-17 9.67715031e-01] |
| Máximo por Objetivo | [0.08174344 0.31823716 1.19897467] |
| Média por Objetivo | [0.01524977 0.1447945  1.03628084] |

### Estatísticas do Conjunto de Referência

| Métrica | Valor |
|---------|-------|
| Número de Pontos | 10201 |
| Mínimo por Objetivo | [6.123234e-17 0.000000e+00 0.000000e+00] |
| Máximo por Objetivo | [1. 1. 1.] |
| Média por Objetivo | [0.63525414 0.40354782 0.40354782] |

### Estatísticas das Distâncias Mínimas

| Métrica | Valor |
|---------|-------|
| Mínimo | 1.917745e-02 |
| Máximo | 1.263915e+00 |
| Média | 8.288736e-01 |
| Mediana | 9.009247e-01 |
| Desvio Padrão | 3.158932e-01 |

### Conclusões e Recomendações

1. **Discrepância Significativa**: Existe uma discrepância significativa entre os valores de IGD calculados e os reportados no artigo original, com razões variando de 699.25 a 682.98 vezes.

2. **Melhor Aproximação**: O método que mais se aproxima dos valores do artigo é "IGD Dividido por Número de Objetivos", com uma razão de 212.21.

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
