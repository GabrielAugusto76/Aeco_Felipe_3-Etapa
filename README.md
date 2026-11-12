# Rede Neural do Zero para Classificação do Iris

Implementação de uma Rede Neural Artificial com uma camada escondida, construída **do zero com NumPy**, para classificar as três espécies de flores do conjunto de dados **Iris** (*setosa*, *versicolor* e *virginica*).

O Scikit-Learn é usado **apenas** para carregar e dividir os dados. Toda a rede (propagação, perda, backpropagation e atualização dos pesos) é feita com operações matriciais do NumPy.

## Resultado

Com a semente `42` e 1000 épocas, o modelo atinge cerca de **96,67% de acurácia** no conjunto de teste (29 de 30 amostras corretas).

> Como os pesos iniciais são aleatórios, o valor pode variar um pouco se a semente for alterada. Com apenas 30 amostras de teste, cada erro equivale a cerca de 3,3 pontos percentuais.

### Execução no terminal


*Treinamento de 1000 épocas: a perda (Cross-Entropy) cai de cerca de 1,43 para menos de 0,05, e a acurácia final no conjunto de teste (30 amostras) é de 96,67%.*

## Arquitetura

| Camada | Neurônios | Ativação |
|---|---|---|
| Entrada | 4 (características do Iris) | - |
| Escondida | 8 | ReLU |
| Saída | 3 (uma por classe) | Softmax |

Parâmetros treináveis: `W1` (4x8), `b1` (1x8), `W2` (8x3) e `b2` (1x3).

## Etapas do projeto

1. **Pré-processamento**
   - Carregamento do Iris com `load_iris` (4 características, 3 classes).
   - Divisão em 80% treino e 20% teste, estratificada para manter a proporção das classes.
   - Padronização das características (média 0 e desvio padrão 1), usando a média e o desvio **do conjunto de treino** para evitar vazamento de dados.
   - Rótulos convertidos para One-Hot Encoding (vetores de tamanho 3).

2. **Forward pass**

   ```
   Z1 = X . W1 + b1
   A1 = ReLU(Z1)
   Z2 = A1 . W2 + b2
   A2 = Softmax(Z2)
   ```

3. **Função de custo**: Cross-Entropy Loss

   ```
   L = -média( soma( Y * log(A2) ) )
   ```

4. **Backpropagation** (regra da cadeia)

   ```
   dZ2 = (A2 - Y) / m
   dW2 = A1ᵀ . dZ2
   db2 = soma(dZ2)
   dZ1 = (dZ2 . W2ᵀ) * ReLU'(Z1)
   dW1 = Xᵀ . dZ1
   db1 = soma(dZ1)
   ```

   Para Softmax combinada com Cross-Entropy, o gradiente em relação a `Z2` se simplifica para `A2 - Y`.

5. **Gradiente descendente**: em cada época, os parâmetros são atualizados por `parametro -= taxa_de_aprendizado * gradiente`.

6. **Avaliação**: forward pass no conjunto de teste e cálculo da acurácia final.

## Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Épocas | 1000 |
| Taxa de aprendizado | 0,1 |
| Neurônios na camada escondida | 8 |
| Inicialização dos pesos | Aleatória (He) |
| Inicialização dos vieses | Zeros |
| Semente aleatória | 42 |

## Como executar

### Requisitos

- Python 3.9 ou superior
- NumPy
- Scikit-Learn

### Instalação das dependências

```bash
pip install numpy scikit-learn
```

### Execução

```bash
python iris_neural_network.py
```

O script imprime a perda e a acurácia de treino a cada 100 épocas e, no final, a acurácia no conjunto de teste.

## Estrutura do repositório

```
.
├── iris_neural_network.py   # Código completo da rede neural
└── README.md                # Este arquivo
```

## Estrutura do código

- `preparar_dados()`: carrega, divide, padroniza e codifica os dados.
- `relu()` e `relu_derivada()`: ativação da camada escondida e sua derivada.
- `softmax()`: ativação da camada de saída, numericamente estável.
- `cross_entropy()`: função de custo.
- `RedeNeural`: classe com os métodos `forward`, `backward`, `atualizar`, `treinar`, `prever` e `acuracia`.

## Tecnologias

- Python
- NumPy
- Scikit-Learn (somente para carregamento e preparação dos dados)

- ## Print

- <img width="1377" height="810" alt="image" src="https://github.com/user-attachments/assets/7075b379-1c2d-421a-b759-ceb503d7ce0f" />
<img width="1374" height="808" alt="image" src="https://github.com/user-attachments/assets/88fa973c-366c-494e-9f59-9cea6a377b27" />


