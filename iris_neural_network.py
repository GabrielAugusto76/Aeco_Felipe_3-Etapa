import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Reprodutibilidade
SEED = 42
EPOCHS = 1000
LEARNING_RATE = 0.1
HIDDEN_UNITS = 8


def softmax(logits):
    """Softmax numericamente estável, aplicada linha a linha."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    """Entropia cruzada média para rótulos one-hot."""
    eps = 1e-12
    probabilities = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(np.sum(y_true * np.log(probabilities), axis=1))


def main():
    # Carregamento e preparação dos dados com Scikit-Learn.
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target
    num_classes = len(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train_one_hot = np.eye(num_classes, dtype=np.float64)[y_train]

    # Arquitetura: 4 entradas -> 8 neurônios ReLU -> 3 saídas Softmax.
    rng = np.random.default_rng(SEED)
    input_units = X_train.shape[1]
    output_units = num_classes
    W1 = rng.normal(0.0, np.sqrt(2.0 / input_units), (input_units, HIDDEN_UNITS))
    b1 = np.zeros((1, HIDDEN_UNITS), dtype=np.float64)
    W2 = rng.normal(0.0, np.sqrt(2.0 / HIDDEN_UNITS), (HIDDEN_UNITS, output_units))
    b2 = np.zeros((1, output_units), dtype=np.float64)

    # Treinamento em lote completo com descida do gradiente e backpropagation.
    sample_count = X_train.shape[0]
    for epoch in range(1, EPOCHS + 1):
        z1 = X_train @ W1 + b1
        a1 = np.maximum(0.0, z1)  # ReLU
        z2 = a1 @ W2 + b2
        probabilities = softmax(z2)
        loss = cross_entropy(y_train_one_hot, probabilities)

        # Para Softmax + entropia cruzada, dL/dz2 = (probabilidades - alvo) / N.
        dz2 = (probabilities - y_train_one_hot) / sample_count
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0.0)  # Derivada da ReLU
        dW1 = X_train.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W1 -= LEARNING_RATE * dW1
        b1 -= LEARNING_RATE * db1
        W2 -= LEARNING_RATE * dW2
        b2 -= LEARNING_RATE * db2

        if epoch == 1 or epoch % 100 == 0:
            print(f"Época {epoch:4d}/{EPOCHS} | perda: {loss:.4f}")

    # Avaliação no conjunto de teste.
    hidden_test = np.maximum(0.0, X_test @ W1 + b1)
    test_probabilities = softmax(hidden_test @ W2 + b2)
    predictions = np.argmax(test_probabilities, axis=1)
    accuracy = np.mean(predictions == y_test)

    print(f"\nAcurácia no teste: {accuracy * 100:.2f}%")
    print(f"Amostras de teste: {len(y_test)}")


if __name__ == "__main__":
    main()
