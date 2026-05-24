# NumPy Neural Network

Учебный проект: реализация базовых компонентов нейронной сети **с нуля на NumPy** без использования PyTorch/TensorFlow для обучения.  
Цель проекта: разобраться, как внутри работают слои, функции активации, backpropagation, функции потерь, оптимизаторы.

## Что реализовано

В ноутбуке реализованы основные строительные блоки нейронных сетей:

- базовый класс `Layer`;
- функции активации:
  - `ReLU`;
  - `Sigmoid`;
  - `Tanh`;
- полносвязный слой:
  - `Linear`;
- контейнер модели:
  - `Sequential`;
- регуляризация и нормализация:
  - `Dropout`;
  - `BatchNorm`;
- оптимизатор:
  - `Adam`;
- функции потерь:
  - `CrossEntropyLoss`;
  - `MSELoss`;
- вспомогательные функции:
  - `softmax`;
  - `one_hot_encode`;
- обучение модели на синтетических данных;
- обучение модели на датасете Kaggle Digit Recognizer;

## Архитектура проекта

Основная идея проекта - сделать мини-фреймворк, похожий на упрощённый PyTorch.

Пример модели:

```python
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)
```

Forward pass:

```python
logits = model.forward(X_batch)
```

Backward pass:

```python
grad_loss = criterion.backward()
model.backward(grad_loss)
```

Обновление параметров:

```python
for layer_id, layer in enumerate(model.layers):
    optimizer.update(layer, layer_id)
```

## Результаты

### 1. Синтетический датасет

Для проверки корректности реализации была создана простая бинарная классификация из двух кластеров:

- класс 0 — точки около `(-2, -2)`;
- класс 1 — точки около `(2, 2)`.

Архитектура:

```text
Linear(2 → 16)
ReLU
Linear(16 → 2)
```

Результат обучения за 20 эпох:

```text
loss: 0.6281 → 0.0249
accuracy: 1.0
```

Это показывает, что forward pass, backward pass и обновление весов работают корректно.

### 2. Kaggle Digit Recognizer

Для проверки на реальной задаче использовался датасет Digit Recognizer:

```text
input: 784 признака, пиксели изображения 28x28
output: 10 классов, цифры от 0 до 9
```

Архитектура:

```text
Linear(784 → 128)
ReLU
Linear(128 → 10)
```

Результаты на validation:

```text
train_loss: 0.4152 → 0.0073
val_loss:   0.2243 → 0.0938
val_acc:    0.9323 → 0.9732
```

Модель успешно обучается: loss падает, accuracy растёт.

На test.cvs был получен 0.97125 скор

## Как запустить

### 1. Установить зависимости

```bash
pip install numpy pandas matplotlib
```

### 2. Скачать датасет

Для эксперимента с MNIST нужно скачать данные соревнования Kaggle Digit Recognizer и положить файлы в корень проекта:

```text
train.csv
test.csv
```

### 3. Запустить ноутбук

Откройте ноутбук:

```text
ML_Layers_Implementation.ipynb
```

и последовательно выполните ячейки.

## Что важно в реализации

### Linear

`Linear` реализует аффинное преобразование:

```python
output = x @ weight + bias
```

Также в backward считаются:

```python
grad_input = grad_output @ weight.T
grad_weight = input.T @ grad_output
grad_bias = np.sum(grad_output, axis=0)
```

### ReLU

`ReLU` зануляет отрицательные значения:

```python
output = np.maximum(0, x)
```

В backward градиент проходит только там, где вход был положительным.

### CrossEntropyLoss

`CrossEntropyLoss` использует softmax и отрицательный логарифм вероятности правильного класса.

Градиент по logits:

```python
grad = softmax_pred.copy()
grad[np.arange(batch_size), targets] -= 1
grad /= batch_size
```

### Adam

Adam хранит для каждого параметра:

- `m` — экспоненциальное скользящее среднее градиента;
- `v` — экспоненциальное скользящее среднее квадрата градиента.

Обновление:

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad ** 2

m_corrected = m / (1 - beta1 ** t)
v_corrected = v / (1 - beta2 ** t)

param = param - learning_rate * m_corrected / (np.sqrt(v_corrected) + eps)
```

## Стек

- Python
- NumPy
- pandas
- matplotlib

## Цель проекта

Главная цель проекта не получить максимальное качество на MNIST, а понять внутреннюю механику нейронных сетей:

```text
forward pass -> loss -> backward pass -> gradients -> optimizer step
```

Проект показывает, что даже на чистом NumPy можно реализовать базовую нейронную сеть, и получить хорошее качество на задаче классификации рукописных цифр.
