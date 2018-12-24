# Билет №8
## Вопрос 1: Regularization: L2, Early Stopping, Dropout, Dropconnect, Batch Normalization.

[Источник того, что ниже - википедия](https://ru.wikipedia.org/wiki/%D0%A0%D0%B5%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F_(%D0%BC%D0%B0%D1%82%D0%B5%D0%BC%D0%B0%D1%82%D0%B8%D0%BA%D0%B0)) [и **это**](http://ru.learnmachinelearning.wikia.com/wiki/%D0%A0%D0%B5%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F)

### Регуляризация
**Регуляризация в машинном обучении** — метод добавления некоторой дополнительной информации к условию с целью решить некорректно поставленную задачу или предотвратить переобучение.
Эта информация часто имеет вид штрафа за сложность модели. Например, это могут быть ограничения гладкости результирующей функции или ограничения по норме векторного пространства.

Часто регуляризация представляет собой просто некоторую добавку $R(w)$ (где $w$ — параметры модели) к функции потерь $L(f(x,w),y)$ так что задача приобретает вид:

$$\min_{w} \sum_{i = 1}^N(f(x_i,w),y_i))+ \lambda R(w)$$

Часто используемые $R(w)$ являются $L_1,\ L_2,\ ElasticNet$

### L1, L2, ElasticNet
* $L_{1}$ -- регуляризация или $lasso\ regression$:
$$R(w)=||w||_1 = \sum_{i = 1}^d|w_i|$$
* $L_{2}$ -- регуляризация или $ridge\ regression$
$$R(w)=||w||_2^2 = \sum_{i = 1}^dw_i^2$$
* $ElasticNet$
$$R(w) = \lambda_1 ||w||_1 + \lambda_2 ||w||_2^2 $$

_Простыми словами: Переобучение в большинстве случаев проявляется в том, что в получающихся многочленах слишком большие коэффициенты. Соответственно, и бороться с этим можно довольно естественным способом: **нужно просто добавить в целевую функцию штраф, который бы наказывал модель за слишком большие коэффициенты**._


### Early Stopping
[Источник того, что ниже.](https://habr.com/company/wunderfund/blog/315476/)

Возможно, самый простой способ использования валидационного множества — настройка количества эпох с помощью процедуры, известной как ранняя остановка — просто остановите процесс обучения, если за заданное количество эпох (параметр $patience$) потери на тестовом множестве не начинают уменьшаться. 

[Английская википедия про раннюю остановку пишет следующее:](https://en.wikipedia.org/wiki/Early_stopping)

These early stopping rules work by splitting the original training set into a new training set and a validation set. **The error on the validation set is used as a proxy for the generalization error in determining when overfitting has begun**. These methods are most commonly employed in the training of neural networks. 
Prechelt gives the following summary of a naive implementation of holdout-based early stopping as follows:

1. Split the training data into a training set and a validation set, e.g. in a 2-to-1 proportion.
2. Train only on the training set and evaluate the per-example error on the validation set once in a while, e.g. after every fifth epoch.
3. **Stop training as soon as the error on the validation set is higher than it was the last time it was checked.**
4. Use the weights the network had in that previous step as the result of the training run.
    _— Lutz Prechelt, Early Stopping – But When?_

> More sophisticated forms use cross-validation – multiple partitions of the data into training set and validation set – instead of a single partition into a training set and validation set. Even this simple procedure is complicated in practice by the fact that the validation error may fluctuate during training, producing multiple local minima. This complication has led to the creation of many ad-hoc rules for deciding when overfitting has truly begun.
 
### Dropout




**Overfitting** - одна из проблем глубоких нейронных сетей, состоящая в следующем: модель хорошо объясняет _только_ примеры из обучающей выборки, адаптируясь к обучающим примерам, вместо того чтобы учиться классифицировать примеры, не участвовавшие в обучении (теряя способность к обобщению).

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_8/dropout.png?raw=true">
</p>

Сети для обучения получаются с помощью исключения из сети (dropping out) нейронов с вероятностью $p$, таким образом, вероятность того, что нейрон останется в сети, составляет $q = 1 - p$. “Исключение” нейрона означает, что при любых входных данных или параметрах он возвращает $0$.

Исключенные нейроны не вносят свой вклад в процесс обучения ни на одном из этапов алгоритма обратного распространения ошибки ($backpropagation$); поэтому исключение хотя бы одного из нейронов равносильно обучению новой нейронной сети.

### Dropconnection

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_8/dropconnect.png?raw=true">
</p>

### Batch Normalization

[Источник того, что ниже - хабр.](https://habr.com/post/309302/)

Рассмотрим классическую нейронную сеть с несколькими слоями. Каждый слой имеет множество входов и множество выходов. Сеть обучается методом обратного распространения ошибки, по батчам, то есть ошибка считается по какому-то подмножестве обучающей выборки.


Стандартный способ нормировки — для каждого $k$ рассмотрим распределение элементов батча. Вычтем среднее и поделим на дисперсию выборки, получив распределение с центром в $0$, $a = 0$, и дисперсией $\sigma^2 = 1$. **Такое распределение позволит сети быстрее обучатся, т.к. все числа получатся одного порядка**. Но ещё лучше ввести две переменные для каждого признака, обобщив нормализацию следующим образом:

$$\hat{x}_k = \frac{x_k - E[x_k]}{\sqrt{Var[x_k]}},$$
где статистики $E[x_k],\ Var[x_k]$ посчитаны по текущему батчу.

[Интересная лекция для почитать на потом про batch normalization.](https://logic.pdmi.ras.ru/~sergey/teaching/dl2017/DLNikolenko-Beeline-03.pdf)

## Вопрос 2: Adversarial attacks: White box, Black box, Targeted, Untargeted.