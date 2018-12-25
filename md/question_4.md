# Билет №4
## Вопрос 1: Convolutions. Causal convolutions. Dilated convolutions. Max pooling. Average pooling. Padding.

### Convolutions

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/conv0.png?raw=true">
</p>

Надо ввести базовые понятия, чтобы потом мы понимали друг друга. Окошко, которое ходит по большой матрице называется $filter$. Фильтр накладывается на участок большой матрицы и каждое значение перемножается с соответствующим ему значением фильтра (красные цифры ниже и правее черных цифр основной матрицы). Потом все получившееся складывается и получается выходное (“отфильтрованное”) значение. 

Окно ходит по большой матрице с каким-то шагом, который по-английски называется $stride$. Этот шаг бывает горизонтальный и вертикальный. 

**Размер итогового изображения можно найти по следующей формулe:**
$$n_{out} = \Big\lfloor\frac{n_{in} - f + 2p}{s}\Big\rfloor + 1,$$
где
$n_{in}$ - высота или ширина входного изображения, 
$n_{out}$ - высота или ширина выходного изображения,
$f$ - размер фильтра, например,$3$,
$p$ - размер паддинга, например, $2$,
$s$ - размер шага, страйда, например, $1$.


<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/conv.png?raw=true">
</p>

Cвертка $1 * 1$ — это специальный вид свертки, который интегрирует все каналы в одно значение, оставляя размер матрицы неизменным.

### Causal convolutions
The distinguishing characteristics of $Temporal\ Convolutional\ Networks - TCN$ are: 
1. the convolutions in the architecture are causal, meaning that there is no information “leakage” from future to past;
2. the architecture can take a sequence of any length and map it to an output sequence of the same length, just as with an $RNN$.



<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/causal_conv.png?raw=true">
</p>


### Dilated convolutions

[Источник того, что ниже.](http://mit.spbau.ru/sewiki/images/e/ed/Kurbanov_diploma_master2017.pdf)
Дырявые свёртки это свёртки, в которых фильтр применяется по диапазону больше своей длины, пропуская входные значения с некоторым шагом. Это эквивалентно свёртке с большим фильтром, ”продырявленным” нулями, но значительно эффективнее вычислительно. Такая эффективность дырявых свёрток позволяет нейросети оперировать более крупными данными, нежели позволили бы обычные свёртки. В частном случае дырявые свёртки с пропуском $1$ эквиваленты обычным свёрткам. Ниже изображены дырявые свёртки с промежутками $1, 2, 4$ и $8.$

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/dilated_conv.png?raw=true">
</p>

Попробуем объяснить интуицию за такой конфигурацией. Во-первых, экспоненциальное увеличение промежутков приводит к экспоненциальному по глубине росту окна. Например, каждый $1, 2, 4, ... 512$ блок имеет окно размера $1024$ его можно воспринимать в качестве более эффективной и имеющей большую описательную силу альтернативы свёртки $1 × 1024$. Во-вторых, дальнейшее объединение таких слоёв в стеки увеличивает объём модели и ширину окна.

Можно посмотреть на расширенную свертку под другим углом, например,
<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/dilated_conv1.png?raw=true">
</p>

### Max and average pooling

Все виды пулинга можно использовать для уменьшения сложности вычислений.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/pooling.png?raw=true">
</p>

Для чего можно использовать
* Max pooling
    * извлекает наиболее важные фичи
* Average pooling
    * извлекает смазанные, усредненные фичи
    * если вам нужна информация о всех входных данных

В целом, $max\ pooling$ используется чаще. $Average\ pooling$ успешно применялся в задачах классификации.

### Padding
<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/padding.png?raw=true">
</p>

Зачем использовать $padding$? [Источник ответа на вопрос.](https://stats.stackexchange.com/questions/246512/convolutional-layers-to-pad-or-not-to-pad)
1. It's easier to design networks if we preserve the height and width and don't have to worry too much about tensor dimensions when going from one layer to another because dimensions will just "work".
2. It allows us to design deeper networks. Without padding, reduction in volume size would reduce too quickly.
3. Padding actually improves performance by keeping information at the borders.
    > "In addition to the aforementioned benefit of keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be “washed away” too quickly."
4. New network architectures need to concatenate convolutional layers with $1x1$, $3x3$ and $5x5$ filters and it wouldn't be possible if they didn't use padding because dimensions wouldn't match. 

## Вопрос 2: Reccurent Neural Networks. LSTM. GRU.

### RNN 
[Источник того, что ниже - википедия.](https://ru.wikipedia.org/wiki/%D0%A0%D0%B5%D0%BA%D1%83%D1%80%D1%80%D0%B5%D0%BD%D1%82%D0%BD%D0%B0%D1%8F_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C)
**Рекуррентные нейронные сети** — вид нейронных сетей, где связи между элементами образуют направленную последовательность. Благодаря этому появляется возможность обрабатывать серии событий во времени или последовательные пространственные цепочки. В отличие от многослойных перцептронов, рекуррентные сети могут использовать свою внутреннюю память для обработки последовательностей произвольной длины. 

[Источник того, что ниже и на русском - эта **очень** крутая статья на хабре.](https://habr.com/company/wunderfund/blog/331310/)
Обратные связи придают рекуррентным нейронным сетям некую загадочность. Тем не менее, если подумать, они не так уж сильно отличаются от обычных нейронных сетей. Рекуррентную сеть можно рассматривать, как несколько копий одной и той же сети, каждая из которых передает информацию последующей копии. Вот, что произойдет, если мы развернем обратную связь:

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/rnn.png?raw=true">
</p>

Одна из привлекательных идей RNN состоит в том, что они потенциально умеют связывать предыдущую информацию с текущей задачей, так, например, знания о предыдущем кадре видео могут помочь в понимании текущего кадра. Если бы RNN обладали такой способностью, они были бы чрезвычайно полезны. Но действительно ли RNN предоставляют нам такую возможность? Это зависит от некоторых обстоятельств.

К сожалению, по мере роста этого расстояния, RNN теряют способность связывать информацию.

### LSTM

Долгая краткосрочная память (Long short-term memory; LSTM) – особая разновидность архитектуры рекуррентных нейронных сетей, способная к обучению долговременным зависимостям. 

LSTM разработаны специально, чтобы избежать проблемы долговременной зависимости. Запоминание информации на долгие периоды времени – это их обычное поведение, а не что-то, чему они с трудом пытаются обучиться.

Любая рекуррентная нейронная сеть имеет форму цепочки повторяющихся модулей нейронной сети. В обычной RNN структура одного такого модуля очень проста, например, он может представлять собой один слой с функцией активации tanh (гиперболический тангенс).

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/rnn1.png?raw=true">
</p>

Структура LSTM также напоминает цепочку, но модули выглядят иначе. Вместо одного слоя нейронной сети они содержат целых четыре, и эти слои взаимодействуют особенным образом.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/lstm1.png?raw=true">
</p>

Не будем пока озадачиваться подробностями. Рассмотрим каждый шаг схемы LSTM позже. Пока познакомимся со специальными обозначениями, которыми мы будем пользоваться.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/plot_history.png?raw=true">
</p>

На схеме выше каждая линия переносит целый вектор от выхода одного узла ко входу другого. Розовыми кружочками обозначены поточечные операции, такие, как сложение векторов, а желтые прямоугольники – это обученные слои нейронной сети. Сливающиеся линии означают объединение, а разветвляющиеся стрелки говорят о том, что данные копируются и копии уходят в разные компоненты сети.


#### Основная идея LSTM

Ключевой компонент LSTM – это состояние ячейки (cell state) – горизонтальная линия, проходящая по верхней части схемы.

Состояние ячейки напоминает конвейерную ленту. Она проходит напрямую через всю цепочку, участвуя лишь в нескольких линейных преобразованиях. Информация может легко течь по ней, не подвергаясь изменениям.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/cell_state.png?raw=true">
</p>

Тем не менее, LSTM может удалять информацию из состояния ячейки; этот процесс регулируется структурами, называемыми фильтрами (gates).

Фильтры позволяют пропускать информацию на основании некоторых условий. Они состоят из слоя сигмоидальной нейронной сети и операции поточечного умножения.

Сигмоидальный слой возвращает числа от нуля до единицы, которые обозначают, какую долю каждого блока информации следует пропустить дальше по сети. Ноль в данном случае означает “не пропускать ничего”, единица – “пропустить все”.

В LSTM три таких фильтра, позволяющих защищать и контролировать состояние ячейки.

#### Пошаговый разбор LSTM
1. Первый шаг в LSTM – определить, какую информацию можно выбросить из состояния ячейки. Это решение принимает сигмоидальный слой, называемый “слоем фильтра забывания” (forget gate layer). Он смотрит на $h_{t - 1}$ и $x$ и возвращает число от $0$ до $1$ для каждого числа из состояния ячейки.

    $1$ -- “полностью сохранить”,
    $0$ -- “полностью выбросить”.

    Вернемся к нашему примеру – языковой модели, предсказывающей следующее слово на основании всех предыдущих. В этом случае состояние ячейки должно сохранить существительного, чтобы затем использовать местоимения соответствующего рода. Когда мы видим новое существительное, мы можем забыть род старого.

    <p align="center">
      <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/forget.png?raw=true">
    </p>
    
2. Следующий шаг – решить, какая новая информация будет храниться в состоянии ячейки. Этот этап состоит из двух частей. Сначала сигмоидальный слой под названием “слой входного фильтра” (input layer gate) определяет, какие значения следует обновить. Затем tanh-слой строит вектор новых значений-кандидатов $\hat{C}_t$, которые можно добавить в состояние ячейки.

    В нашем примере с языковой моделью на этом шаге мы хотим добавить род нового существительного, заменив при этом старый.
    
    <p align="center">
      <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/c.png?raw=true">
    </p>
    
3. Настало время заменить старое состояние ячейки $C_{t - 1}$ на новое состояние $C_t$. Что нам нужно делать — мы уже решили на предыдущих шагах, остается только выполнить это.

    Мы умножаем старое состояние на $f_t$, забывая то, что мы решили забыть. Затем прибавляем $i_t * \hat{C}_t$. Это новые значения-кандидаты, умноженные на $t$ – на сколько мы хотим обновить каждое из значений состояния.

    В случае нашей языковой модели это тот момент, когда мы выбрасываем информацию о роде старого существительного и добавляем новую информацию.
    
    <p align="center">
      <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/c_t.png?raw=true">
    </p>
    
4. Наконец, нужно решить, какую информацию мы хотим получать на выходе. Выходные данные будут основаны на нашем состоянии ячейки, к ним будут применены некоторые фильтры. Сначала мы применяем сигмоидальный слой, который решает, какую информацию из состояния ячейки мы будем выводить. Затем значения состояния ячейки проходят через tanh-слой, чтобы получить на выходе значения из диапазона от $-1$ до $1$, и перемножаются с выходными значениями сигмоидального слоя, что позволяет выводить только требуемую информацию.

    Мы, возможно, захотим, чтобы наша языковая модель, обнаружив существительное, выводила информацию, важную для идущего после него глагола. Например, она может выводить, находится существительное в единственном или множественном числе, чтобы правильно определить форму последующего глагола.

    <p align="center">
      <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/last.png?raw=true">
    </p>

#### Еще раз архитектура LSTM:

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/lstm.png?raw=true">
</p>

В самом начале, в момент времени $t = 0:$
$c_{0}=0$ и $h_{0}=0$

В следующие моменты времени:

\begin{aligned}
f_{t}&=\sigma _{g}(W_{f}x_{t}+U_{f}h_{t-1}+b_{f}),\\
i_{t}&=\sigma _{g}(W_{i}x_{t}+U_{i}h_{t-1}+b_{i}),\\
o_{t}&=\sigma _{g}(W_{o}x_{t}+U_{o}h_{t-1}+b_{o}),\\
c_{t}&=f_{t}\circ c_{t-1}+i_{t}\circ \sigma _{c}(W_{c}x_{t}+U_{c}h_{t-1}+b_{c}),\\
h_{t}&=o_{t}\circ \sigma _{h}(c_{t}),
\end{aligned}

где 
* оператор $\circ$ - это произведение Адамара, которое можно найти по следующей схеме:
    <p align="center">
      <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/Hadamard_product.png?raw=true">
    </p>
* $x_{t}\in \mathbb {R} ^{d}$ -- вектор на входе,
* $f_{t}\in \mathbb {R} ^{h}$ -- forget gate's activation vector,
* $i_{t}\in \mathbb {R} ^{h}$ -- input gate's activation vector,
* $o_{t}\in \mathbb {R} ^{h}$ -- output gate's activation vector,
* $h_{t}\in \mathbb {R} ^{h}$ -- hidden state vector also known as output vector of the LSTM unit,
* $c_{t}\in \mathbb {R} ^{h}$ -- cell state vector,
* $W\in \mathbb {R} ^{h\times d}$, $U\in \mathbb {R} ^{h\times h}$ and $b\in \mathbb {R} ^{h}$ -- weight matrices and bias vector parameters which need to be learned during training.

### GRU
[Источник того, что ниже - википедия.](https://ru.wikipedia.org/wiki/%D0%A3%D0%BF%D1%80%D0%B0%D0%B2%D0%BB%D1%8F%D0%B5%D0%BC%D1%8B%D0%B9_%D1%80%D0%B5%D0%BA%D1%83%D1%80%D1%80%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D0%B9_%D0%B1%D0%BB%D0%BE%D0%BA)
**Управляемые рекуррентные блоки** (Gated Recurrent Units, GRU) — механизм вентилей для рекуррентных нейронных сетей, представленный в 2014 году. Было установлено, что его эффективность при решении задач моделирования музыкальных и речевых сигналов сопоставима с использованием долгой краткосрочной памяти (LSTM). По сравнению с LSTM у данного механизма меньше параметров, т.к. отсутствует выходной вентиль.

На схеме ниже представлена архитектура RGU.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/gru.png?raw=true">
</p>

В самом начале для $t = 0$, выходной вектор получается $h_{0}=0$.

$$\begin{aligned}
z_{t}&=\sigma _{g}(W_{z}x_{t}+U_{z}h_{t-1}+b_{z}),\\
r_{t}&=\sigma _{g}(W_{r}x_{t}+U_{r}h_{t-1}+b_{r}),\\
h_{t}&=z_{t}\circ h_{t-1}+(1-z_{t})\circ \sigma _{h}(W_{h}x_{t}+U_{h}(r_{t}\circ h_{t-1})+b_{h}),
\end{aligned}$$ 

где
* $x_{t}$ -- вектор на входе,
* $h_{t}:$ -- вектор на выходе,
* $z_{t}:$ -- update gate vector, 
* $r_{t}:$ -- reset gate vector,
* $W$, $U$ and  $b$: матрицы параметров и bias-вектор, которые и обучаются.

Функции активации:
* $\sigma$ -- сигмоид,
* $tanh$ -- гиперболический тангенс.
