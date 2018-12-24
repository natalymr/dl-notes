# Билет №4
## Вопрос 1: Convolutions. Causal convolutions. Dilated convolutions. Max pooling. Average pooling. Padding.

### Convolutions

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_4/conv0.png?raw=true">
</p>

Надо ввести базовые понятия, чтобы потом мы понимали друг друга. Окошко, которое ходит по большой матрице называется $filter$. Фильтр накладывается на участок большой матрицы и каждое значение перемножается с соответствующим ему значением фильтра (красные цифры ниже и правее черных цифр основной матрицы). Потом все получившееся складывается и получается выходное (“отфильтрованное”) значение. 

Окно ходит по большой матрице с каким-то шагом, который по-английски называется $stride$. Этот шаг бывает горизонтальный и вертикальный. 

Размер итогового изображения можно найти по следующей формулe:
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