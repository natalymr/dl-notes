# Билет №2
## Вопрос 1: Activation functions: ReLU, LeakyReLU, PReLU, MaxOut.

[Источник того, что ниже - википедия.](https://ru.wikipedia.org/wiki/%D0%A4%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F_%D0%B0%D0%BA%D1%82%D0%B8%D0%B2%D0%B0%D1%86%D0%B8%D0%B8)

**Функция активации нейрона** определяет выходной сигнал, который определяется входным сигналом или набором входных сигналов. 

В биологических нейронных сетях функция активации обычно является абстракцией, представляющей скорость возбуждения потенциала действия в клетке. В наиболее простой форме эта функция является двоичной — то есть нейрон либо возбуждается, либо нет. 
В этом случае нужно использовать много нейронов для вычислений за пределами линейного разделения категорий.

Прямая с положительным угловым коэффициентом может быть использована для отражения увеличения скорости возбуждения по мере увеличения входного сигнала. Такая функция имела бы вид $\phi (v_{i})=\mu v_{i}$, где $\mu$  — наклон прямой. Эта функция активации линейна, а потому имеет те же проблемы, что и двоичная функция. Кроме того, сети, построенные с использованием таковой модели, имеют нестабильную сходимость, поскольку возбуждение приоритетных входов нейронов стремится к безграничному увеличению, так как эта функция не нормализуема.

Все проблемы, упомянутые выше, можно решить с помощью нормализуемой сигмоидной функции активации. Одна из реалистичных моделей остаётся в нулевом состоянии, пока не придёт входной сигнал, в этот момент скорость возбуждения сначала быстро возрастает, но постепенно достигает асимптоты в $100 \%$ скорости возбуждения. Математически, это выглядит как $\phi (v_{i})=U(v_{i})\mathrm {th} \,(v_{i})$, где ${th} \,(v_{i})$ - гиперболический тангенс можно заменить любой сигмоидой. Такое поведение реально отражается в нейроне, поскольку нейроны не могут физически возбуждаться быстрее некоторой определённой скорости.

Последняя модель, которая используется в многослойных перцептронах — сигмоидная функция активации в форме гиперболического тангенса. Обычно используются два вида этой функции: $\phi (v_{i})=\mathrm {th} \,(v_{i})$, образ которой нормализован к интервалу $[-1, 1]$, и $\phi (v_{i})=(1+\exp(-v_{i}))^{-1}$, сдвинутая по вертикали для нормализации от $0$ до $1$. Последняя модель считается более биологически реалистичной, но имеет теоретические и экспериментальные трудности с вычислительными ошибками некоторых типов.

#### Некоторые желательные свойства функция активации:
* Нелинейность – Если функция активации нелинейна, можно доказать, что двухуровневая нейронная сеть будет универсальным аппроксиматором функции. Тождественная функция активации не удовлетворяет этому свойству. Если несколько уровней используют тождественную функцию активации, вся сеть эквивалентна одноуровневой модели.
* Непрерывная дифференцируемость – Это свойство желательно ($RELU$ не является непрерывно дифференцируемой и имеет некоторые проблемы с оптимизацией, основанной на градиентном спуске, но остаётся допустимой возможностью) для обеспечения методов оптимизации на основе градиентного спуска. Двоичная ступенчатая функция активации не дифференцируема в точке $0$ и её производная равна 0 во всех других точках, так что методы градиентного спуска не дают никакого успеха для неё.
* Область значений – Если множество значений функции активации ограничено, методы обучения на основе градиента более стабильны, поскольку представления эталонов существенно влияют лишь на ограниченный набор весов связей. Если область значений бесконечна, обучение, как правило, более эффективно, поскольку представления эталонов существенно влияют на большинство весов. В последнем случае обычно необходим меньший темп обучения.
* Монотонность – Если функция активации монотонна, поверхность ошибок, ассоциированная с одноуровневой моделью, гарантированно будет выпуклой.
* Гладкие функции с монотонной производной – Показано, что в некоторых случаях они обеспечивают более высокую степень общности.
* Аппроксимирует тождественную функцию около начала координат – Если функции активации имеют это свойство, нейронная сеть будет обучаться эффективно, если её веса инициализированы малыми случайными значениями. Если функция активации не аппроксимирует тождество около начала координат, нужно быть осторожным при инициализации весов. 

### ReLU
Линейный выпрямитель

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_2/relu.png?raw=true">
</p>

### LeakyReLu
Линейный выпрямитель с «утечкой» 

$Leaky ReLU$ allows a small, positive gradient when the unit is not active.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_2/leakyReLU.png?raw=true">
</p>

### PReLU
Параметрический линейный выпрямитель 

$Parametric ReLU\ (PReLU)$ takes $LeakyReLU$ idea further by making the coefficient of leakage into a parameter that is learned along with the other neural network parameters.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_2/prelu.png?raw=true">
</p>

Note that for $a\leq 1$, this is equivalent to $f(x)=\max(x,ax)$ and thus has a relation to "maxout" networks

### MaxOut

$f({\vec {x}})=\max_i x_{i}$

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_2/ReLU-PReLU-and-Maxout-activation-functions.png?raw=true">
</p>


## Вопрос 2: Image Classification Network: AlexNet, VGGnet, GoogleNet.