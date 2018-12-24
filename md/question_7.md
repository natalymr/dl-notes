# Билет №7
## Вопрос 1: Word embeddings: Co-occurrence Matrix, Word2Vec, CBOW, Skip-Gram, GloVE, FastText.

### Word embedding
[Источник того, что ниже - википедия.](https://ru.wikipedia.org/wiki/%D0%92%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BD%D0%BE%D0%B5_%D0%BF%D1%80%D0%B5%D0%B4%D1%81%D1%82%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5_%D1%81%D0%BB%D0%BE%D0%B2)
**Векторное представление** — общее название для различных подходов к моделированию языка и обучению представлений в обработке естественного языка, направленных на сопоставление словам (и, возможно, фразам) из некоторого словаря векторов из $R^{n}$ для $n$, значительно меньшего количества слов в словаре. Теоретической базой для векторных представлений является дистрибутивная семантика.
> **Дистрибутивна семантика** — это область лингвистики, которая занимается вычислением степени есмантической близости между лингвистическими единицами на основании их распределения (дистрибуции) в больших массивах лингвистических данных.
>
> Каждому слову присваивается свой контекстный вектор. Множество векторов формирует словесное векторное пространство.
> 
> Семантическое расстояние между понятиями, выраженными словами естественного языка, обычно вычисляется как косинусное расстояние между векторами словесного пространства.


[Источник того, что ниже.](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html)
There are two main approaches for learning word embedding, both relying on the contextual knowledge.

* **Count-based**: The first one is unsupervised, based on matrix factorization of a global word co-occurrence matrix. Raw co-occurrence counts do not work well, so we want to do smart things on top.
    * Co-occurence matrix
    * glove
* **Context-based**: The second approach is supervised. Given a local context, we want to design a model to predict the target words and in the meantime, this model learns the efficient word embedding representation.
    * word2vec
    * cbow
    * skip-gram
    * glove
    * fasttext

### Co-occurence matrix
[Источник того, что ниже.](https://medium.com/data-science-group-iitr/word-embedding-2d05d270b285)
Words co-occurrence matrix describes how words occur together that in turn captures the relationships between words. Words co-occurrence matrix is computed simply by counting how two or more words occur together in a given corpus. As an example of words co-occurrence, consider a corpus consisting of the following documents:
$$penny\ wise\ and\ pound\ foolish. \\
a\ penny\ saved\ is\ a\ penny\ earned.$$
Letting $count(w(next)|w(current))$ represent how many times word $w(next)$ follows the word $w(current)$, we can summarize co-occurrence statistics for words “a” and “penny” as:

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_7/co_oc_matrix_table.png?raw=true">
</p>

Now, a column can also be understood as word vector for the corresponding word in the matrix $M$. For example, the word vector for ‘cat’ in the above matrix is $[1,1]$ and so on.Here, the rows correspond to the documents in the corpus and the columns correspond to the tokens in the dictionary. The second row in the above matrix may be read as — Document $2$ contains ‘hat’: once, ‘dog’: once and ‘the’ thrice and so on.

Дальше, получив матрицу совместной встречаемости, можно уменьшить ее размер, применив [PCA.](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B3%D0%BB%D0%B0%D0%B2%D0%BD%D1%8B%D1%85_%D0%BA%D0%BE%D0%BC%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%82)
> **Вычисление главных компонент, PCA**, один из основных способов уменьшить размерность данных, потеряв наименьшее количество информации. Этот способ может быть сведен к вычислению сингулярного разложения матрицы данных или к вычислению собственных векторов и собственных значений ковариационной матрицы исходных данных.

### word2vec
[Источник того, что ниже - лекции.]()
Предположим, что у нас есть языковая модель, то есть набор вероятностей: пусть у нас есть набор слов, мы по нему проходимся, и в какой-то момент стоим на слове  $w_i$, мы хотим предсказать по словам  $w_{i - k}, ..., w_i$, какое слово будет стоять следующим и для каждого слова из словаря считаем его вероятность быть $w_{i + 1}$. 
Основное отличие заключается в том, что _теперь рассматриваем отношение не одного слова к другому, а **группы слов к одному**_.

Наша задача найти такую функцию $f$, которая 
1. от слов, которые часто встречаются вместе, давала бы что-то большое,
2. от слов, которые встречаются рядом редко, давала бы что-то маленькое.

$$P(w_{candidate} | w_{context}) = \frac{\exp(f(w_{candidate}, w_{context}))}{\sum_{w_i \in Dictionary} \exp(f(w_i, w_{context}))}$$

Самый простой, неплохо работающий вариант 
$$f(w_{candidate}, w_{context}) = <u_{w_{candidate}}, v_{w_{context}}>,$$
то есть $f$ - скалярное произведение векторов.

В итоге, хотим максимизировать следующую функцию (логарифм от softmax):
$$J(\theta)=−\frac{1}{T} \sum_{t = 1}^T \sum_{-m \le j \le m,\ j \ne 0} \log(p(w_{t + j} | w_t)),$$
где
$T$ -- размер словаря;
$w_t$ -- текущее слово и мы пытаемся найти вероятность, что другое слово, $w_{t + j}$ появится с текущим в одном окне, то есть среди $\pm m$ слов вокруг;
$\theta$ -- векторные представления слов, которые мы ищем.

### CBOW, Skip-gram

[Источник того, что ниже.](https://towardsdatascience.com/beyond-word-embeddings-part-2-word-vectors-nlp-modeling-from-bow-to-bert-4ebd4711d0ec)
There are two main Word2Vec architectures that are used to produce a distributed representation of words:
* **Continuous bag-of-words (CBOW)**  — The order of context words does not influence prediction (bag-of-words assumption). In the continuous skip-gram architecture, the model uses the current word to predict the surrounding window of context words.
* **Continuous skip-gram** weighs nearby context words more heavily than more distant context words. While order still is not captured each of the context vectors are weighed and compared independently vs CBOW which weighs against the average context.

CBOW is faster while skip-gram is slower but does a better job for infrequent words.

Еще раз, но на русском, [скопировано отсюда](https://ru.wikipedia.org/wiki/Word2vec):
В word2vec существуют два основных алгоритма обучения: CBOW и Skip-gram:
* **Skip-gram** использует текущее слово, чтобы предугадывать окружающие его слова.
    <p align="center">
      <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_7/skip_gram.png?raw=true">
    </p>
* **CBOW** — «непрерывный мешок со словами» модельная архитектура, которая предсказывает текущее слово, исходя из окружающего его контекста. 
    <p align="center">
      <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_7/cbow.png?raw=true">
    </p>

_Порядок слов контекста не оказывает влияния на результат ни в одном из этих алгоритмов._

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_7/cbow_skip_gram.png?raw=true">
</p>

### GloVe

[Источник того, что ниже.](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html#glove-global-vectors)
**The Global Vector (GloVe)** model proposed by Pennington et al. (2014) aims _to combine the count-based matrix factorization and the context-based skip-gram model together_.

We all know the counts and co-occurrences can reveal the meanings of words. To distinguish from $p(w_O|w_I)$ in the context of a word embedding word, we would like to define the co-ocurrence probability as:

$$p_{co}(w_k|w_i)=\frac{C(w_i,w_k)}{C(w_i)}$$

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_7/glove.png?raw=true">
</p>

Получив матрицу совместной встречаемости, $X$, необходимо проделать следующие шаги:
<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_7/glove1.png?raw=true">
</p>

[Источник того, что ниже - лекции.](https://docs.google.com/document/d/1BnfFRAGzioCf1b8n1MQbIb54m0t5moQPUAlYQ5xkOzA/edit)
Здесь снова появляется матрица совместной встречаемости, и от нее берем логарифм и умножаем на функцию, которая ограничивает совместную встречаемость. То есть это сделано, чтобы слова, которые встречаются очень часто, не давали очень большую ошибку.
GloVE хорош тем, что  есть в общем доступе вектора, натренированные на большом количестве датасетов.

### FastText
[Источник того, что ниже - лекции.]()
Сейчас используют алгоритм FastText, суть которого состоит в следующем:
    вместе одного слова добавим в словарь ещё и все его n-граммы. Ниже кроме слова where мы добавим в словарь его три-граммы.
    $$where \rightarrow\ \ <where>\ +\ <wh + whe + her + ere + re>$$
    
Зачем это нужно?
Если не использовать стемминг, то у одного слова может появиться огромное количество производных, которые размазывают представление о смысле и негативно влияет на совместную встречаемость. А стемминг убивает синтаксические связи, которые важны.

С fasttext получается, что вместо одного вектора для каждого слова у нас их много, можно уловить соотношения, потому что одни вектора несут информацию о смысле слова, другие о его строении. А что еще замечательно в алгоритме -- он дает представление о тех словах, которые мы не встречали и на которых не обучались. 

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_7/embed_results.png?raw=true">
</p>

Есть разные метрики: семантическая точность и синтаксическая точность. Тут приведены сравнения разных алгоритмов. Интересно, что даже если разница большая на тестовой выборке, то при применении к конкретной задаче разница ошибок сильно сокращается с $7\%-70\%$ до разницы в $5\%$.




## Вопрос 2: Deep Reinforcement Learning: Deep Q-Network, Deep Deterministic Policy Gradient.
__Обучение с подкреплением__ (reinforcement learning) — один из способов машинного обучения, в ходе которого испытуемая система (агент) обучается, взаимодействуя с некоторой средой. Откликом среды на принятые решения являются сигналы подкрепления, поэтому такое обучение является частным случаем обучения с учителем, но учителем является среда или её модель. 

![picture](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg)

_Основные параметры марковского процесса:_
* Набор состояний S окружения
* Набор действий агента A
* Функция перехода из одного состояния в другое $P(s’ | s, a)$
-- она может быть дискретной или вероятностной, и описывает вероятность/возможность перейти из одного состояния в другое при совершение действия $а$
* Функция награды $R(s, a, s’)$
-- возвращающая награду за переход в состояние $s’$ после совершения действия $а$ в состояние $s$
* Начальное состояние $s_0$
* Дисконтирующий фактор $\gamma$
-- описывающий то, насколько робот ценит выгоду, получаемую в будущем 

### Deep Q-Network

Функиця $Q^*(s, a)$ означает то, какую максимальную награду можно получить, если начать действовать из этого состояния и совершить конкретное действие а, а дальше действовать оптимальным образом

[Про $Q$-learing](https://medium.freecodecamp.org/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc)
__$Q$-value__

![](https://cdn-images-1.medium.com/max/800/1*YHjnBy5C4vfUX1yPoYTX7g.png)
```
For each state s and action a do {
    Q'(s,a) = 0;
}
Do forever {
    Select action a and execute it in current state s;
    r = reward received;
    s' = new state;
    Q'(s,a) = (1-alpha) Q'(s,a) + alpha (r + gamma * max_a'  Q'(s',a'))
    s = s';
}
```
__Bellman Equation:__

![ebellman](https://cdn-images-1.medium.com/max/1000/1*js8r4Aq2ZZoiLK0mMp_ocg.png)


__Q-Value Iteration:__
$Q^*_{k+1}(s, a)\leftarrow\sum_{s'}P(s'|s, a)(R(s, a, s') + \gamma \cdot max_{a'}Q^*_k(s', a'))$

Обучаем значения также, как Value Iteration:
* Задаем функци $Q^*_0(s, a)$ нулевые значения во всех состояниях.
* Идем по всем состояниям и смотрим в какое состояние мы можем перейти с максимальной выгодой за один ход, т.е в каком из состояний $s'$ функция $Q(s', a')$ максимальна.
* Далее продолжаем процесс аналогичным образом, при с каждой $i$-й итерацией мы считаем какую максимальную награду можно получить, если начать действовать из этого состояния и действовать оптимальным образом при этом совершая не больше i шагов.
* Повторяем до сходимости.

_Value Iteration, Policy Iteration, доказательство сходимости_ в вопросе нет(прямым текстом), но полагаю это важная часть, она норм расписана в конспектах


__Q-network__

Теперь расмотрим $Q$-value network. На вход мы получаем интерпретацию состояния системы, например картинку экрана игры, на выходе сети мы получаем значения $Q(s_t, a)$ для поданного на вход состояния и всех возможных ходов из него. После каждого хода в игре мы получаем четверку $(s, a, r, s')$. По истечении некоторого количества ходов мы скармливаем эти четверки нашей сетке.

![picture](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_folder_7/DQNBreakoutBlocks.png)

_Loss function_
$L_i(\theta_i)=E_{s,a\sim\rho(\cdot)}[(y_i-Q(s,a;\theta_i))^2]$

$y_i=E_{s'\sim\varepsilon}[r+\gamma\cdot max_{a'}Q(s',a';Q_{i-1})|s,a]$

$\nabla_{\theta_i}L_i(\theta_i)=E_{s,a\sim\rho;s'\sim\varepsilon}[(r + \gamma\cdot maxQ(s',a';\theta_{i-1})-Q(s,a;\theta_i))\nabla_{\theta_i}Q(s,a;\theta_i)]$

$\theta$-веса сетки, внезапно

Тут есть большой недостаток - конечное число действий

[Картинка в других обозначениях, где объяснено где что в градиенте](https://cdn-images-1.medium.com/max/1000/1*Zplt-1wTWu_7BGmZCBFjbQ.png)

### Deep Deterministic Policy Gradient

Он решает проблему с бесконечным множеством действий.
Тут появляются две сущьности: Actor и Critic. Critic похож на предыдущую сетку и выдает $Q$-value.

Actor принимает на вход текущее состояние и на выход дает число, представляющее действие выбраное из бесконечного пространства действий. Critic вычисляет Q-value текущего состояния и действия данного Actor. 

![](https://pemami4911.github.io/img/actor-critic.png)

$Q(s,a)\rightarrow Q(s, a|\theta^Q)$-critic
$\mu(s)=argmax(Q)\rightarrow \mu(s|\theta^{\mu})$-actor

__Critic__

$L=\frac{1}{N}\sum_i(y_i-Q(s_i,a_i|\theta^Q)^2)$
$y_i=r_i+\gamma Q'(s_{i+1},\mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})$

__Actor($\pi$)__

$\nabla_{\theta^{\mu}}\mu\approx E_{s_t\sim\rho^{\beta}}[\nabla_{\theta^{\mu}}Q(s,a|\theta^Q)|_{s=s_t,a=\mu(s_t|\theta^{\mu})}] =$
$E_{s_t\sim\rho^{\beta}}[\nabla_{a}Q(s,a|\theta^Q)|_{s=s_t,a=\mu(s_t)}\nabla_{\theta_{\mu}}\mu(s|\theta^{mu})|_{s=s_t}]$

![](https://static1.squarespace.com/static/58523b79cd0f68eedd3d0007/t/5aac16f51ae6cf8c22bdd009/1521227682873/Screen+Shot+2018-03-16+at+3.11.28+PM.png?format=750w)



[Чуть более подробно](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)