# Билет №5
## Вопрос 1: Skip connections. ResNet. Highway connection.

### ResNet

[Источник того, что ниже - хабр.](https://habr.com/post/303196/)

Известно, что если тупо увеличивать количество уровней в каком-нибудь $VGG$ — он начнет тренироваться все хуже и хуже, и в смысле точности на тренировочном сете, и на validation.

Что в некотором смысле странно — более глубокая сеть обладает строго большим representational power.

И, вообще говоря, можно тривиально получить более глубокую модель, которая не хуже менее глубокой, тупо добавив несколько identity layers, то есть уровней, которые просто пропускают сигнал дальше без изменений. Однако, дотренировать обычным способом до такой точности глубокие модели не получается.

Вот это наблюдение, что всегда можно сделать не хуже identity, и есть основная мысль ResNets.
Давайте сформулируем задачу так, чтобы более глубокие уровни предсказывали разницу между тем, что выдают предыдущие лееры и таргетом, то есть всегда могли увести веса в $0$ и просто пропустить сигнал.
Отсюда название — Deep Residual Learning, то есть обучаемся предсказывать отклонения от прошлых лееров.

Более конкретно это выглядит следующим образом.
Основной building block сети — вот такая конструкция:

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/building_block.png?raw=true">
</p>

Два слоя с весами (могут быть convolution, могут быть нет), и shortcut connection, который тупо identity. Результат после двух лееров добавляется к этому identity. Почему каждые два уровня, а не каждый первый? Объяснений нет, видимо на практике заработало вот так.
Поэтому если в весах некого уровня будет везде $0$, он просто пропустит дальше чистый сигнал.

Эта сеть показывает результаты лучше, чем $VGG$!

Чтобы получилось больше лееров, надо делать их полегче — есть идея вместо двух convolutions делать например один и меньшей толщины:

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/building_block1.png?raw=true">
</p>

Было как слева, сделаем как справа. Количество и вычислений, и параметров 
уменьшается радикально.

И вот тут _пацанам_ начинает переть и они начинают тренировать версию на 101 и 152(!) леера. Причем даже у таких сверх-глубоких сетей количество параметров меньше, чем у толстых версий VGG.

_ResNet vs. Plain Network vs. VGG-19_

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/resnet0.png?raw=true">
</p>

Финальный результат на ансамбле, как было упомянуто раньше — $3.57\%$ top5 на Imagenet.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/resnet.png?raw=true">
</p>

### Highway connection

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/highway_connection.png?raw=true">
</p>

[Источник того, что ниже.](http://yanran.li/peppypapers/2016/01/10/highway-networks-and-deep-residual-networks.html)

$ResNet$ пример $Highway\ Networks$.

The ResNet is also motivated by the difficult information flow in deep networks.
> Deeper is not better? Such phenomeon is called “degradation problem”

The motivation is to address the “gradient vanishing” problem, especially when exacerbated the information flow in deeper layers.  In other words, the information is blocked in “traffic problem”. And the intuition is to design mechanism, set up “special path” that rejuvenates the “traffic” in the deep networks, just like “Highway” in our real life. So, that’s where the name comes. The Highway networks.

> To overcome this, we take inspiration from Long Short Term Memory (LSTM) recurrent networks. We propose to modify the architecture of very deep feedforward networks such that information flow across layers becomes much easier. This is accomplished through an LSTM-inspired adaptive gating mechanism that allows for paths along which information can flow across many layers without attenuation. We call such paths information highways. They yield highway networks, as opposed to traditional ‘plain’ networks.
> 
> _attenuation - затухание_





## Вопрос 2: Generative Adversarial Networks (GAN)
[Источник того, что ниже.](https://habr.com/post/332000/)
При всех преимуществах вариационных автоэнкодеров $VAE$, которыми мы занимались в предыдущих постах, они обладают одним существенным недостатком: из-за плохого способа сравнения оригинальных и восстановленных объектов, сгенерированные ими объекты хоть и похожи на объекты из обучающей выборки, но легко от них отличимы (например, размыты).

Этот недостаток в куда меньшей степени проявляется у другого подхода, а именно у генеративных состязающихся сетей — $GAN$’ов.

Формально $GAN$’ы, конечно, не относятся к автоэнкодерам, однако между ними и вариационными автоэнкодерами есть сходства, они также пригодятся для следующей части. Так что не будет лишним с ними тоже познакомиться.

Схема $GAN$:
<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/gan_scheme.png?raw=true">
</p>

$GAN$’ы состоят из $2$ нейронных сетей:
* $1$-ая — генератор сэмплит случайные числа из какого-то заданного распределения $P(Z)$, например  $N(0,I)$ и генерируют из них объекты $X_p = G(Z; \theta_g)$, которые идут на вход второй сети;
* $2$-ая — дискриминатор получает на вход объекты из выборки $X_s$ и созданные генератором $X_p$, и учится предсказывать вероятность того, что конкретный объект реальный, выдавая скаляр $D(X; \theta_d)$.

При этом генератор тренируется создавать объекты, который дискриминатор не отличит от реальных.

### Процесс обучения GAN

Генератор и дискриминатор обучаются отдельно, но в рамках одной сети.

Делаем $k$ шагов обучения дискриминатора: за шаг обучения дискриминатора параметры $\theta_d$ обновляются в сторону уменьшения кросс-энтропии:
$$\theta_d = \theta_d - \nabla_{\theta_d} \left(\log(D(X_s)) + \log(1 - D(G(Z))) \right)$$

Далее шаг обучения генератора: обновляем параметры генератора $\theta_g$ в сторону увеличения логарифма вероятности дискриминатору присвоить сгенерированному объекту лейбл реального.

$$\theta_g = \theta_g + \nabla_{\theta_g} \log(1 - D(G(Z)))$$

Схема обучения:

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/learning_scheme_gan.png?raw=true" >
</p>

_На левой картинке_ шаг обучения дискриминатора: 
градиент (красные стрелки) протекает от лосса только до дискриминатора, где обновляются $\theta_d$ (зеленые) в сторону уменьшения лосса. 
_На правой картинке_ градиент от правой части лосса (ошибка идентификации сгенерированного объекта) протекает до генератора, при этом обновляются только веса генератора $\theta_g$ (зеленые) в сторону увеличения вероятности дискриминатора ошибиться.

Задача, которую решает $GAN$ формулируется так:Задача, которую решает GAN формулируется так:

$$\min_G \max_D \mathbb{E}_{X \sim P}[ \log(D(X))] + \mathbb{E}_{Z \sim P_z}[ \log(1 - D(G(Z)))]$$

При заданном генераторе оптимальный дискриминатор выдает вероятность $D^*(X) = \frac{P(X)}{P_g(X) + P(X)}$, что почти очевидно, предлагаю на секунду об этом задуматься.

[Тут](https://arxiv.org/abs/1406.2661) показывается, что при достаточной мощности обеих сетей у данной задачи есть оптимум, в котором генератор научился генерировать распределение $P_g(X)$, совпадающее с $P(X)$, а везде на $X$ дискриминатор выдает вероятность $1/2$.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/gan_distr.png?raw=true">
</p>

Обозначения:
* черная точечная кривая — настоящее распределение $P(X)$;
* зеленая — распределение генератора $P_g(X)$;
* синяя — распределение вероятности $D(X;\theta_d)$ дискриминатора предсказать класс реального объекта;
* нижняя и верхняя прямые — множество всех $Z$ и множество всех $X$, стрелочки олицетворяют отображение $G(Z;\theta_g)$.

На картинке:

a. $P(X)$ и $P_g(X)$ довольно разные, но дискриминатор неуверенно отличает одно от другого;
b. дискриминатор после $k$ шагов обучения уже отличает их увереннее;
с. это позволяет генератору $G$, руководствуясь хорошим градиентом дискриминатора $D$, на границе двух распределений подвинуть $P_g(X)$ ближе к $P(X)$;
d. в результате многих повторений шагов $(а), (b), (с)$ $P_g$ совпало с $P$, и дискриминатор более не способен отличать одно от другого: $D(X) = 1/2$. Точка оптимума достигнута.

### VAE + GAN

Есть $Conditional Variational Autoencoders$, $CVAE$ ([подробнее о них на русском тут](https://habr.com/post/331552/)), декодер которого умеет генерировать цифру заданного лейбла. Также при помощи $CVAE$ можно создавать картинки цифр других лейблов в стиле заданной картинки. Получилось довольно хорошо, однако цифры генерировались смазанными.

У $GAN'$ов, наоборот, получаются довольно четкие изображения цифр, однако пропала возможность _кодирования (embedding)_ и переноса стиля.

Попробуем взять лучшее от обоих подходов путем совмещения вариационных автоэнкодеров $(VAE)$ и генеративных состязающихся сетей $(GAN)$.

Подход, который будет описан далее, основан на [этой статье](https://arxiv.org/abs/1512.09300).

#### Разберемся более подробно, почему восстановленные изображения получаются смазанные

В первом билете про $VAE$ рассматривался процесс генерации изображений $X$ из скрытых _(latent)_ переменных $Z$.
Так как размерность скрытых переменных $Z$ значительно ниже, чем размерность объектов $X$, а также всегда присутствует некоторая случайность, то одному и тому же $Z$ может соответствовать многомерное распределение $X$, то есть $P(X|Z)$. Это распределение можно представить как:
$$P(X|Z) = f(Z) + \varepsilon,$$
где $f(Z)$ некоторый средний наиболее вероятный объект при заданном $Z$,
а $\varepsilon$ — шум какой-то сложной природы.

Когда мы обучаем автоэнкодеры, мы сравниваем вход из выборки $X_s$ и выход автоэнкодера $\tilde X_s$ с помощью некоторого функционала ошибки $L$,
$$L(X_s, \tilde X_s), \\
\tilde X_s = f_d(Z; \theta_d), \\
Z \sim Q(Z|X_s; \theta_e),$$
где $Q,\ f_d$ — энкодер и декодер.

Задавая $L$, мы определяем шум $\varepsilon_L$, которым приближаем настоящий шум $\varepsilon$.
Минимизируя $L$, мы учим автоэнкодер подстраиваться под шум $\varepsilon_L$, убирая его, то есть находить среднее значение в заданной метрике (во второй части это показывалось наглядно на простом искусственном примере).

Если шум $\varepsilon_L$, который мы определяем функционалом $L$, не соответствует реальному шуму $\varepsilon$, то $f_d(Z; \theta_2)$ окажется сильно смещенным от реального $f(Z)$ (пример: если в регрессии реальный шум лаплассовский, а минимизируется разность квадратов, то предсказанное значение будет смещено в сторону выбросов).

Возвращаясь к картинкам: посмотрим, как связана попиксельная метрика, которой определен лосс в предыдущих частях, и метрика, используемая человеком.  

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/vae_example.png?raw=true">
</p>

На картинке выше:
(a) - это оригинальное изображение цифры;
(b) - получена из (а) отрезанием куска;
(с) - это цифра (а), сдвинутая на полпикселя вправо.

С точки зрения попиксельной метрики (а) намного ближе к (b), чем к (с); хотя с точки зрения человеческого восприятия (b) — даже не цифра, а вот разница между (а) и (b) практически незаметна.

Автоэнкодеры с попиксельной метрикой, таким образом, размазывали изображение, отражая тот факт, что в рамках близких $Z$:
* положение цифр слегка гуляет по картинке;
* нарисованы цифры слегка по-разному (хотя попиксельно может быть значительно далеко).

По метрике же человеческого восприятия тот факт, что цифра размылась, уже заставляет ее быть сильно непохожей на оригинал. Таким образом, если мы будем знать метрику человека или близкую к ней и оптимизировать в ней, то цифры не будут размываться, а важность того, чтобы цифра была полноценной, не как с картинки (b), резко возрастет.

Можно пытаться вручную придумывать метрику, которая будет ближе к человеческой. Но используя подход $GAN$, можно обучить нейронную сеть самой искать хорошую метрику.

#### Соединяя VAE и GAN

Генератор $GAN$ выполняет функцию, аналогичную декодеру в $VAE$: оба сэмплят из априорного распределения $P(Z)$ и переводят его в $P_g(X)$. 
Однако роли у них разные: 
* _декодер_ восстанавливает объект, закодированный энкодером, при обучении опираясь на некоторую метрику сравнения;
* _генератор_ же генерирует случайный объект, который ни с чем не сравнивается, лишь бы дискриминатор не мог отличить, какому из распределений $P$ или $P_g$ он принадлежит.

Идея: добавить в $VAE$ третью сеть — дискриминатор и подавать ей на вход и восстановленный объект и оригинал, а дискриминатор обучать определять, какой из них какой.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/vae_plus_gan.png?raw=true">
</p>

Разумеется, использовать ту же самую метрику сравнения из $VAE$ мы уже не можем, потому что, обучаясь в ней, декодер генерирует изображения, легко отличимые от оригинала. Не использовать метрику вообще — тоже, так как нам бы хотелось, чтобы воссозданный $\tilde X$ был похож на оригинал, а не просто какой-то случайный из $P(X)$, как в чистом $GAN$.

Задумаемся, однако, вот о чем: дискриминатор, учась отличать реальный объект от сгенерированного, будет вычленять какие-то характерные черты одних и других. Эти черты объекта будут закодированы в слоях дискриминатора, и на основе их комбинации он уже будет выдавать вероятность объекта быть реальным. Например, если изображение размыто, то какой-то нейрон в дискриминаторе будет активироваться сильнее, чем если оно четкое. При этом чем глубже слой, тем более абстрактные характеристики входного объекта в нем закодированы.

Так как каждый слой дискриминатора является кодом-описанием объекта и при этом кодирует черты, позволяющие дискриминатору отличать сгенерированные объекты от реальных, то можно заменить какую-то простую метрику (например, попиксельную) на метрику над активациями нейронов в каком-то из слоев:


$$L(X_s, \tilde X_s) \longrightarrow L_d(d_l(X_s), d_l(\tilde X_s)) \\
\tilde X_s = f_d(Z; \theta_d), \\
Z \sim Q(X_s; \theta_e),$$
где $d_l$ — активации на l-ом слое дискриминатора,
а $Q, \ f_d$ — энкодер и декодер.

При этом можно надеяться, что новая метрика $L_d$ будет лучше.

Ниже приведена схема работы получившейся $VAE+GAN$ сети, предлагаемая авторами упомянутой выше статьи

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_5/vae_plus_gan_learning.png?raw=true">
</p>

Здесь:

$X$ — входной объект из $P(X)$;
$Z_p$ — сэмплированный $Z$ из $P(Z)$;
$X_p$ — объект сгенерированный декодером из $Z_p$;
$\tilde X$ — объект восстановленный из $X$,
$\mathcal L_{prior} = KL \left[ Q(Z|X)||P(Z) \right]$ — лосс, заставляющий энкодер переводить $P(X)$ в нужное нам $P(Z)$;
$\mathcal L_{llike}^{Dis_l} = L_d(d_l(X), d_l(\tilde X))$ — метрика между активациями $l$-ого слоя дискриминатора $D$ на реальном $X$ и восстановленным $\tilde X = f_d(Q(X))$;
$\mathcal L_{GAN} = \log(D(X)) + \log(1 - D(f_d(Z))) + \log(1 - D(f_d(Q(X))))$ — кросс-энтропия между реальным распределением лейблов настоящих/сгенерированных объектов, и распределением вероятности предсказываемым дискриминатором.

Как и в случае с $GAN$, мы не можем обучать все $3$ части сети одновременно. Дискриминатор надо обучать отдельно, в частности, не нужно, чтобы дискриминатор пытался уменьшать $\mathcal L_{llike}^{Dis_l}$, так как это схлопнет разницу активаций в $0$. Поэтому обучение всех сетей надо ограничить только на релевантные им лоссы.

Схема, предлагаемая авторами:
$$\theta_{Enc} = \theta_{Enc} - \Delta_{\theta_{Enc}} (\mathcal L_{prior} + \mathcal L^{Dis_l}_{llike}), \\
\theta_{Dec} = \theta_{Dec} - \Delta_{\theta_{Dec}} (\gamma \mathcal L^{Dis_l}_{llike} - \mathcal L_{GAN}), \\
\theta_{Dis} = \theta_{Dis} - \Delta_{\theta_{Dis}} (\mathcal L_{GAN})$$

Выше видно, на каких лоссах какие сети учатся. Особое внимание разве что стоит уделить декодеру: он, с одной стороны, пытается уменьшить расстояние между входом и выходом в метрике $l$-го слоя дискриминатора $(\mathcal L^{Dis_l}_{llike})$, а с другой, пытается обмануть дискриминатор (увеличивая $\mathcal L_{GAN}$). В статье авторы утверждают, что, меняя коэффициент $\gamma$, можно влиять на то, что важнее для сети: контент ($\mathcal L^{Dis_l}_{llike}$) или стиль ($\mathcal L_{GAN}$). Не могу, однако, сказать, что наблюдал этот эффект.
