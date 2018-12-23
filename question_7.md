# Билет №7
## Вопрос 1: Word embeddings: Co-occurrence Matrix, Word2Vec, CBOW, Skip-Gram, GloVE, FastText.
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