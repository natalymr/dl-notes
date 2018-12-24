# Билет №3
## Вопрос 1: Attention. Gated Attention.
## Вопрос 2: Image Segmentation Networks: FCN, SegNet, UNet.

[Источник того, что ниже - википедия.]()

**Сегментация изображения**

Результатом сегментации изображения является множество сегментов, которые вместе покрывают всё изображение, или множество контуров, выделенных из изображения. Все пиксели в сегменте похожи по некоторой характеристике или вычисленному свойству, например, по цвету, яркости или текстуре. Соседние сегменты значительно отличаются по этой характеристике.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/segmentation.png?raw=true">
</p>

### FCN

[Истоник того, что ниже - лекции.](https://docs.google.com/document/d/1nRC4KPQAxBrNBLu8Jr7Ii_xa5ztsmN6ZupraCeYUIVQ/edit#heading=h.1txw3rlwi08z)

Проблемы segmentation мы коснемся вкратце. Здесь тоже пока рулит FCN. Сама идея довольно простая -- давайте сделаем deconvolution. $21$ -- фон и $20$ классов. В конце вместо fully connected снова делаем convolution и получаем свертки для объектов. После этого deconvolution -- по значению нейрона восстанавливаем картинку. Из одного числа как-то не очень, а вот из $16$ уже неплохо. То есть мы из $16x16x21$ восстанавливаем исходную картинку $64x64$.


[Источник того, что ниже - презентация.](http://www.machinelearning.ru/wiki/images/c/c4/NN_for_segmentation_and_Keras.pdf)

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/FCN.png?raw=true">
</p>

* FCN полностью состоят из сверточных слоев
* Получают сегментацию того же размера, что и исходное изображение
* Используют идею transfer learning -- адаптируют VGG, GoogleNet и другие
    * Предобученную сеть для классификации можно использовать для сегментации
    * После встраивания слоев можно произвести fine-tuning
        <p align="center">
          <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/fcn1.png?raw=true">
        </p>
    * Основная проблема -- низкое разрешение на выходе
        * Для повышения разрешения (upsampling) можно использовать как простые, так и обучаемые методы

#### Upsampling

[Источник того, что ниже.](https://www.quora.com/What-is-the-difference-between-Deconvolution-Upsampling-Unpooling-and-Convolutional-Sparse-Coding)
**Upsampling refers to any technique that, well, upsamples your image to a higher resolution.**

1. **Deconvolution** in the context of convolutional neural networks is often used to denote a sort of reverse convolution, which importantly and confusingly is not actually a proper mathematical deconvolution. _In contrast to unpooling, using ‘deconvolution’ the upsampling of an image can be learned._ It is often used for upsampling the output of a convnet to the original image resolution. 
[Источник того, что ниже - хабр.](https://habr.com/company/oleg-bunin/blog/340184/)
    <p align="center">
      <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/deconv.png?raw=true">
    </p>
    
    > По сути, это обучаемый Upsampling. То есть вы уменьшили в какой-то момент ваше изображение до какого-то небольшого размера, может, даже до $\ ps$. Скорее, не до пикселя, а до какого-то небольшого вектора. Потом можно взять этот вектор и раскрыть.
    >
    > Или, если в какой-то момент получилось изображение $10*10\ ps$, теперь можно делать Upsampling этого изображения, но каким-то хитрым способом, в котором веса Upsampling также обучаются.
    >
    >Это — не магия, это работает, и фактически это позволяет обучать нейросети, которые из входной картинки получают какую-то выходную картинку. То есть вы можете подавать образцы входа/выхода, а то, что посередине, обучится само.

2. **Unpooling** is commonly used in the context of convolutional neural networks to denote reverse max pooling.
    <p align="center">
      <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/deconv1.png?raw=true">
    </p>
    
    > [Истоник того, что ниже - лекции.](https://docs.google.com/document/d/1nRC4KPQAxBrNBLu8Jr7Ii_xa5ztsmN6ZupraCeYUIVQ/edit#heading=h.1txw3rlwi08z)
    > Обычный max pooling -- это downsampling, уменьшение количества информации. Сейчас будем делать наоборот. Для этого при downsampling будем запоминать из какого места взяли максимум и потом при upsampling эти максимумы вписываем в нужные места, а дальше просто выравниваем.


**Архитектура различных FCN**
<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/fcn2.png?raw=true">
</p>

Можно ансамблировать результаты по разным разрешениям:

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/fcn3.png?raw=true">
</p>

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/ensemble.png?raw=true">
</p>

### SegNet
[Источник того, что ниже.](http://mi.eng.cam.ac.uk/projects/segnet/)

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/segnet.png?raw=true">
</p>

The architecture consists of a sequence of non-linear processing layers (encoders) and a corresponding set of decoders followed by a pixelwise classifier. Typically, each encoder consists of one or more convolutional layers with batch normalisation and a ReLU non-linearity, followed by non-overlapping maxpooling and sub-sampling. The sparse encoding due to the pooling process is upsampled in the decoder using the maxpooling indices in the encoding sequence (see the figure below). One key ingredient of the SegNet is **the use of max-pooling indices in the decoders to perform upsampling of low resolution feature maps**. This has the important advantages of retaining high frequency details in the segmented images and also reducing the total number of trainable parameters in the decoders. The entire architecture can be trained end-to-end using stochastic gradient descent. The raw SegNet predictions tend to be smooth even without a CRF based post-processing.

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/segnet1.png?raw=true">
</p>

### UNet

[Источник того, что ниже.](http://robocraft.ru/blog/machinelearning/3671.html)

<p align="center">
  <img src = "https://github.com/natalymr/dl-notes/blob/master/pictures/question_3/unet.png?raw=true">
</p>

Архитектура сети представляет собой последовательность слоёв свёртка+пулинг, которые сначала уменьшают пространственное разрешение картинки, а потом увеличивают его, предварительно объединив с данными картинки и пропустив через другие слои свёртки. Таким образом, сеть выполняет роль своеобразного фильтра.

Для обучения сети, считается коэффициент Дайса (Dice coefficient) (так же называется коэффициент Сёренсена — Sorensen–Dice coefficient) или Жаккара (Jaccard similarity coefficient), который показывает меру сходства — в данном случае, показывающий меру площади правильно отмеченных сегментов (отношение площади пересечения к площади объединения).