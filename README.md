# StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN

## Abstract
StyleGAN은 고해상도의 이미지를 현실적으로 생성한다. 하지만 특정 부분에서 노이즈를 생성하고 또 부자연스러운 부분을 생성하기도 한다.

![](https://images.velog.io/images/heaseo/post/fa8cd4cd-b240-4ead-ba00-27b5b8229415/StyleGAN_artifacts.png)

빨간 동그라미로 표시된 노이즈는 항상 발생되는 것은 아니지만 특정 해상도 64x64에 모든 feature maps에서 발생된다. 논문의 저자는 normalization layer (AdaIN) 때문에 생성되는 것이라고 생각하고 있다. 스파이크 유형 분포가 작은 feature map에 들어오면 원래 값이 작더라도 AdaIN에 의해 값이 증가하여 큰 영향을 준다. 

다음은 부자연스러운 부분에 대한 지적이다. 때때로 생성된 이미지의 작은 특성들이 얼굴의 각도가 달라졌음에도 불구하고 고정된 방향성을 가지고 있어서 부자연스럽게 보인다. 이 논문의 저자는 아마 이러한 부자연스러움이 progressive growing에서 비롯된 것이라고 한다. Progressive growing에서 각 해상도에서 이미지를 독립적으로 생성한 것이 문제였던 것이다.

![](https://images.velog.io/images/heaseo/post/cffbd8ad-0af7-4c69-a356-516ce146d07a/StyleGANv1_artifacts.png)

StyleGAN2는 이러한 문제점을 해결하기 위해 PPL과 향상된 droplet noise & non-follwing modes로 극복하고자 한다. [더보기](https://velog.io/@heaseo/StyleGAN-V2-Analyzing-and-Improving-the-Image-Quality-of-StyleGAN)

<br />

## Training
```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=3 python3 train.py --train-dir ${train directory} --eval-dir ${test directory} - --outputs-dir ${a directory where trained models will be saved}  --batch-size 16 --patch-size 256
```

```bash
# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --train-dir ${train directory} --eval-dir ${test directory} - --outputs-dir ${a directory where trained models will be saved}  --batch-size 16 --patch-size 256--distributed
```

## 현재까지 학습 진행 결과
![](https://images.velog.io/images/heaseo/post/d2da0882-456c-4ca9-a688-90b6391cadef/preds_0.jpg)

<br />

![](https://images.velog.io/images/heaseo/post/f7d60a9b-7847-4fd4-9d1d-66908815204f/preds_1.jpg)

<br />

![](https://images.velog.io/images/heaseo/post/44babf96-1c01-44fb-904d-7f8ef83318ba/preds_10.jpg)

<br />

![](https://images.velog.io/images/heaseo/post/d21d242a-7f6b-405a-ae0e-6abd1c379346/preds_31.jpg)

아직 학습이 끝나지는 않았지만 점차 학습이 진행되면서 모델이 어떤 결과를 보여주는지 볼 수 있었다. 