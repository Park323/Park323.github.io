---
title: "[Tutorial] Optimizing Transformer for deployment"
date: 2022-12-28 14:40:00 -0400
categories: Deployment
---

# Optimizing ViT model for Deployment

본 tutorial에서 다루는 내용
- scripting
- quantization
- optimization

본 과정을 통해 모델을 iOS와 Android app에서 사용가능하게 한다.

## Script and Optimize for Mobile Recipe

> Pytorch model -> TorchScript

TorchScript : 모델을 고성능 C++ 환경에서 실행 가능

Pytorch model은 python-dependent하기 때문에 TorchScript로 바꿔주어 mobile app에 최적화할 수 있게 한다.

TorchScript 모델로 바꾸는 방법은 *trace*와 *script*가 있으며 둘을 섞어서도 사용가능하다.

### Trace

Sample 입력을 생성하고, trace 메서드를 통해 sample이 model을 통과하는 과정을 추적*trace*한다.

Model에는 control flow (e.g. if문)이 존재해서는 안된다. 

단순히 model의 forward를 호출하고 그 경과를 추적하기 때문이라고 하는데..

```
import torch

dummy_input = torch.rand(1, 3, 224, 224)
torchscript_model = torch.jit.trace(model_quantized, dummy_input)
```

Model에 flow control이 포함되는 경우 trace가 올바르게 작동하지 않을 수 있다.

```
"""Raise Warning"""
import torch

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

x = torch.rand(3, 4)
traced_cell = torch.jit.trace(MyDecisionGate(), x)
print(traced_cell.code)
```

Error로 인해 위 코드가 중단되지는 않지만, Warning이 발생한다.

해당 내용인즉, sample input외에 다른 input에 대해 일반화되지 않을 수 있다는 것인데, control flow로 인해 input에 따라 추적되는 flow가 달라질 수 있다는 점에서 당연하다.

### Script

위 문제에서는 대신 script를 사용할 수 있다.

```
scripted_cell = torch.jit.script(MyDecisionGate())
print(scripted_cell.code)
```

__Q : script로 모든 model을 TorchScript로 바꿀 수 있나?__
아니다. TorchScript는 python의 subset에 불과하기 때문에, convert할 모델은 그것에 포함된 문법만을 사용해야 한다.

## Quantization

Quantization은 floating point precision대신 더 낮은 bitwidths를 사용해서 tensor계산하는 것을 의미한다. 모델 크기를 줄여서 더 compact한 표현으로 바꾸고, 이를 더 많은 hardware에서 고성능의 벡터 연산을 가능케 한다.

Pytorch에서는 FP32->INT8 quantization을 지원하며 FP32에 비해 INT8에서는 2~4배 빠른 연산이 가능하다. fake-quantization도 지원함.


### Pytorch의 Quantization API mode : Eager Mode & FX Graph Mode

#### Eager mode
사용자가 직접 quantization과 dequantization의 발생을 통제해야하며, functional은 제공되지 않고 module만 제공됨.

#### FX Graph mode
Pytorch가 제공하는 자동화 모듈로, eager mode에 비해 functinoal과 자동화 기능이 추가된다.

Symbolically traceable하지 않는 임의의 모델에는 적용이 안될 수도 있다. 이 경우 refactoring이 필요하며, symbolically traceable한 모델 설계를 위해 **torch.fx**를 공부하자.

Quantization을 처음 사용한다면 FX Graph 모드에서 적용하는 것을 추천하며, 안될 경우 eager mode로 custom할 수 있다.

### Supported quantization

||weight|activation|training|
|---:|:---:|:---:|:---:|
|dynamic|quantized|read/stored in fp and quantized for cumpute|-|
|static|quantized|quantized|require post-training|
|static (aware training)|quantized|quantized|quantization numerics modeled during training|

- dynamic quantization
  weight : 

- static quantization

- static quantization - aware training



# References
https://pytorch.org/tutorials/recipes/script_optimized.html
https://pytorch.org/blog/introduction-to-quantization-on-pytorch/
https://pytorch.org/docs/stable/quantization.html?highlight=quantization#dynamic-quantization
https://pytorch.org/tutorials/beginner/vt_tutorial.html#scripting-deit
https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html