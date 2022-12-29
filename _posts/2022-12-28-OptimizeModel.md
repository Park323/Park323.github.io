---
title: "[Tutorial] Optimizing Transformer for deployment"
date: 2022-12-28 14:40:00 -0400
categories: Deployment
---

# Optimizing Model for Deployment

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

Pytorch에서 Quantized model은 traceable & scriptable하다.
Floating point tensor를 quantized tensor로의 mapping은 사용자가 정의한 block에 의해 결정된다(customizable).

### Pytorch Quantization

#### Quantized Tensor

Quantized Tensor는 int8/uint8/int32로 표현되는 값과 scale/zero_point 등의 quantization parameter 정보를 담고 있다. **Per tensor**는 모든 tensor에 대해 같은 parameter로 quantization하는 것을 의미하며, **per channel**은 각 channel의 tensor 별로 같은 parameter를 공유하는 것을 의미한다. Outlier에 대응하는 관점에서, per channel이 권장됨.

$$Q(x, \mathrm{scale}, \mathrm{zero point}) = \mathrm{round}(\frac{x}{\mathrm{scale}}, \mathrm{zero point})$$

##### Key Attributes

**Qscheme (torch.qscheme)**
Tensor에 대한 quantization 방법을 나타냄.
e.g. per_tensor_affine, per_tensor_symmetric, ...

**dtype (torch.dtype)**
e.g. quint8, qint8, qint32, float16, ...

**quantization parameters (Qscheme에 따라 달라짐)**
e.g. scale, zero_point

#### Quantize and Dequantize

**Quantize (float -> quantized)**
torch.quantize_per_tensor, torch.quantize_per_channel, ...

**Dequantize (quantized -> float)**
torch.dequantize(x), quantized_tensor.dequantize()

#### Quantized Operators/Modules

quntized tensor를 입력받아 quantized tensor를 출력하는 Modules

#### Observer and FakeQuantize

Observer로 tensor의 min/max value 등을 저장한다. 수집된 statistics를 통해 quantization parameter를 계산할 수 있음.

FakeQuantize는 network에서 quantization을 simulate하는데, observer의 statistics를 활용하거나 직접 pararmeter를 배울 수도 있다.

### Pytorch Quantization General Flow

1. Prepare
   Observer/FakeQuantize Module을 삽입한다. (QConfig 사용자 정의)
2. Calibrate / Train
   Observer 또는 FakeQuantize를 동작시킴.
3. Convert
   Model -> Quantized Model

Quantization이 적용되는 시점에 따라 아래와 같이 분류됨.

- Post Training Quantization
- Quantization Aware Training (QAT)

Quantization Operator에 따라 아래와 같이 분류됨

- Weight Only Quantization
  - only weight
- Dynamic Quantization
  - weight + activation (dynamic)
- Static Quantization
  - weight + activation

### Pytorch의 Quantization API mode : Eager Mode & FX Graph Mode

#### Eager mode
사용자가 직접 quantization과 dequantization의 발생을 통제해야하며, functional은 제공되지 않고 module만 제공됨.

#### FX Graph mode
Pytorch가 제공하는 자동화 모듈로, eager mode에 비해 functinoal과 자동화 기능이 추가된다.

Symbolically traceable하지 않는 임의의 모델에는 적용이 안될 수도 있다. 이 경우 refactoring이 필요하며, symbolically traceable한 모델 설계를 위해 **torch.fx**를 공부하자.

Quantization을 처음 사용한다면 FX Graph 모드에서 적용하는 것을 추천하며, 안될 경우 eager mode로 custom할 수 있다.

### Supported quantization

#### Dynamic Quantization

Weights를 INT8로 변환하고, 연산 중간에 Activations를 INT8로 변환하며 on-the-fly로 실행된다(dynamic).
즉, memory에는 fp가 그대로 남아있는 대신에 연산 과정에서는 INT8로 변환되어 효율적인 matrix 연산이 가능해진다.

```
import torch.quantization

# torch.quantization.quantize_dynamic(모델, List[quantization의 대상이 될 submodules], dtype) -> torch.nn.quantized.*
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

#### Post-Training Static Quantization

Integer arithmetic과 INT8 memory access를 둘 다 사용해서 더 빠르게 동작시킬 수 있다.
1. Forward pass
2. Observer를 삽입해서, 다른 activations에서 어떤 분포가 출력되는지 계산하고 기록한다.
3. 2에서 얻은 기록은 각 activation에 대해 어떻게 quantize할지 결정하는 데에 사용된다.
   (e.g. 한 activation에서의 출력 범위를 통해 256 level로 동일하게 구간을 나누어 정한다. the simplest technique)

```
# set quantization config for server (x86)
deploymentmyModel.qconfig = torch.quantization.get_default_config('fbgemm')

# insert observers
torch.quantization.prepare(myModel, inplace=True)
# Calibrate the model and collect statistics

# convert to quantized version
torch.quantization.convert(myModel, inplace=True)
```

#### Quantization-Aware Training (=fake quantization)

보통 가장 높은 정확도를 보이는 방법. 
Training 중에는 값에 round를 적용해서 INT8과 같은 값을 갖게 하지만 계산은 fp를 통해 이뤄진다.

```
# specify quantization config for QAT
qat_model.qconfig=torch.quantization.get_default_qat_qconfig('fbgemm')

# prepare QAT
torch.quantization.prepare_qat(qat_model, inplace=True)

# convert to quantized version, removing dropout, to check for accuracy on each
epochquantized_model=torch.quantization.convert(qat_model.eval(), inplace=False)
```

#### Which approach should I choose?

아래 두 가지 요인에 따라 결정할 수 있다.

1. **Model/Target Requirements**
   Quantization에 예민하게 반응할 수 있는 모델들은 QAT가 권장됨.
2. **Operator/Backend Supports**
   몇몇 backend는 모든 연산이 quantized 되어야함.

#### Quantization Table
||weight|activation|training|
|---:|:---:|:---:|:---:|
|dynamic|quantized|read/stored in fp and quantized for cumpute|-|
|static|quantized|quantized|require post-training|
|static (aware training)|quantized|quantized|quantization numerics modeled during training|


# References
https://pytorch.org/docs/stable/quantization-support.html
https://pytorch.org/tutorials/recipes/script_optimized.html
https://pytorch.org/blog/introduction-to-quantization-on-pytorch/
https://pytorch.org/docs/stable/quantization.html?highlight=quantization#dynamic-quantization
https://pytorch.org/tutorials/beginner/vt_tutorial.html#scripting-deit
https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html