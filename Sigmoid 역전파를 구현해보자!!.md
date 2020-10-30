## 필요한 라이브러리 호출
```python
import torch
import torch.nn as nn
from torch.autograd import Function
```

## 동일한 x 사전 정의 (gradient를 저장할 leaf 생성)
```python
x = torch.randn(10,)
x.requires_grad = True
x
>>> tensor([-1.1781,  0.7656, -0.4253,  1.5910,  1.1055,  1.4665, -1.3581, -2.5128,
>>>         -0.6691,  0.6490], requires_grad=True)
```

## Pytorch에서 정의하는 역전파는?
```python
z = torch.sigmoid(x)
z
>>> tensor([0.2354, 0.6826, 0.3952, 0.8308, 0.7513, 0.8125, 0.2045, 0.0750, 0.3387,
>>>         0.6568], grad_fn=<SigmoidBackward>)

z.mean().backward(retain_graph=True)
x.grad
>>> tensor([0.0180, 0.0217, 0.0239, 0.0141, 0.0187, 0.0152, 0.0163, 0.0069, 0.0224,
>>>         0.0225])
```

## 만약 직접 이런 함수(chain rule)를 정의하고 싶다면?
```python
import numpy as np


class Custom(Function):
    
    @staticmethod
    def forward(ctx, i):
        result = 1 / (1 + np.exp(-i))
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result * (1 - result)
```

```python
# 새로운 계산 그래프 정의
z = Custom.apply(x)
z # 계산값은 위와 동일
>>> tensor([0.2354, 0.6826, 0.3952, 0.8308, 0.7513, 0.8125, 0.2045, 0.0750, 0.3387,
>>>         0.6568], grad_fn=<CustomBackward>)
```

```python
x.grad = torch.zeros(x.size())
z.mean().backward()
x.grad # 위와 같다.
>>> tensor([0.0180, 0.0217, 0.0239, 0.0141, 0.0187, 0.0152, 0.0163, 0.0069, 0.0224,
>>>         0.0225])
```

## 이는 무조건 chain rule 계산 함수에서만 쓰이나? No! DNI 활용체에선 이렇게도 정의한다.

```python
class _SyntheticGradientUpdater(torch.autograd.Function):

    @staticmethod
    def forward(ctx, trigger, synthetic_gradient):
        (_, needs_synthetic_gradient_grad) = ctx.needs_input_grad
        if not needs_synthetic_gradient_grad:
            raise ValueError(
                'synthetic_gradient should need gradient but is does not'
            )
        ctx.save_for_backward(synthetic_gradient)
        # clone trigger to force creating a new Variable with
        # requires_grad=True
        return trigger.clone()

    @staticmethod
    def backward(ctx, true_gradient):
        (synthetic_gradient,) = ctx.saved_variables
        # compute MSE gradient manually to avoid dependency on PyTorch
        # internals
        (batch_size, *_) = synthetic_gradient.size()
        grad_synthetic_gradient = (
            2 / batch_size * (synthetic_gradient - true_gradient)
        )
        return (true_gradient, grad_synthetic_gradient)
            
```
