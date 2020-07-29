# Funnel Activation for Visual Recognition
#### 收录到 [PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)
# 说明
- [非官方](https://github.com/megvii-model/FunnelAct/issues/1)
- 即插即用，专为视觉任务设计
- 比Relu\PRelu\Swish等更有效，且迁移性更好
----------

# 环境

| python版本 | pytorch版本 | 系统   |
|------------|-------------|--------|
| 3.6        | 1.5       | Ubuntu |

----------

# Funnel Relu
<img width="869" height="444" src="https://github.com/bobo0810/FunnelAct_Pytorch/blob/master/frelu.png"/>

----------
# 使用
- 对于包含特殊结构的backbone网络，作者表示仅替换主网络中的Relu就足以获得提升！

  ```python
  # 举例ResNet-50  仅替换Bottleneck中的Relu
  #!/usr/bin/env python3
  from frelu import FReLU
  def Bottleneck:
    def __init__():
      self.conv_1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
      self.bn_1 = norm_layer(group_width)
      self.frelu_1 = FReLU(group_width)
      ...
    def forward(self, x):
      out = self.conv_1(x)
      out = self.bn_1(out)
      out = self.frelu_1(out)
      ...
  ```

----------
 # 参考

```
@inproceedings{ma2020funnel, 
            title={Funnel activation for visual recognition},  
            author={Ma, Ningning and Zhang, Xiangyu and Sun, Jian},  
            booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},  
            year={2020} 
}

官方实现MegEngine  https://github.com/megvii-model/FunnelAct
```
