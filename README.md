# Once for All: Train One Network and Specialize it for Efficient Deployment [[arXiv]](https://arxiv.org/abs/1908.09791) 
```bash
@inproceedings{
  cai2020once,
  title={Once for All: Train One Network and Specialize it for Efficient Deployment},
  author={Han Cai and Chuang Gan and Tianzhe Wang and Zhekai Zhang and Song Han},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=HylxE1HKwS}
}
```

# Train once, specialize for many deployment scenarios
![](https://hanlab.mit.edu/files/OnceForAll/figures/overview_new.png)

# 80% top1 ImageNet accuracy under mobile setting
<p align="center">
    <img src="https://hanlab.mit.edu/files/OnceForAll/figures/ImageNet_mobile_80acc.png" width="80%" />
</p>

# Consistently outperforms MobileNetV3
![](https://hanlab.mit.edu/files/OnceForAll/figures/many_hardwares.png)

# How to use / evaluate **OFA Specialized Networks**
### Use
```python
""" OFA Specialized Networks.
Example: net, image_size = ofa_specialized('flops@595M_top1@80.0_finetune@75', pretrained=True)
""" 
from model_zoo import ofa_specialized
net, image_size = ofa_specialized(net_id, pretrained=True)
```

### Evaluate

`python eval_specialized_net.py --path 'Your path to imagent' --net flops@595M_top1@80.0_finetune@75 `

### Network List
**FLOPs**
* flops@595M_top1@80.0_finetune@75
* flops@482M_top1@79.6_finetune@75
* flops@389M_top1@79.1_finetune@75

**Google Pixel1**
* pixel1_lat@143ms_top1@80.1_finetune@75
* pixel1_lat@132ms_top1@79.8_finetune@75
* pixel1_lat@79ms_top1@78.7_finetune@75
* pixel1_lat@58ms_top1@76.9_finetune@75
* pixel1_lat@40ms_top1@74.9_finetune@25
* pixel1_lat@28ms_top1@73.3_finetune@25
* pixel1_lat@20ms_top1@71.4_finetune@25

**Google Pixel2**
* pixel2_lat@62ms_top1@75.8_finetune@25
* pixel2_lat@50ms_top1@74.7_finetune@25
* pixel2_lat@35ms_top1@73.4_finetune@25
* pixel2_lat@25ms_top1@71.5_finetune@25

**Samsung Note10**
* note10_lat@64ms_top1@80.2_finetune@75
* note10_lat@50ms_top1@79.7_finetune@75
* note10_lat@41ms_top1@79.3_finetune@75
* note10_lat@30ms_top1@78.4_finetune@75
* note10_lat@22ms_top1@76.6_finetune@25
* note10_lat@16ms_top1@75.5_finetune@25
* note10_lat@11ms_top1@73.6_finetune@25
* note10_lat@8ms_top1@71.4_finetune@25

**Samsung Note8**
* note8_lat@65ms_top1@76.1_finetune@25
* note8_lat@49ms_top1@74.9_finetune@25
* note8_lat@31ms_top1@72.8_finetune@25
* note8_lat@22ms_top1@70.4_finetune@25

**Samsung S7 Edge**
* s7edge_lat@88ms_top1@76.3_finetune@25
* s7edge_lat@58ms_top1@74.7_finetune@25
* s7edge_lat@41ms_top1@73.1_finetune@25
* s7edge_lat@29ms_top1@70.5_finetune@25

**LG G8**
* LG-G8_lat@24ms_top1@76.4_finetune@25
* LG-G8_lat@16ms_top1@74.7_finetune@25
* LG-G8_lat@11ms_top1@73.0_finetune@25
* LG-G8_lat@8ms_top1@71.1_finetune@25

**1080ti GPU (Batch Size 64)**
* 1080ti_gpu64@27ms_top1@76.4_finetune@25
* 1080ti_gpu64@22ms_top1@75.3_finetune@25
* 1080ti_gpu64@15ms_top1@73.8_finetune@25
* 1080ti_gpu64@12ms_top1@72.6_finetune@25

**V100 GPU (Batch Size 64)**
* v100_gpu64@11ms_top1@76.1_finetune@25
* v100_gpu64@9ms_top1@75.3_finetune@25
* v100_gpu64@6ms_top1@73.0_finetune@25
* v100_gpu64@5ms_top1@71.6_finetune@25

**Jetson TX2 GPU (Batch Size 16)**
* tx2_gpu16@96ms_top1@75.8_finetune@25
* tx2_gpu16@80ms_top1@75.4_finetune@25
* tx2_gpu16@47ms_top1@72.9_finetune@25
* tx2_gpu16@35ms_top1@70.3_finetune@25

**Intel Xeon CPU with MKL-DNN (Batch Size 1)**
* cpu_lat@17ms_top1@75.7_finetune@25
* cpu_lat@15ms_top1@74.6_finetune@25
* cpu_lat@11ms_top1@72.0_finetune@25
* cpu_lat@10ms_top1@71.1_finetune@25

# How to use / evaluate **OFA Networks**
### Use
```python
""" OFA Networks.
    Example: ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
""" 
from model_zoo import ofa_net
ofa_network = ofa_net(net_id, pretrained=True)
    
# Randomly sample sub-networks from OFA network
ofa_network.sample_active_subnet()
random_subnet = ofa_network.get_active_subnet(preserve_weight=True)
    
# Manually set the sub-network
ofa_network.set_active_subnet(ks=7, e=6, d=4)
manual_subnet = ofa_network.get_active_subnet(preserve_weight=True)
```
### Evaluate

`python eval_ofa_net.py --path 'Your path to imagent' --net ofa_mbv3_d234_e346_k357_w1.0 `

# How to train **OFA Networks**
```bash
mpirun -np 32 -H <server1_ip>:8,<server2_ip>:8,<server3_ip>:8,<server4_ip>:8 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python train_ofa_net.py
```
or 
```bash
horovodrun -np 32 -H <server1_ip>:8,<server2_ip>:8,<server3_ip>:8,<server4_ip>:8 \
    python train_ofa_net.py
```

# Requirement
* Python 3.6
* Pytorch 1.0.0
* ImageNet Dataset 
* Horovod

# Related work on automated and efficient deep learning:
[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/pdf/1812.00332.pdf) (ICLR’19)

[AutoML for Architecting Efficient and Specialized Neural Networks](https://ieeexplore.ieee.org/abstract/document/8897011) (IEEE Micro)

[AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/pdf/1802.03494.pdf) (ECCV’18)

[HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/pdf/1811.08886.pdf)  (CVPR’19, oral)
