# 实验方法原始文献与引用对应

本清单覆盖正文和已有实验路线。正文按首次引用排序；没有在正文讨论的模型保留为备查来源。早期来源、正式版本和后续扩展分别注明。

| 方法 | 原始/早期文献 | 正文编号 | 使用说明 |
|---|---|:---:|---|
| AlexNet | A. Krizhevsky, I. Sutskever, and G. E. Hinton，[ImageNet classification with deep convolutional neural networks](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)，2012 | 1 |  |
| CLIP | A. Radford et al.，[Learning transferable visual models from natural language supervision](https://proceedings.mlr.press/v139/radford21a.html)，2021 | 2 |  |
| SigLIP2 / NaFlex | M. Tschannen et al.，[SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features](https://arxiv.org/abs/2502.14786)，2025 | 3 |  |
| PE-Core-L / PE-Core-G | D. Bolya et al.，[Perception Encoder: The best visual embeddings are not at the output of the network](https://arxiv.org/abs/2504.13181)，2025 | 4 |  |
| SIFT | D. G. Lowe，[Object recognition from local scale-invariant features](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)，1999 | 5 | 1999 年初始提出；与 2004 年完整版本区分。 |
| SIFT | D. G. Lowe，[Distinctive image features from scale-invariant keypoints](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)，2004 | 6 | 稳定性、描述子及完整分析版本，不称为首次提出。 |
| 视觉词袋 | J. Sivic and A. Zisserman，[Video Google: A text retrieval approach to object matching in videos](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/)，2003 | 7 | 视觉词汇/图像检索的早期来源；本项目改用于回归。 |
| Ridge | A. E. Hoerl and R. W. Kennard，[Ridge regression: Biased estimation for nonorthogonal problems](https://doi.org/10.1080/00401706.1970.10488634)，1970 | 8 | 出版信息经 Crossref 原始 DOI 元数据核对。 |
| AdamW | I. Loshchilov and F. Hutter，[Decoupled weight decay regularization](https://arxiv.org/abs/1711.05101)，2019 | 9 | 2017 年预印本，2019 年 ICLR 正式版本；解耦衰减不等同于 Adam 的 L2 惩罚。 |
| Adam | D. P. Kingma and J. Ba，[Adam: A method for stochastic optimization](https://arxiv.org/abs/1412.6980)，2015 | 12 | 2014 年预印本，2015 年 ICLR。 |
| Dropout | G. E. Hinton et al.，[Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)，2012 | 10 | 早期提出版本。 |
| Dropout | N. Srivastava et al.，[Dropout: A simple way to prevent neural networks from overfitting](https://jmlr.org/papers/v15/srivastava14a.html)，2014 | 11 | 完整期刊版本。 |
| 余弦学习率 | I. Loshchilov and F. Hutter，[SGDR: Stochastic gradient descent with warm restarts](https://arxiv.org/abs/1608.03983)，2017 | 13 | 本项目只借鉴余弦衰减，不使用 warm restarts；不将两者混称。 |
| 小卷积核堆叠 | K. Simonyan and A. Zisserman，[Very deep convolutional networks for large-scale image recognition](https://arxiv.org/abs/1409.1556)，2015 | 16 | 2014 年预印本；作为结构设计依据，不称本项目训练了完整 VGG。 |
| 多尺度 / Inception | C. Szegedy et al.，[Going deeper with convolutions](https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)，2015 | 17 |  |
| 残差连接 | K. He et al.，[Deep residual learning for image recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_2016_CVPR_paper.html)，2016 | 18 | 2015 年预印本，2016 年 CVPR；项目只改 AlexNet 残差模块。 |
| Transformer | A. Vaswani et al.，[Attention is all you need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)，2017 | 19 | 本项目使用 CNN token 上的单层编码器，不是原论文完整翻译模型。 |
| RBF 核岭回归 | C. Saunders, A. Gammerman, and V. Vovk，[Ridge regression learning algorithm in dual variables](https://eprints.soton.ac.uk/258942/)，1998 | 20 | 引用双变量核岭回归方法来源；不混用 RBF 神经网络文献。 |
| PLS / 迭代最小二乘历史源头 | H. Wold，Estimation of principal components and related models by iterative least squares，1966 | 21 | 1966 年历史书章，原始全文未获取；与可公开核对的 1984 年回归研究配套引用，不赋予后者的 DOI。 |
| PLS 回归 | S. Wold, A. Ruhe, H. Wold, and W. J. Dunn III，[The collinearity problem in linear regression. The partial least squares (PLS) approach to generalized inverses](https://doi.org/10.1137/0905052)，1984 | 22 | 早期回归论文；与 1966 年历史源头配套引用。 |
| PCA | K. Pearson，[On lines and planes of closest fit to systems of points in space](https://doi.org/10.1080/14786440109462720)，1901 | 23 | 出版信息经 Crossref 原始 DOI 元数据核对。 |
| OOF 预测融合 | L. Breiman，[Stacked regressions](https://doi.org/10.1007/BF00117832)，1996 | 24 | 非负回归融合的早期来源；本项目另加权重和为 1 与晋级门槛。 |
| 区域遮挡敏感性 | M. D. Zeiler and R. Fergus，[Visualizing and understanding convolutional networks](https://arxiv.org/abs/1311.2901)，2014 | 25 | 2013 年预印本；本项目的 4×4 灰色/模糊控制是任务适配。 |
| Adagrad | J. Duchi, E. Hazan, and Y. Singer，[Adaptive subgradient methods for online learning and stochastic optimization](https://jmlr.org/papers/v12/duchi11a.html)，2011 | 备查 |  |
| Xavier 初始化 | X. Glorot and Y. Bengio，[Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)，2010 | 14 |  |
| Kaiming 初始化 | K. He et al.，[Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)，2015 | 15 |  |
| SigLIP 家族原始方法 | X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer，[Sigmoid loss for language image pre-training](https://arxiv.org/abs/2303.15343)，2023 | 备查 |  |
| DINOv2 冻结特征 | M. Oquab et al.，[DINOv2: Learning robust visual features without supervision](https://arxiv.org/abs/2304.07193)，2023 | 备查 |  |
| BLIP 图像编码器 | J. Li, D. Li, C. Xiong, and S. Hoi，[BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation](https://proceedings.mlr.press/v162/li22n.html)，2022 | 备查 |  |
| StreetCLIP | L. Haas, M. Skreta, and S. Alberti，[Learning generalized zero-shot learners for open-domain image geolocalization](https://arxiv.org/abs/2302.00275)，2023 | 备查 |  |
| AIMv2 | A. El-Nouby et al.，[Multimodal autoregressive pre-training of large vision encoders](https://arxiv.org/abs/2411.14402)，2024 | 备查 |  |
| EVA-CLIP | Q. Sun et al.，[EVA-CLIP: Improved training techniques for CLIP at scale](https://arxiv.org/abs/2303.15389)，2023 | 备查 |  |
| 视觉编码器 LoRA 适配 | E. J. Hu et al.，[LoRA: Low-rank adaptation of large language models](https://arxiv.org/abs/2106.09685)，2022 | 备查 | 2021 年预印本。引用低秩适配方法，不代表课程方案使用 MLLM。 |
| 随机擦除 | Z. Zhong et al.，[Random erasing data augmentation](https://arxiv.org/abs/1708.04896)，2020 | 备查 | 2017 年预印本；本项目灰色填充等设置按自身协议。 |
| 传统空间金字塔 | S. Lazebnik, C. Schmid, and J. Ponce，[Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories](https://doi.org/10.1109/CVPR.2006.68)，2006 | 备查 |  |
| SVR 传统回归对照 | H. Drucker et al.，[Support vector regression machines](https://papers.nips.cc/paper_files/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html)，1996 | 备查 |  |
| ViT 编码器架构基础 | A. Dosovitskiy et al.，[An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/abs/2010.11929)，2021 | 备查 | 2020 年预印本；作为 CLIP/PE 等架构背景，不冒充项目额外训练的完整 ViT。 |
| 动量方法历史源头 | B. T. Polyak，[Some methods of speeding up the convergence of iteration methods](https://doi.org/10.1016/0041-5553(64)90137-5)，1964 | 备查 | 不把确定性 heavy-ball 分析等同于本项目的随机小批量训练。 |
| 随机优化历史基础 | H. Robbins and S. Monro，[A stochastic approximation method](https://doi.org/10.1214/aoms/1177729586)，1951 | 备查 | 随机逼近基础，不称为现代深度学习 SGD 实现的完整来源。 |

## 不虚构单一首创来源的部分

- 亮度、对比度、饱和度、翻转、缩放、平移、噪声与 Gaussian 滤波：基础操作按项目配置描述，不为增加条数附会某篇首创论文。
- 多视图拼接、PLS 与图像核相乘、PCA 中位距离带宽校准和融合门槛：本项目的组合与适配；引用基础方法，不称完整方案由这些论文提出。
- PLS 的 1966 年书章是历史来源；1984 年为可核对的早期回归论文。PCA 使用 Pearson 1901 年源头文献。
- Muon：初始来源主要是作者技术文章及代码，而非一篇可直接认定的首创同行评审论文。原始技术来源：[Keller Jordan](https://kellerjordan.github.io/posts/muon/)、[Muon 代码](https://github.com/KellerJordan/Muon)。
- SVR、随机优化/动量、ViT 等底层方法另列备查来源；只有相应实验在正文展开时，才加入正文编号。
- 未执行的候选及课程排除的独立 MLLM 路线不作为正文实验引用。
