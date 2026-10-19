# 基于冻结视觉特征与核回归融合的房价预测

## 引言

房屋售价预测需要综合房屋自身属性与外部环境，包括面积、房屋状况、地理位置和市场条件。外观照片提供了其中一部分信息：计算机视觉可以描述建筑形态、表面外观、庭院绿化及周边场景。然而，单张照片的覆盖范围有限，也无法直接给出建筑面积、精确地址或交易条件。基于这一背景，本研究探讨如何更有效地提取房屋图像中的视觉特征，从而更准确地预测房屋售价。

这一问题可以从不同层次的视觉表示入手。传统手工局部描述子概括梯度与纹理模式，而 AlexNet [1] 等卷积网络能够从训练图像中学习层次化特征。随着大规模预训练的发展，视觉表示可以在不同任务之间复用。CLIP [2]、SigLIP2 [3] 和 Perception Encoder [4] 提供了这类表示，使冻结编码器、单独学习任务回归器成为可行路线。

据此，本研究分三个阶段展开。首先，使用 SIFT 与视觉词袋检验局部外观是否包含价格信号；随后，引入 AlexNet 回归模型，通过训练调优、数据增强和架构对照逐步改进。尽管这些调整改善了基线，剩余误差仍然较大，因此第三阶段转向冻结预训练特征，进一步研究图像覆盖、特征变换与回归方式如何影响价格预测，并组合互补分支。

最终三编码器融合的验证 MSE 为 52,201.21，MAE 为 159.58；全量重拟合后的 Kaggle 公榜 MSE 为 47,291.35。后续分析将这些收益与视觉表示及核回归设计联系起来，同时考察持续存在的高价低估和不同价位之间不均匀的改善。

## 方法

### 数据、指标与评价方法

数据集包含来自南加州房产挂牌信息的 8,000 张有标签训练图像和 3,000 张无标签测试图像，价格单位为千美元。模型使用课程提供的图像和价格拟合，通用预训练图像编码器提供可迁移的视觉特征。

我们使用种子 2026 固定划分 6,399 张训练图像和 1,601 张验证图像，将已识别的重复或关联图像保留在同一组中。主要指标为 MSE，同时使用 MAE 和有符号残差 $\hat y-y$ 描述典型误差与预测偏差。MSE 的单位为（千美元）$^2$，MAE 和残差的单位为千美元。

随机训练模型先使用种子 2026 初筛，有希望的候选再以种子 2027 和 2028 复核。当三种子的平均 MSE 下降，且至少两次同种子配对比较改善时，保留相应改动，并报告均值与样本标准差。冻结特征回归器通过训练集上的分组三折折外预测（OOF）选择。视觉词典、标准化器、PCA 和监督变换均在各折训练部分拟合。

### 基线与 CNN 受控实验

我们首先以训练集均价建立不使用图像的参照。为检验局部外观能否提供额外信息，提取灰度 SIFT 描述子 [5, 6]，构建包含 256 个视觉词的词典 [7]。直方图依次进行 L1 归一化、平方根变换、L2 归一化及标准化，再拟合岭回归 [8]（$\alpha=10$）。虽然这种表示能够描述局部梯度结构，但直方图汇总会丢失空间排列，因此进一步引入直接从图像学习特征组合的 CNN。

AlexNet 风格回归网络 [1]由五个卷积层和 9216--4096--4096--1 的预测头组成，从随机初始化开始训练。经过预处理比较，选定直接缩放至 $224\times224$，使用均值和标准差均为 0.5 的 RGB 归一化，不进行滤波。保留配置为 AdamW [9]，学习率与权重衰减均为 $10^{-4}$，dropout [10, 11] 为 0.5。每个微批次包含 4 张图像，累积 8 次梯度，得到有效批量 32。训练时将价格除以 1,000，预测时恢复至数据集尺度。每次最多训练 60 轮，至少训练 40 轮，早停耐心为 15 轮，有效改善阈值为 0.1%；保存验证 MSE 严格最低的检查点。

实验先比较学习率、包含 Adam [12] 的优化器、不重启的余弦衰减 [13]、有效批量、正则化和初始化 [14, 15]，再研究数据增强。几何变换与退化变换分别测试，其中水平翻转的概率为 0.5。架构实验借鉴小核堆叠 [16]，用两个 $3\times3$ 卷积替换第二个 $5\times5$ 卷积，检验局部特征组合；通过 Inception 风格的多尺度模块 [17] 比较不同感受野；通过投影残差连接 [18] 检验特征复用。为减小全连接预测头，我们将 $6\times6$ 卷积特征投影到 128 维，加入位置信息与归一化，对 36 个 token 求均值后进行线性回归。随后测试单层 Transformer [19]，维度为 128，注意力头数为 4，前馈层维度为 512。每项改动均与对应对照比较。

另以种子 2026 进行独立的光度增强初筛：亮度或对比度分别在每张训练图像上使用从 $[0.9,1.1]$ 均匀抽取的倍率，在缩放与归一化前应用。两者均与新训练的无增强 AlexNet 比较，保持初始化、划分和训练设置一致。

### 冻结表示、多视图与回归

为进一步改进从零训练模型，我们将视觉表示与价格回归分开研究。冻结 CLIP ViT-B/32 特征经标准化后结合岭回归，建立迁移学习参照；最终分支使用 SigLIP2 So400m NaFlex、PE-Core-L/14 和 PE-Core-G/14。NaFlex 使用 576 个 patch 保留原始宽高比，输出 1,152 维特征。考虑到单次裁剪可能移除有用环境，PE-L 组合三个 1,024 维视图：官方输入、336 像素灰色补边整图，以及保留原图宽高各 60% 的中央裁剪。PE-G 组合官方 448 像素视图与灰色补边整图。各视图特征先独立单位化，拼接后再次单位化，得到 3,072 维 PE-L 特征和 2,560 维 PE-G 特征。

我们使用 RBF 核岭回归 [20] 描述冻结特征空间中的非线性关系，并通过正则化控制拟合系数。对于单位特征 $x$：

$$

 K_{ij}=\exp(-\gamma\|x_i-x_j\|^2),\quad
 c=(K+\alpha I)^{-1}(y-\bar y\mathbf{1}).

$$

预测为 $\bar y+k(x)^Tc$，并将负预测截断为零。普通核回归在训练折中搜索 $\gamma\in\{1,4,16\}$ 和 $\alpha\in\{0.01,0.1,1,10\}$。NaFlex 选定 $\gamma=1$、$\alpha=0.1$。

然而，通用视觉相似度也可能强调与价格无关的变化。因此，PE-L 通过折内拟合的偏最小二乘（PLS） [21, 22] 学习价格相关方向。每折拟合 16 个分量，选定预测头使用前四个标准化得分 $s$，构造乘积核：

$$

 k(x,x')=\exp\left(-\|x-x'\|^2-0.1\frac{\|s-s'\|^2}{4}\right),

$$

其中 $\alpha=1$。PLS 得分在保持图像编码器冻结的同时，将价格相关方向引入相似度计算。

与监督式 PLS 不同，PCA [23] 不使用价格，而是沿主要变化方向压缩特征。对于 PE-G，我们比较 64、256 和 1,024 维，白化幂取 0 和 0.5，变换后再次单位化。由于这会改变特征距离，我们同时对原有 $\gamma=4$ 进行校准：

$$

 \gamma'=4\frac{\operatorname{median}\|x_i-x_j\|^2}
 {\operatorname{median}\|z_i-z_j\|^2}.

$$

其中 $z$ 为变换后的单位向量，中位数由各折训练部分固定抽取的 8,192 对不同图像计算。最终选定不白化的 PCA256，$\alpha=1$。

### 融合与最终重拟合

借鉴堆叠回归 [24]，各分支以非负权重融合，并增加权重总和为 1 的约束。权重由训练 OOF 预测拟合，并通过元模型交叉拟合评价：每折使用其余折拟合的权重预测。当新增分支使 meta-CV MSE 至少改善 1%，至少两折改善，且验证 MSE 同时下降时，保留该分支。

最终权重分别为 NaFlex 0.108065、PE-L PLS 0.080806、PE-G PCA 0.811129。生成 Kaggle 提交时，保持选定超参数和融合权重，在全部 8,000 张有标签图像上重新拟合变换与回归头，并重新计算训练集的 PCA 和带宽统计，再预测 3,000 张测试图像的价格。

## 结果

| 方法 | 种子数 | 验证 MSE | 验证 MAE |
|---|:---:|---:|---:|
| AlexNet（无增强） | 3 | 127,774 ± 486 | 264.52 |
| AlexNet（Adam，lr=0.001） | 3 | 128,284 ± 1,106 | 264.80 |
| AlexNet + 水平翻转 | 3 | 126,810 ± 1,902 | 268.02 |
| 翻转 AlexNet + 小卷积核堆叠 | 1 | 127,320 | 262.19 |
| 翻转 AlexNet + 均值池化小头 | 3 | 122,334 ± 1,589 | 261.48 |
| 均值池化小头 + Transformer | 1 | 124,389 | 259.55 |
| 冻结 CLIP B32 + 岭回归 | — | 91,821 | 223.27 |
| 冻结 PE-G 双视图 + PCA/KRR | — | 52,806 | 160.67 |
| 三编码器融合 | — | 52,201 | 159.58 |

表 1. 验证集 n = 1,601；MSE 单位为（千美元）²，MAE 单位为千美元。三种子结果报告 MSE 均值 ± 样本标准差及 MAE 均值；单种子使用 2026；破折号表示冻结特征评价。


### 从基线到冻结视觉特征

均价基线的 MSE 为 151,839.69；SIFT--BoVW--Ridge 在三种子下将其降至 $145,336.73\pm1,091.58$，表明局部外观包含有限但可利用的价格信息。原始补边输入 AlexNet 的平均 MSE 为 128,415.68。表 1从后续的直接缩放版本开始，其 MSE 为 127,774.42，也是训练调优阶段统一使用的无增强对照。

仅调整优化方式带来的收益有限。尽管学习率为 0.001 的 Adam 在种子 2026 下改善，另外两次配对比较却退步，平均 MSE 增加 509.89。类似地，余弦调度、替代批量和降低 dropout 均未达到三种子保留条件，因此沿用原训练设置。

固定这些设置后，水平翻转使平均 MSE 降低 964.41，赢得两次配对比较，但 MAE 从 264.52 升至 268.02。更明确的收益来自均值池化小头：三个种子均改善，相对翻转 AlexNet 的 MSE 降低 4,475.78。相比之下，小卷积核、多尺度和残差结构的初筛均未超过对照；单层 Transformer 相对小头的单种子 MSE 增加 3,615.11。因此保留结构紧凑的预测头。

独立光度初筛中新对照的 MSE 为 131,389.94，MAE 为 282.64。亮度增强后的 MSE 为 131,433.91，基本不变，MAE 降至 274.50。对比度增强将 MSE 降至 128,860.53（改善 1.93%），MAE 降至 269.01，但价格高于 1,300 的尾部 MSE 增加 17.14%；亮度增强使该尾部指标增加 3.86%。对比度在两个较低价位组的收益抵消了两个较高价位组的退步。因此，可将其列为后续配对多种子复核候选，尚不替换已保留的增强方案。

综合来看，从零训练路线虽然调整了优化、增强与容量，收益仍然有限。相比之下，冻结 CLIP 与岭回归将验证 MSE 降至 91,820.73。更大的改善使后续实验转向更丰富的预训练表示，以及如何更有效地利用这些表示。

### 图像覆盖与核几何


![分析图](figures/report_pca_distance_bandwidth_oof.png)

PE-G 几何与回归诊断。（a、b）8,192 对训练样本的距离与核相似度经验累积分布；（c）不白化 PCA 的分组三折 OOF MSE。变换与带宽均在各折内拟合；（b）展示相似度 0--0.4 范围。



在冻结特征路线中，我们首先研究图像覆盖。为 PE-G 补充整图后，单分支验证 MSE 从 55,097.49 降至 53,252.81，OOF MSE 从 55,679.73 降至 53,646.46。这表明，在保留官方视图的基础上补充外围环境能够改善预测。

随后，我们检验压缩这些特征能否继续改善预测。PCA256 最初沿用 $\gamma=4$ 时，OOF MSE 升至 77,771.43；经带宽校准后则降至 52,531.27。结果反转伴随着明显的距离尺度变化：开发训练集的平方距离中位数从 0.5402 升至 2.0515，对应校准后的 $\gamma=1.05323$（图 1）。选定分支的验证 MSE 为 52,805.77，MAE 为 160.67。因此，这一步的收益来自对特征变换与核带宽的配套评价。

加入该分支后，原融合的 meta-CV MSE 从 53,052.57 降至 52,210.22，三个折均改善；验证 MSE 从 52,524.78 降至 52,201.21，对应 Kaggle 公榜 MSE 从 47,837.69 降至 47,291.35。

### 融合互补性

为评价各分支的作用，我们固定基础预测，在移除分支后重新拟合融合权重。移除 PE-G PCA 后，验证 MSE 增至 57,372.90；移除 NaFlex 后为 52,597.50。等权融合为 53,847.68，支持使用学习得到的权重。PE-L 的收益较小：移除后验证 MSE 略降至 52,194.81，但 meta-CV MSE 从 52,210.22 升至 52,244.24。因此，依据训练侧选择标准保留完整融合，其中 PE-G 提供主要预测能力。

### 误差分析与区域敏感性


![分析图](figures/report_prediction_scatter_price_band_errors.png)

最终融合在 1,601 张验证图像上的表现。（a）预测与真实价格，虚线为理想预测，颜色对应（b）的价位区间；（b）平均残差 $\hat y-y$；（c）样本与平方误差占比。价位区间为 $(0,300]$、$(300,700]$、$(700,1300]$ 和高于 1,300 千美元。


![分析图](figures/report_pe_g_occlusion_model_comparison.png)

三种 PE-G 回归流程在 $4\times4$ 网格模糊替换下的区域敏感性。两行分别为 1,543 张横向验证图像中补充整图后改善最大（4920）与退步最大（5712）的案例。红色表示替换使预测下降，蓝色表示上升；全部热图共用色标，图内标注真实价格 $y$ 与未扰动预测 $\hat y$，单位为千美元。



最终模型的 RMSE 为 228.48，绝对误差中位数为 113.41，平均残差为 $+6.51$。较小的整体偏差掩盖了不同价位相反的系统性误差（图 2）：不超过 300 的房屋平均残差为 $+93.72$，高于 1,300 的房屋则为 $-385.16$。159 张最高价位图像的 MAE 为 410.90，贡献总平方误差的 47.13%。误差也集中于少量样本：绝对误差最大的 161 张图像贡献总平方误差的 63.30%。

最后一步融合改进的收益并不均匀。最低和最高价位组对总体 MSE 变化的贡献分别为 $-224.23$ 和 $-882.25$，两个中间价位组则为 $+185.43$ 和 $+597.48$，合计 $-323.57$。因此，总体 MSE 改善并不意味着所有价位均获益。

极端误差案例体现了仅凭外观估价的困难。图像 5352 展示露台与远处海景，预测价格 1,679.82，明显高于真实价格 949.0。相反，图像 2138 展示木质外观房屋，真实价格为 1,995.0，却被预测为 605.27。这些案例表明，可见外观和环境对价值的指示可能存在歧义。

通过区域遮挡 [25] 考察敏感性，我们从 1,543 张横向验证图像中，选取补充整图后改善最大和退步最大的图像 4920 与 5712。将原图划分为 $4\times4$ 网格，分别用灰色像素或半径为 8 的高斯模糊内容替换各区域。图 3比较单视图、双视图和 PCA PE-G 模型的模糊替换响应，最大响应与房屋区域重叠。然而，图像 5712 的双视图模型在灰色与模糊替换下的相关系数仅为 0.267。因此，这些图描述的是对特定扰动的敏感性，难以据此可靠识别具体的价格相关属性。

## 讨论

三条路线分别回答了视觉信息应如何服务于房屋估价。手工 SIFT 特征使表示方式明确可控，为局部外观建立了有用参照。然而，视觉词袋汇总丢失空间关系，使其难以区分那些价值与建筑及周边环境的排列有关的房屋。这条路线揭示了图像证据的组织方式对估价的重要性。

从零训练 CNN 则让这种组织方式能够从数据中学习。受控实验展示了优化、增强和架构如何与有限的标注数据共同影响学习过程。紧凑预测头的表现提示，使模型容量与可用监督相匹配，可能比增加架构复杂度更有效。这些发现也提醒我们，更强的表达能力并不自动保证网络学到更适合房价预测的表示。

冻结预训练编码器引入了在房价标注数据之外学到的知识，使任务学习能够集中于回归。然而，这些知识仍需适配：视图构造决定哪些证据能够经过预处理保留下来，特征变换与核几何则决定视觉相似性如何转化为价格估计。特征缓存使这种分工能够在有限硬件上实现，但稠密核仍限制可扩展性。融合能够利用不同表示的互补性，各分支的作用则不必相同。

综合三条路线，更准确的视觉房价预测需要同时考虑表示的信息量、图像覆盖以及与估价任务相适应的映射。房屋外观提供的是价值的一部分线索；图像未揭示的位置、面积和交易条件仍可能带来估价歧义。后续研究应使空间与环境特征更直接地反映价格差异。由于路线比较同时改变了多个组件，并重复使用开发数据，还需通过独立的分组评价检验这些认识的泛化性。

## 参考文献


[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton [ImageNet classification with deep convolutional neural networks](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html). Advances in Neural Information Processing Systems, vol. 25, 2012.

[2] A. Radford et al. [Learning transferable visual models from natural language supervision](https://proceedings.mlr.press/v139/radford21a.html). Proc. ICML, 2021, pp. 8748–8763.

[3] M. Tschannen et al. [SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features](https://arxiv.org/abs/2502.14786). arXiv:2502.14786, 2025.

[4] D. Bolya et al. [Perception Encoder: The best visual embeddings are not at the output of the network](https://arxiv.org/abs/2504.13181). arXiv:2504.13181, 2025.

[5] D. G. Lowe [Object recognition from local scale-invariant features](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf). Proc. ICCV, 1999, pp. 1150–1157.

[6] D. G. Lowe [Distinctive image features from scale-invariant keypoints](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf). International Journal of Computer Vision, vol. 60, 2004, pp. 91–110.

[7] J. Sivic and A. Zisserman [Video Google: A text retrieval approach to object matching in videos](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/). Proc. ICCV, 2003, pp. 1470–1477.

[8] A. E. Hoerl and R. W. Kennard [Ridge regression: Biased estimation for nonorthogonal problems](https://doi.org/10.1080/00401706.1970.10488634). Technometrics, vol. 12, 1970, pp. 55–67.

[9] I. Loshchilov and F. Hutter [Decoupled weight decay regularization](https://arxiv.org/abs/1711.05101). Proc. ICLR, 2019.

[10] G. E. Hinton et al. [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580). arXiv:1207.0580, 2012.

[11] N. Srivastava et al. [Dropout: A simple way to prevent neural networks from overfitting](https://jmlr.org/papers/v15/srivastava14a.html). Journal of Machine Learning Research, vol. 15, 2014, pp. 1929–1958.

[12] D. P. Kingma and J. Ba [Adam: A method for stochastic optimization](https://arxiv.org/abs/1412.6980). Proc. ICLR, 2015.

[13] I. Loshchilov and F. Hutter [SGDR: Stochastic gradient descent with warm restarts](https://arxiv.org/abs/1608.03983). Proc. ICLR, 2017.

[14] X. Glorot and Y. Bengio [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html). Proc. AISTATS, 2010, pp. 249–256.

[15] K. He et al. [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html). Proc. ICCV, 2015, pp. 1026–1034.

[16] K. Simonyan and A. Zisserman [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/abs/1409.1556). Proc. ICLR, 2015.

[17] C. Szegedy et al. [Going deeper with convolutions](https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html). Proc. CVPR, 2015, pp. 1–9.

[18] K. He et al. [Deep residual learning for image recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_2016_CVPR_paper.html). Proc. CVPR, 2016, pp. 770–778.

[19] A. Vaswani et al. [Attention is all you need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html). Advances in Neural Information Processing Systems, vol. 30, 2017.

[20] C. Saunders, A. Gammerman, and V. Vovk [Ridge regression learning algorithm in dual variables](https://eprints.soton.ac.uk/258942/). Proc. ICML, 1998, pp. 515–521.

[21] H. Wold Estimation of principal components and related models by iterative least squares. Multivariate Analysis, P. R. Krishnaiah, Ed. Academic Press, 1966, pp. 391–420.

[22] S. Wold, A. Ruhe, H. Wold, and W. J. Dunn III [The collinearity problem in linear regression. The partial least squares (PLS) approach to generalized inverses](https://doi.org/10.1137/0905052). SIAM Journal on Scientific and Statistical Computing, vol. 5, 1984, pp. 735–743.

[23] K. Pearson [On lines and planes of closest fit to systems of points in space](https://doi.org/10.1080/14786440109462720). Philosophical Magazine, vol. 2, 1901, pp. 559–572.

[24] L. Breiman [Stacked regressions](https://doi.org/10.1007/BF00117832). Machine Learning, vol. 24, 1996, pp. 49–64.

[25] M. D. Zeiler and R. Fergus [Visualizing and understanding convolutional networks](https://arxiv.org/abs/1311.2901). Proc. ECCV, 2014, pp. 818–833.
