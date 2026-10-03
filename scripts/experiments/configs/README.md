# 实验配置索引

由 generate_configs.py 生成；重新生成会覆盖候选配置，请先保存人工修改。selected 目录不会被生成器写入。

| 配置文件 | 中文实验名称 | 继承配置 | 研究问题 |
| --- | --- | --- | --- |
| [configs/00_preparation/prepare_split.json](00_preparation/prepare_split.json) | 实验前准备：数据分组与冻结划分 | ../base.json | 避免重复图片跨训练验证侧 |
| [configs/00_preparation/preprocess_01_stretch.json](00_preparation/preprocess_01_stretch.json) | 预处理备选阶段—实验一：拉伸尺寸调整 | ../base.json | 输入诊断提供依据后，固定 AlexNet 训练设置比较该输入因素 |
| [configs/00_preparation/preprocess_02_crop.json](00_preparation/preprocess_02_crop.json) | 预处理备选阶段—实验二：中心裁剪 | ../base.json | 输入诊断提供依据后，固定 AlexNet 训练设置比较该输入因素 |
| [configs/00_preparation/preprocess_03_resolution.json](00_preparation/preprocess_03_resolution.json) | 预处理备选阶段—实验三：提高输入分辨率 | ../base.json | 输入诊断提供依据后，固定 AlexNet 训练设置比较该输入因素 |
| [configs/00_preparation/preprocess_04_normalize.json](00_preparation/preprocess_04_normalize.json) | 预处理备选阶段—实验四：训练侧颜色标准化 | ../base.json | 输入诊断提供依据后，固定 AlexNet 训练设置比较该输入因素 |
| [configs/00_preparation/preprocess_05_spatial_filter.json](00_preparation/preprocess_05_spatial_filter.json) | 预处理备选阶段—实验五：空间低通滤波 | ../base.json | 输入诊断提供依据后，固定 AlexNet 训练设置比较该输入因素 |
| [configs/00_preparation/preprocess_06_frequency_filter.json](00_preparation/preprocess_06_frequency_filter.json) | 预处理备选阶段—实验六：频率低通滤波 | ../base.json | 输入诊断提供依据后，固定 AlexNet 训练设置比较该输入因素 |
| [configs/01_baselines/mean_price.json](01_baselines/mean_price.json) | 基础基线：训练集平均房价预测 | ../base.json | 看图片是否优于直接猜训练侧均价 |
| [configs/01_baselines/alexnet.json](01_baselines/alexnet.json) | 神经网络基线阶段—实验一：AlexNet 房价回归 | ../base.json | 建立无增强从零网络参照 |
| [configs/01_baselines/sift_ridge.json](01_baselines/sift_ridge.json) | 传统参照阶段—实验一：局部特征与房价回归 | ../base.json | 局部纹理是否能提供价格信息 |
| [configs/01_baselines/traditional_02_spatial.json](01_baselines/traditional_02_spatial.json) | 传统参照阶段—实验二：空间金字塔 | sift_ridge.json | 只替换或追加登记的特征因素，其他条件一致 |
| [configs/01_baselines/traditional_03_color.json](01_baselines/traditional_03_color.json) | 传统参照阶段—实验三：颜色统计补充 | sift_ridge.json | 只替换或追加登记的特征因素，其他条件一致 |
| [configs/01_baselines/traditional_04_svr.json](01_baselines/traditional_04_svr.json) | 传统参照阶段—实验四：回归器替换 | sift_ridge.json | 只替换或追加登记的特征因素，其他条件一致 |
| [configs/01_baselines/traditional_05_vocabulary.json](01_baselines/traditional_05_vocabulary.json) | 传统参照阶段—实验五：视觉词典大小 | sift_ridge.json | 只替换或追加登记的特征因素，其他条件一致 |
| [configs/01_baselines/traditional_06_shading.json](01_baselines/traditional_06_shading.json) | 传统参照阶段—实验六：明暗统计补充 | sift_ridge.json | 只替换或追加登记的特征因素，其他条件一致 |
| [configs/01_baselines/traditional_07_gradient.json](01_baselines/traditional_07_gradient.json) | 传统参照阶段—实验七：梯度结构补充 | sift_ridge.json | 只替换或追加登记的特征因素，其他条件一致 |
| [configs/02_training/tune_01_lr_3e-05.json](02_training/tune_01_lr_3e-05.json) | 训练调优阶段—实验一：固定学习率大小比较 | ../selected/input.json | 全程固定学习率，比较更新幅度 |
| [configs/02_training/tune_01_lr_0.0001.json](02_training/tune_01_lr_0.0001.json) | 训练调优阶段—实验一：固定学习率大小比较 | ../selected/input.json | 全程固定学习率，比较更新幅度 |
| [configs/02_training/tune_01_lr_0.0003.json](02_training/tune_01_lr_0.0003.json) | 训练调优阶段—实验一：固定学习率大小比较 | ../selected/input.json | 全程固定学习率，比较更新幅度 |
| [configs/02_training/tune_02_sgd_0.001.json](02_training/tune_02_sgd_0.001.json) | 训练调优阶段—实验二：优化器选择 | ../selected/input.json | 与三个 AdamW 学习率候选比较，两个优化器各三个候选 |
| [configs/02_training/tune_02_sgd_0.01.json](02_training/tune_02_sgd_0.01.json) | 训练调优阶段—实验二：优化器选择 | ../selected/input.json | 与三个 AdamW 学习率候选比较，两个优化器各三个候选 |
| [configs/02_training/tune_02_sgd_0.03.json](02_training/tune_02_sgd_0.03.json) | 训练调优阶段—实验二：优化器选择 | ../selected/input.json | 与三个 AdamW 学习率候选比较，两个优化器各三个候选 |
| [configs/02_training/tune_03_cosine.json](02_training/tune_03_cosine.json) | 训练调优阶段—实验三：学习率变化策略比较 | ../selected/optimizer.json | 固定起点学习率，比较全程固定与逐轮衰减 |
| [configs/02_training/tune_04_batch_16.json](02_training/tune_04_batch_16.json) | 训练调优阶段—实验四：批量大小 | ../selected/schedule.json | 实际批量固定四，改变有效批量，不自动缩放学习率 |
| [configs/02_training/tune_04_batch_32.json](02_training/tune_04_batch_32.json) | 训练调优阶段—实验四：批量大小 | ../selected/schedule.json | 实际批量固定四，改变有效批量，不自动缩放学习率 |
| [configs/02_training/tune_04_batch_64.json](02_training/tune_04_batch_64.json) | 训练调优阶段—实验四：批量大小 | ../selected/schedule.json | 实际批量固定四，改变有效批量，不自动缩放学习率 |
| [configs/02_training/tune_05_dropout_0.json](02_training/tune_05_dropout_0.json) | 训练调优阶段—实验五：随机失活概率 | ../selected/batch.json | 原全连接头位置不变，只改两处概率 |
| [configs/02_training/tune_05_dropout_0.2.json](02_training/tune_05_dropout_0.2.json) | 训练调优阶段—实验五：随机失活概率 | ../selected/batch.json | 原全连接头位置不变，只改两处概率 |
| [configs/02_training/tune_05_dropout_0.5.json](02_training/tune_05_dropout_0.5.json) | 训练调优阶段—实验五：随机失活概率 | ../selected/batch.json | 原全连接头位置不变，只改两处概率 |
| [configs/02_training/tune_06_no_penalty.json](02_training/tune_06_no_penalty.json) | 训练调优阶段—实验六：权重正则化 | ../selected/dropout.json | 建立显式惩罚和衰减关闭的共同对照 |
| [configs/02_training/tune_06_l1_1e-07.json](02_training/tune_06_l1_1e-07.json) | 训练调优阶段—实验六：权重正则化 | tune_06_no_penalty.json | 显式惩罚按权重求和，优化器衰减关闭 |
| [configs/02_training/tune_06_l1_1e-06.json](02_training/tune_06_l1_1e-06.json) | 训练调优阶段—实验六：权重正则化 | tune_06_no_penalty.json | 显式惩罚按权重求和，优化器衰减关闭 |
| [configs/02_training/tune_06_l2_1e-06.json](02_training/tune_06_l2_1e-06.json) | 训练调优阶段—实验六：权重正则化 | tune_06_no_penalty.json | 显式惩罚按权重求和，优化器衰减关闭 |
| [configs/02_training/tune_06_l2_1e-05.json](02_training/tune_06_l2_1e-05.json) | 训练调优阶段—实验六：权重正则化 | tune_06_no_penalty.json | 显式惩罚按权重求和，优化器衰减关闭 |
| [configs/02_training/tune_06_decay_1e-05.json](02_training/tune_06_decay_1e-05.json) | 训练调优阶段—实验六：权重正则化 | tune_06_no_penalty.json | 仅当已选优化器为 AdamW 才执行解耦衰减候选 |
| [configs/02_training/tune_06_decay_0.0001.json](02_training/tune_06_decay_0.0001.json) | 训练调优阶段—实验六：权重正则化 | tune_06_no_penalty.json | 仅当已选优化器为 AdamW 才执行解耦衰减候选 |
| [configs/02_training/tune_06_decay_0.001.json](02_training/tune_06_decay_0.001.json) | 训练调优阶段—实验六：权重正则化 | tune_06_no_penalty.json | 仅当已选优化器为 AdamW 才执行解耦衰减候选 |
| [configs/02_training/tune_07_early_stop.json](02_training/tune_07_early_stop.json) | 训练调优阶段—实验七：提前停止 | ../selected/regularization.json | 评价是否节省成本及错过后期改善 |
| [configs/02_training/tune_08_initialization.json](02_training/tune_08_initialization.json) | 训练调优阶段—实验八：初始化检查 | ../selected/training.json | 仅在初始化诊断后登记卷积 Xavier 与原 Kaiming 对照 |
| [configs/03_augmentation/augment_01_flip.json](03_augmentation/augment_01_flip.json) | 增强阶段—实验一：水平翻转 | ../selected/training.json | 相对同一调优后无增强 AlexNet，仅开启本项操作 |
| [configs/03_augmentation/augment_02_brightness.json](03_augmentation/augment_02_brightness.json) | 增强阶段—实验二：亮度变化 | ../selected/training.json | 相对同一调优后无增强 AlexNet，仅开启本项操作 |
| [configs/03_augmentation/augment_03_contrast.json](03_augmentation/augment_03_contrast.json) | 增强阶段—实验三：对比度变化 | ../selected/training.json | 相对同一调优后无增强 AlexNet，仅开启本项操作 |
| [configs/03_augmentation/augment_04_saturation.json](03_augmentation/augment_04_saturation.json) | 增强阶段—实验四：饱和度变化 | ../selected/training.json | 相对同一调优后无增强 AlexNet，仅开启本项操作 |
| [configs/03_augmentation/augment_05_rotation.json](03_augmentation/augment_05_rotation.json) | 增强阶段—实验五：小角度旋转 | ../selected/training.json | 相对同一调优后无增强 AlexNet，仅开启本项操作 |
| [configs/03_augmentation/augment_06_translation.json](03_augmentation/augment_06_translation.json) | 增强阶段—实验六：小幅平移 | ../selected/training.json | 相对同一调优后无增强 AlexNet，仅开启本项操作 |
| [configs/03_augmentation/augment_07_crop.json](03_augmentation/augment_07_crop.json) | 增强阶段—实验七：轻度裁剪 | ../selected/training.json | 相对同一调优后无增强 AlexNet，仅开启本项操作 |
| [configs/03_augmentation/augment_08_blur.json](03_augmentation/augment_08_blur.json) | 增强阶段—实验八：高斯模糊 | ../selected/training.json | 相对同一调优后无增强 AlexNet，仅开启本项操作 |
| [configs/03_augmentation/augment_09_noise.json](03_augmentation/augment_09_noise.json) | 增强阶段—实验九：高斯噪声 | ../selected/training.json | 相对同一调优后无增强 AlexNet，仅开启本项操作 |
| [configs/03_augmentation/augment_10_erase.json](03_augmentation/augment_10_erase.json) | 增强阶段—实验十：局部擦除 | ../selected/training.json | 相对同一调优后无增强 AlexNet，仅开启本项操作 |
| [configs/03_augmentation/augment_11_combination.json](03_augmentation/augment_11_combination.json) | 增强阶段—实验十一：有效增强组合 | ../selected/training.json | 这里是示例组合，必须根据两个有效单项登记实际组合后将 status 改为 ready |
| [configs/04_architecture/structure_01_small_kernels.json](04_architecture/structure_01_small_kernels.json) | 结构改进阶段—实验一：小卷积核堆叠 | ../selected/augmentation.json | 只继承已保留改动，共享训练设置一致，全部从零训练 |
| [configs/04_architecture/structure_02_inception.json](04_architecture/structure_02_inception.json) | 结构改进阶段—实验二：多尺度卷积 | ../selected/small.json | 只继承已保留改动，共享训练设置一致，全部从零训练 |
| [configs/04_architecture/structure_03_residual.json](04_architecture/structure_03_residual.json) | 结构改进阶段—实验三：残差连接 | ../selected/multi.json | 只继承已保留改动，共享训练设置一致，全部从零训练 |
| [configs/05_encoder/encoder_01_mean_head.json](05_encoder/encoder_01_mean_head.json) | 卷积特征处理阶段—实验一：直接汇总特征预测房价 | ../selected/convolution.json | 先区分替换大型全连接头的影响，不使用编码器 |
| [configs/05_encoder/encoder_02_one_layer.json](05_encoder/encoder_02_one_layer.json) | 卷积特征处理阶段—实验二：加入单层 Transformer 编码器 | ../selected/head.json | 保留汇总及读出方式，比较有无编码器 |
| [configs/05_encoder/encoder_03_two_layers.json](05_encoder/encoder_03_two_layers.json) | 卷积特征处理阶段—实验三：比较一层与两层编码器 | ../selected/encoder.json | 单层有效后只增加编码器深度 |
