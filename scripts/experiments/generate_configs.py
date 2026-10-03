"""Generate the reviewed experiment catalog. Does not train or select winners."""
from pathlib import Path
from common.config import ROOT, write_json


def generate():
    folder = ROOT / "configs"
    base = {
        "experiment": {"id": "alexnet_baseline", "name": "神经网络基线阶段—实验一：AlexNet 房价回归", "stage": "神经网络基线",
                       "question": "从零训练是否能从图片学习房价信息", "comparison": "训练集平均房价与传统特征基线",
                       "changed_fields": [], "enforce_changes": False, "status": "ready"},
        "method": "neural", "seed": 2026, "device": "auto", "output_root": "../../output/experiment_record",
        "data": {"root": "../../comp-90086-2026/house_dataset", "split": "artifacts/split_v1.csv", "expected_train": 8000,
                 "expected_test": 3000, "split_seed": 2026, "validation_fraction": .2, "near_pairs": [["1078.jpg", "3568.jpg"]]},
        "preprocess": {"size": 224, "geometry": "letterbox", "normalization": "fixed", "filter": "none"},
        "augmentation": {},
        "model": {"pretrained": False, "small_kernels": False, "inception": False, "residual": False, "head": "alexnet",
                  "dropout": .5, "embedding_dim": 128, "attention_heads": 4, "feedforward_dim": 512, "encoder_dropout": .1, "encoder_layers": 1},
        "training": {"optimizer": "adamw", "lr": .0001, "momentum": .9, "schedule": "constant", "batch_size": 4,
                     "accumulation": 8, "epochs": 60, "workers": 2, "cpu_threads": 4, "weight_decay": .0001,
                     "penalty": "none", "penalty_coefficient": 0., "initialization": "default",
                     "early_stopping": {"enabled": False, "patience": 10, "relative_improvement": .001, "min_epochs": 15}},
        "target": {"transform": "scale", "scale": 1000., "loss": "mse", "huber_delta": .1},
        "traditional": {"words": 256, "long_edge": 320, "max_keypoints": 256, "per_image_sample": 128, "max_descriptors": 200000,
                        "spatial_pyramid": False, "appearance": "none", "regressor": "ridge", "ridge_alpha": 10.},
    }
    write_json(folder / "base.json", base)
    catalog = []
    def add(stage, filename, name, parent, patch, changes, question, comparison=None, status="ready"):
        cfg = {"extends": parent, "experiment": {"id": filename.replace('.', 'p'), "name": name, "stage": stage, "question": question,
                "comparison": comparison or f"继承配置 {parent} 对应的已保留方案", "changed_fields": changes,
                "enforce_changes": True, "status": status}, **patch}
        path = folder / stage / f"{filename}.json"
        write_json(path, cfg)
        catalog.append((path.relative_to(ROOT).as_posix(), name, parent, question))
    add("00_preparation", "prepare_split", "实验前准备：数据分组与冻结划分", "../base.json", {}, [], "避免重复图片跨训练验证侧")
    prep = [("stretch", "拉伸尺寸调整", {"geometry": "stretch"}), ("crop", "中心裁剪", {"geometry": "center_crop"}),
            ("resolution", "提高输入分辨率", {"size": 320}), ("normalize", "训练侧颜色标准化", {"normalization": "train_stats"}),
            ("spatial_filter", "空间低通滤波", {"filter": "spatial"}), ("frequency_filter", "频率低通滤波", {"filter": "frequency"})]
    numbers = list("一二三四五六七八九十")
    for i, (slug, title, patch) in enumerate(prep):
        add("00_preparation", f"preprocess_{i+1:02}_{slug}", f"预处理备选阶段—实验{numbers[i]}：{title}", "../base.json",
            {"preprocess": patch}, ["preprocess." + k for k in patch], "输入诊断提供依据后，固定 AlexNet 训练设置比较该输入因素")
    add("01_baselines", "mean_price", "基础基线：训练集平均房价预测", "../base.json", {"method": "mean"}, ["method"], "看图片是否优于直接猜训练侧均价")
    add("01_baselines", "alexnet", "神经网络基线阶段—实验一：AlexNet 房价回归", "../base.json", {}, [], "建立无增强从零网络参照")
    add("01_baselines", "sift_ridge", "传统参照阶段—实验一：局部特征与房价回归", "../base.json", {"method": "traditional"}, ["method"], "局部纹理是否能提供价格信息")
    traditional = [(2, "spatial", "空间金字塔", {"spatial_pyramid": True}), (3, "color", "颜色统计补充", {"appearance": "color"}),
                   (4, "svr", "回归器替换", {"regressor": "svr"}), (5, "vocabulary", "视觉词典大小", {"words": 512}),
                   (6, "shading", "明暗统计补充", {"appearance": "shading"}), (7, "gradient", "梯度结构补充", {"appearance": "gradient"})]
    for num, slug, title, patch in traditional:
        add("01_baselines", f"traditional_{num:02}_{slug}", f"传统参照阶段—实验{numbers[num-1]}：{title}", "sift_ridge.json", {"traditional": patch},
            ["traditional." + k for k in patch], "只替换或追加登记的特征因素，其他条件一致")
    # Each stage reads the explicitly selected prior result, rather than assuming a winner.
    for lr in [3e-5, 1e-4, 3e-4]:
        add("02_training", f"tune_01_lr_{lr:g}", "训练调优阶段—实验一：固定学习率大小比较", "../selected/input.json",
            {"training": {"lr": lr}}, ["training.lr"], "全程固定学习率，比较更新幅度")
    for lr in [.001, .01, .03]:
        add("02_training", f"tune_02_sgd_{lr:g}", "训练调优阶段—实验二：优化器选择", "../selected/input.json",
            {"training": {"optimizer": "sgd", "lr": lr}}, ["training.optimizer", "training.lr"], "与三个 AdamW 学习率候选比较，两个优化器各三个候选")
    add("02_training", "tune_03_cosine", "训练调优阶段—实验三：学习率变化策略比较", "../selected/optimizer.json",
        {"training": {"schedule": "cosine"}}, ["training.schedule"], "固定起点学习率，比较全程固定与逐轮衰减")
    for accumulation in [4, 8, 16]:
        add("02_training", f"tune_04_batch_{4*accumulation}", "训练调优阶段—实验四：批量大小", "../selected/schedule.json",
            {"training": {"accumulation": accumulation}}, ["training.accumulation"], "实际批量固定四，改变有效批量，不自动缩放学习率")
    for p in [0., .2, .5]:
        add("02_training", f"tune_05_dropout_{p:g}", "训练调优阶段—实验五：随机失活概率", "../selected/batch.json",
            {"model": {"dropout": p}}, ["model.dropout"], "原全连接头位置不变，只改两处概率")
    add("02_training", "tune_06_no_penalty", "训练调优阶段—实验六：权重正则化", "../selected/dropout.json",
        {"training": {"weight_decay": 0., "penalty": "none", "penalty_coefficient": 0.}},
        ["training.weight_decay", "training.penalty", "training.penalty_coefficient"], "建立显式惩罚和衰减关闭的共同对照")
    for penalty, coeffs in [("l1", [1e-7, 1e-6]), ("l2", [1e-6, 1e-5])]:
        for coeff in coeffs:
            add("02_training", f"tune_06_{penalty}_{coeff:g}", "训练调优阶段—实验六：权重正则化", "tune_06_no_penalty.json",
                {"training": {"penalty": penalty, "penalty_coefficient": coeff}}, ["training.penalty", "training.penalty_coefficient"], "显式惩罚按权重求和，优化器衰减关闭")
    for wd in [1e-5, 1e-4, 1e-3]:
        add("02_training", f"tune_06_decay_{wd:g}", "训练调优阶段—实验六：权重正则化", "tune_06_no_penalty.json",
            {"training": {"weight_decay": wd}}, ["training.weight_decay"], "仅当已选优化器为 AdamW 才执行解耦衰减候选")
    add("02_training", "tune_07_early_stop", "训练调优阶段—实验七：提前停止", "../selected/regularization.json",
        {"training": {"early_stopping": {"enabled": True}}}, ["training.early_stopping.enabled"], "评价是否节省成本及错过后期改善")
    add("02_training", "tune_08_initialization", "训练调优阶段—实验八：初始化检查", "../selected/training.json",
        {"training": {"initialization": "xavier_convolution"}}, ["training.initialization"], "仅在初始化诊断后登记卷积 Xavier 与原 Kaiming 对照", status="design_required")
    ops = [("flip", "水平翻转", .5, None), ("brightness", "亮度变化", 1., [.9, 1.1]),
           ("contrast", "对比度变化", 1., [.9, 1.1]), ("saturation", "饱和度变化", .5, [.9, 1.1]),
           ("rotation", "小角度旋转", .3, [-3, 3]), ("translation", "小幅平移", .3, [-.03, .03]),
           ("crop", "轻度裁剪", .3, [.9, 1.]), ("blur", "高斯模糊", .2, [.3, .8]),
           ("noise", "高斯噪声", .2, None), ("erase", "局部擦除", .15, [.01, .03])]
    for i, (key, title, probability, span) in enumerate(ops):
        op = {"probability": probability}
        if span:
            op["range"] = span
        if key == "noise":op["std"] = .01
        if key == "erase":op["aspect"] = [.5, 2.]
        add("03_augmentation", f"augment_{i+1:02}_{key}", f"增强阶段—实验{numbers[i]}：{title}", "../selected/training.json",
            {"augmentation": {key: op}}, ["augmentation"], "相对同一调优后无增强 AlexNet，仅开启本项操作")
    add("03_augmentation", "augment_11_combination", "增强阶段—实验十一：有效增强组合", "../selected/training.json",
        {"augmentation": {"flip": {"probability": .5}, "brightness": {"probability": 1., "range": [.9, 1.1]}}},
        ["augmentation"], "这里是示例组合，必须根据两个有效单项登记实际组合后将 status 改为 ready", status="design_required")
    for i, (key, title, parent) in enumerate([("small_kernels", "小卷积核堆叠", "augmentation"), ("inception", "多尺度卷积", "small"), ("residual", "残差连接", "multi")]):
        add("04_architecture", f"structure_{i+1:02}_{key}", f"结构改进阶段—实验{numbers[i]}：{title}", f"../selected/{parent}.json",
            {"model": {key: True}}, ["model." + key], "只继承已保留改动，共享训练设置一致，全部从零训练")
    add("05_encoder", "encoder_01_mean_head", "卷积特征处理阶段—实验一：直接汇总特征预测房价", "../selected/convolution.json",
        {"model": {"head": "mean"}}, ["model.head"], "先区分替换大型全连接头的影响，不使用编码器")
    add("05_encoder", "encoder_02_one_layer", "卷积特征处理阶段—实验二：加入单层 Transformer 编码器", "../selected/head.json",
        {"model": {"head": "encoder", "encoder_layers": 1}}, ["model.head", "model.encoder_layers"], "保留汇总及读出方式，比较有无编码器")
    add("05_encoder", "encoder_03_two_layers", "卷积特征处理阶段—实验三：比较一层与两层编码器", "../selected/encoder.json",
        {"model": {"encoder_layers": 2}}, ["model.encoder_layers"], "单层有效后只增加编码器深度")
    lines = ["# 实验配置索引", "", "由 generate_configs.py 生成；重新生成会覆盖候选配置，请先保存人工修改。selected 目录不会被生成器写入。", "", "| 配置文件 | 中文实验名称 | 继承配置 | 研究问题 |", "| --- | --- | --- | --- |"]
    for path, name, parent, question in catalog:
        lines.append(f"| [{path}]({path.removeprefix('configs/')}) | {name} | {parent} | {question} |")
    (folder / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Generated {len(catalog)} experiment configurations")


if __name__ == "__main__":
    generate()
