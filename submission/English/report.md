# House Price Prediction with Frozen Visual Features and Kernel Fusion

## Introduction

Estimating a house's sale price involves size, condition, location and market conditions. Exterior photographs reveal some of this information: computer vision can describe architectural form, surface appearance, landscaping and surrounding scenes. However, a single view provides incomplete coverage and does not directly reveal floor area, an exact address or transaction conditions. We therefore investigate how to extract more informative visual features from house photographs to predict sale prices more accurately.

Visual representations offer several approaches. Handcrafted descriptors summarize local gradients and textures, while networks such as AlexNet [1] learn hierarchical features. Large-scale pretraining further enables transfer across tasks: CLIP [2], SigLIP2 [3] and Perception Encoder [4] motivate freezing the visual encoder and learning price regression separately.

Our study follows three stages, each testing a limitation exposed by the preceding experiments. SIFT with a visual vocabulary first tests whether local appearance predicts price. An AlexNet-style regressor then undergoes controlled training, augmentation and architecture experiments. Its remaining error motivates frozen pretrained features, for which we investigate image coverage, feature transformation and regression before combining complementary branches.

The selected three-encoder fusion achieves validation MSE 52,201.21 and MAE 159.58, with public Kaggle MSE 47,291.35 after full-data refitting. Our analysis relates these gains to representation and kernel design, while identifying persistent high-price underestimation and uneven benefits across price groups.

## Methods

### Data, Metrics and Evaluation Protocol

The dataset contains 8,000 labelled training images and 3,000 unlabelled test images from Southern California real-estate listings. Prices are measured in thousand USD. We fit the models using the supplied images and prices, with general-purpose pretrained image encoders providing transferable visual features.

We split the data into 6,399 training and 1,601 validation images with seed 2026, keeping identified duplicate or related-image groups together. For $N$ evaluated images with true prices $y_i$ and predictions $\hat y_i$, we report

$$

\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2,\qquad
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|\hat y_i-y_i|.

$$

MSE, the primary metric, emphasizes large errors; MAE measures average absolute error. Their units are (thousand USD)$^2$ and thousand USD, respectively. Signed residuals $\hat y_i-y_i$ describe bias.

Stochastic models are screened with seed 2026; promising candidates are repeated with seeds 2027 and 2028. Adoption requires lower mean MSE and at least two seed-matched wins; we report means and sample standard deviations. Frozen-feature regressors use grouped three-fold training out-of-fold (OOF) selection, fitting codebooks, scalers, PCA and supervised transforms within each fold.

### Baselines and Controlled CNN Development

We first predict the training mean as a non-visual reference. To determine whether local appearance adds information, we extract grayscale SIFT descriptors [5, 6] and build a 256-word visual vocabulary [7]. Histograms undergo L1--square-root--L2 normalization and standardization before Ridge regression [8] ($\alpha=10$). Although this captures local gradient structure, pooling into a histogram discards spatial arrangement. We therefore also examine a CNN that learns feature composition directly from images.

Our AlexNet-style regressor [1] has five convolutional layers and a 9216--4096--4096--1 head, initialized from scratch. After preprocessing comparisons, the selected input is a direct $224\times224$ resize, RGB normalization with mean and SD 0.5, and no filtering. The retained configuration uses AdamW [9] ($10^{-4}$ learning rate and weight decay), dropout [10, 11] 0.5, and effective batch 32 through microbatch 4 and eight-step accumulation. Training uses prices divided by 1,000; predictions are restored to dataset units. Runs last at most 60 epochs, with a minimum of 40 and patience 15 under a 0.1% improvement stopping rule; the checkpoint with strictly lowest validation MSE is retained.

To test optimization sensitivity, representative screens use learning rates $3\times10^{-5}$, $10^{-4}$, $3\times10^{-4}$ and $10^{-3}$, effective batches 16/32/64 and dropout 0/0.2/0.5. We compare optimizers including Adam [12], cosine decay [13] without restarts, effective batch size, regularization and initialization [14, 15], then test augmentation. Geometric and degradation transforms are evaluated separately; horizontal flip uses probability 0.5. Architecture experiments test stacked $3\times3$ convolutions [16] replacing the second $5\times5$ layer, an Inception-style multi-scale block [17], and a projected residual connection [18]. To test whether the dense head overfits limited supervision, we project the $6\times6$ convolutional grid to 128 channels, add positional information and normalization, and average the 36 tokens before linear regression. We then test a one-layer Transformer [19] with dimension 128, four heads and a 512-dimensional feed-forward layer. Each change is evaluated against its corresponding control.

In a separate seed-2026 screen, brightness and contrast are each multiplied by a factor sampled uniformly from $[0.9,1.1]$ on every training image before resizing and normalization. Both are compared with a freshly trained unaugmented AlexNet using the same initialization, split and training settings.

### Frozen Representations, Views and Regression

To extend the scratch models, we decouple visual representation from price regression. Frozen CLIP ViT-B/32 features with standardized Ridge regression establish the transfer-learning reference; the final branches use SigLIP2 So400m NaFlex, PE-Core-L/14 and PE-Core-G/14. NaFlex preserves native aspect ratio with 576 patches and produces 1,152-dimensional features. Because a single crop can remove useful context, PE-L combines three 1,024-dimensional views: the official input, a 336-pixel gray-padded whole frame, and a central crop spanning 60% of image width and height. PE-G combines its official 448-pixel view with a gray-padded whole frame. Features are individually unit-normalized, concatenated and normalized again, yielding 3,072 PE-L and 2,560 PE-G dimensions.

We use RBF kernel ridge regression [20] to model nonlinear relationships in the frozen feature space, with regularization controlling the coefficients. For unit features, squared distance equals twice one minus cosine similarity, so $\gamma$ controls how locally visual neighbors influence price. For $x$,

$$

 K_{ij}=\exp(-\gamma\|x_i-x_j\|^2),\quad
 c=(K+\alpha I)^{-1}(y-\bar y\mathbf{1}),

$$

with prediction $\bar y+k(x)^Tc$ and non-negative clipping. Ordinary kernel searches use $\gamma\in\{1,4,16\}$ and $\alpha\in\{0.01,0.1,1,10\}$ on training folds. NaFlex selects $\gamma=1$, $\alpha=0.1$.

However, generic visual similarity may emphasize variation unrelated to price. PE-L therefore uses fold-fitted PLS [21, 22] to learn price-related directions. Sixteen components are fitted; the selected head uses the first four standardized scores $s$ and the product kernel

$$

 k(x,x')=\exp\left(-\|x-x'\|^2-0.1\frac{\|s-s'\|^2}{4}\right),

$$

with $\alpha=1$. The PLS scores introduce price-related directions into the similarity measure while keeping the image encoder frozen.

In contrast to supervised PLS, PCA [23] compresses features along their principal variation directions without using prices. For PE-G, we test 64, 256 and 1,024 dimensions with whitening powers 0 and 0.5, followed by unit normalization. Since this changes distances, we also recalibrate the original $\gamma=4$:

$$

 \gamma'=4\frac{\operatorname{median}\|x_i-x_j\|^2}
 {\operatorname{median}\|z_i-z_j\|^2},

$$

using 8,192 fixed non-self pairs drawn only from fitting-fold images. Here $z$ is the transformed unit vector. The selected transform is non-whitened PCA256 with $\alpha=1$.

### Fusion and Final Refit

Following stacked regression [24], we combine branch predictions using non-negative weights, additionally constrained to sum to one. Weights are fitted to training OOF predictions and evaluated by meta cross-fitting: each fold is predicted using weights fitted on the remaining folds. A new branch is retained when meta-CV MSE improves by at least 1%, at least two of three folds improve, and validation MSE also decreases.

The final weights are NaFlex 0.108065, PE-L PLS 0.080806 and PE-G PCA 0.811129. For the Kaggle submission, we retain the selected hyperparameters and weights and refit the transforms and regression heads on all 8,000 labelled images, recomputing PCA and bandwidth statistics from this training set. The fitted model then predicts the 3,000 test prices.

## Results

| Method | Seeds | Validation MSE | Validation MAE |
|---|:---:|---:|---:|
| AlexNet (no augmentation) | 3 | 127,774 ± 486 | 264.52 |
| AlexNet + horizontal flip | 3 | 126,810 ± 1,902 | 268.02 |
| Flip AlexNet + mean-pooling head | 3 | 122,334 ± 1,589 | 261.48 |
| Frozen CLIP B32 + Ridge | — | 91,821 | 223.27 |
| NaFlex + RBF | — | 63,086 | 181.28 |
| PE-L, three views + RBF | — | 59,213 | 172.90 |
| PE-L, three views + PLS/RBF | — | 59,483 | 171.73 |
| PE-G, official view + RBF | — | 55,097 | 165.80 |
| PE-G, two views + RBF | — | 53,253 | 161.88 |
| PE-G, two views + PCA/RBF | — | 52,806 | 160.67 |
| Selected three-encoder fusion | — | 52,201 | 159.58 |

Table 1. Validation n = 1,601. Three seeds: MSE mean ± sample SD and MAE mean; one seed: seed 2026; —: frozen-feature evaluation.





### Local Evidence and the Need for Learned Representations

The training-mean predictor obtains MSE 151,839.69. SIFT--BoVW--Ridge reduces it to $145,336.73\pm1,091.58$, establishing a limited local-appearance signal. Since visual-word pooling discards spatial arrangement, we next test learned feature composition. The padded AlexNet obtains mean MSE 128,415.68; direct resizing reaches 127,774.42 (Table~1). This improvement motivates examining whether training and capacity, rather than representation alone, explain its remaining error.

### Testing Training and Capacity Before Transfer

Seed-matched trials provide little support for optimization changes. Adam at learning rate 0.001 improves seed 2026 but loses the other two comparisons, increasing mean MSE by 509.89; cosine decay, alternative batches and lower dropout also fail adoption. Horizontal flip instead reduces mean MSE by 964.41 with two paired wins, although MAE rises from 264.52 to 268.02.

To test whether the large prediction head limits learning, we replace it with mean pooling. All three seeds improve, reducing MSE by 4,475.78 relative to flip AlexNet. Small-kernel, multi-scale and residual screens fail their controls; the Transformer increases single-seed MSE by 3,615.11 over the compact head. Greater capacity therefore does not resolve this setting's prediction difficulty.

Photometric changes reveal a further trade-off. Against a fresh seed-2026 control (MSE 131,389.94, MAE 282.64), brightness gives 131,433.91 and 274.50; contrast gives 128,860.53 and 269.01. Contrast improves overall MSE by 1.93% but worsens the above-1,300 tail by 17.14%; brightness worsens it by 3.86%. Thus, lower average error need not help expensive properties. Both remain single-seed screens.

The retained scratch model still has substantial error. Frozen CLIP with Ridge reaches MSE 91,820.73, motivating transfer of richer visual representations. This is a complete-route comparison, with both representation and readout changing.

### Recovering Context and Repairing Kernel Geometry


![Report figure](report_source/figures/report_pca_distance_bandwidth_oof.png)

PE-G geometry and regression. (a,b) Empirical CDFs of distances and kernel similarities for 8,192 training pairs; (c) grouped three-fold OOF MSE for non-whitened PCA. Transforms and bandwidths are fitted within each fold. Panel (b) shows the similarity range 0--0.4.

Supplementing PE-G's official view with the whole frame reduces validation MSE from 55,097.49 to 53,252.81 and OOF MSE from 55,679.73 to 53,646.46, supporting broader coverage.

However, compressing these features with PCA256 and fixed $\gamma=4$ worsens OOF MSE to 77,771.43. The training median squared distance changes from 0.5402 to 2.0515, so we recalibrate $\gamma$ to 1.05323 (Fig.~1). OOF MSE falls to 52,531.27; validation MSE reaches 52,805.77 and MAE 160.67. The failed fixed-bandwidth experiment therefore identifies a distance-scale problem that must be addressed alongside compression.

Supervised adaptation is also not uniformly beneficial: PE-L PLS slightly worsens validation MSE relative to its three-view RBF control, while improving MAE. Its usefulness must consequently be judged through complementarity, rather than assumed from price supervision.

### Complementarity and Remaining Valuation Ambiguity


![Report figure](report_source/figures/report_prediction_scatter_price_band_errors.png)

Selected fusion on 1,601 validation images. (a) Predicted versus true price; the dashed line is perfect prediction and colors follow the price bands in (b). (b) Mean residual $\hat y-y$. (c) Image and squared-error shares. Bands are $(0,300]$, $(300,700]$, $(700,1300]$ and above 1,300 thousand USD.


![Report figure](report_source/figures/report_regional_and_outcome_cases.png)

Left four columns: PE-G regional sensitivity under $4\times4$ blur replacement; red indicates a prediction decrease, blue an increase, on one shared scale. Cases 4920/5712 are the largest improvement/worsening after adding the whole frame among 1,543 landscape validation images. Right column: separate final-fusion cases, 518 at the 25th absolute-error percentile and retained failure 2138. Prices are in thousand USD.

Adding PE-G PCA lowers meta-CV MSE from 53,052.57 to 52,210.22 with all three folds improving, and validation MSE from 52,524.78 to 52,201.21. Public Kaggle MSE changes from 47,837.69 to 47,291.35. Removing PE-G PCA raises validation MSE to 57,372.90; removing NaFlex gives 52,597.50; equal weights give 53,847.68. Without PE-L, validation MSE slightly improves to 52,194.81, but meta-CV worsens to 52,244.24. Its retained contribution is therefore marginal and criterion-dependent.

Aggregate success leaves a new problem: mean residual $+6.51$ masks low-price overestimation ($+93.72$ at prices up to 300) and high-price underestimation ($-385.16$ above 1,300). The 159 highest-price images contribute 47.13% of squared error (Fig.~2); the 161 largest errors contribute 63.30%. RMSE is 228.48 and median absolute error 113.41.

The right column of Fig.~3 contrasts success and failure. Image 518, at the 25th absolute-error percentile, has absolute error 50.17; the wood-clad house 2138 is strongly underestimated. Both show complete buildings, illustrating that adequate coverage can coexist with large errors. Their visual differences suggest questions for further study, without establishing causes of accuracy.

Regional perturbation [25] tests a narrower question. Replacing $4\times4$ regions with gray pixels or radius-8 blur gives the largest responses near house regions (Fig.~3). However, the two-view gray/blur correlation for 5712 is only 0.267. These maps describe perturbation sensitivity, not reliable identification of price-related attributes.

## Discussion

The three routes reveal different requirements for visual valuation. Handcrafted SIFT features explicitly represent local appearance, but visual-word pooling discards spatial relationships, potentially limiting discrimination between properties with different building and environmental layouts. This highlights the importance of organizing image evidence.

The scratch-CNN route makes this organization learnable, but requires learning features and price relationships from limited supervision. The compact head suggests that matching capacity to available labels can be more useful than architectural complexity. Greater expressive power therefore does not automatically produce more informative price features.

Frozen pretrained encoders supply knowledge learned beyond the price-labelled dataset, allowing task learning to focus on regression. Adaptation remains essential: views determine which evidence survives preprocessing, while transforms and kernel geometry connect visual similarity to price. Caching enables reuse on limited hardware, although dense kernels constrain scalability; fusion exploits complementary representations with unequal contributions.

Together, accurate prediction requires informative representations, adequate coverage and a mapping aligned with valuation. Appearance provides partial evidence of value; unobserved location, floor area and transaction conditions may explain remaining ambiguity. Future work should connect spatial and contextual features more directly to price differences. Multiple component changes and reused development data also warrant independent grouped evaluation.
## References

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
