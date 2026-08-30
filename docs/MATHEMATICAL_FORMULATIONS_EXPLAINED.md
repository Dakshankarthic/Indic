# Comprehensive Mathematical Formulations & Continuous Pipeline Derivations

This document presents the complete, unbroken end-to-end mathematical formulations governing the **AutoAnn-Indic** document layout analysis and transcription pipeline. Every equation is formulated with strict mathematical continuity, where the output of each stage serves as the direct input to the next.

---

## Stage 0: Physical Document Formation & Orthographic Challenges

### 1. Continuous Headline (*Shirorekha*) Union
Devanagari characters are bound along the upper boundary by a continuous horizontal stroke $\mathcal{H}_{\text{shirorekha}}$. Standard connected component analysis fuses all individual glyphs $\mathcal{G}_i$ into a single component:
$$\mathcal{S}_{\text{word}} = \bigcup_{i=1}^{M} \mathcal{G}_i \cup \mathcal{H}_{\text{shirorekha}}$$

### 2. Substrate Degradation & Illumination Formulation
Optical capture $I(x,y)$ on palm leaf (*Borassus flabellifer*) and handmade rag paper models true reflectance $R(x,y)$, spatially varying illumination field $L(x,y)$, and additive substrate decay noise $\eta(x,y)$:
$$I(x,y) = R(x,y) \cdot L(x,y) + \eta(x,y)$$

### 3. Hierarchical Target Partitioning (PAGE-XML 2013 Framework)
Given image $I \in \mathbb{R}^{H \times W \times 3}$, find optimal semantic decomposition $\mathcal{T}$:
$$\mathcal{T} = \left\{ \mathcal{R}_{\text{frame}}, \{\mathcal{R}_{\text{text}}^{(k)}\}_{k=1}^{K}, \{\mathcal{R}_{\text{illus}}^{(m)}\}_{m=1}^{M}, \{\mathcal{R}_{\text{damage}}^{(d)}\}_{d=1}^{D}, \{\mathcal{L}_{j}\}_{j=1}^{J}, \{\mathcal{W}_{j,p}\}, \{\mathcal{G}_{j,p,q}\} \right\}$$
$$\mathcal{P} = \{(x_1, y_1), (x_2, y_2), \dots, (x_V, y_V)\} \subset \mathbb{R}^2$$

---

## Stage 1: Adaptive Preprocessing & Binarization

### 4. Adaptive Gaussian Thresholding & Morphological Cleaning
Foreground binary mask $B(x,y)$ is computed by local Gaussian neighborhood convolution ($r=25, C=5$):
$$\mu_G(x,y) = (I_{\text{gray}} * G_\sigma)(x,y) = \sum_{u=-r}^r \sum_{v=-r}^r I_{\text{gray}}(x-u, y-v) \cdot \frac{1}{2\pi\sigma^2} e^{-\frac{u^2+v^2}{2\sigma^2}}$$
$$B(x,y) = \begin{cases} 1 & \text{if } I_{\text{gray}}(x,y) < \mu_G(x,y) - C \\ 0 & \text{otherwise} \end{cases}$$
$$B_{\text{clean}}(x,y) = (B \ominus \mathcal{E}_{3\times 3}) \oplus \mathcal{E}_{3\times 3}$$

---

## Stage 2: Foundation Vision Transformer Manifolds (DINOv2)

### 5. Patch Discretization & Position Encoding
Image $I$ is discretized into $N = \frac{H'}{14} \times \frac{W'}{14}$ non-overlapping $14\times 14$ patches $\mathbf{x}_p \in \mathbb{R}^{14 \times 14 \times 3}$:
$$\mathbf{x}_0 = \left[ \mathbf{x}_{\text{cls}}; \, \mathbf{x}_p^1 \mathbf{E}; \, \mathbf{x}_p^2 \mathbf{E}; \, \dots; \, \mathbf{x}_p^N \mathbf{E} \right] + \mathbf{E}_{\text{pos}}, \quad \mathbf{E} \in \mathbb{R}^{588 \times 768}, \quad \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times 768}$$

### 6. Multi-Head Self-Attention (MHSA) & Cosine Affinity
Linear projection into Query, Key, Value representations across $h=12$ attention heads ($d_k = 64$):
$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$
$$A_{ij} = \frac{\mathbf{z}_i \cdot \mathbf{z}_j}{\|\mathbf{z}_i\|_2 \|\mathbf{z}_j\|_2} = \cos(\theta_{ij})$$

---

## Stage 3: Substrate-Adaptive Feature Clustering

### 7. Border-Invariant Manifold Clustering
Boundary band luminance $\bar{L}_{\text{border}}$ classifies substrate:
$$\bar{L}_{\text{border}} = \frac{1}{|\mathcal{B}|} \sum_{(x,y) \in \mathcal{B}} I_{\text{gray}}(x,y)$$
* **Printed Paper Mode ($\bar{L}_{\text{border}} > 200, k=2$):**
  $$\mathcal{J}_{\text{paper}} = \sum_{i=1}^N \min_{j \in \{0,1\}} \|\mathbf{z}_i - \mathbf{c}_j\|^2$$
* **Palm-Leaf Mode ($\bar{L}_{\text{border}} \le 200, k=3$):**
  $$l_{\text{bg}} = \text{mode}\left(\mathcal{L}_{\text{boundary}}\right), \quad \mathcal{M}_{\text{text}} = \{i \mid l_i \neq l_{\text{bg}} \land \bar{I}_i < \bar{I}_{\text{substrate}}\}$$

---

## Stage 4: 6-Channel nnU-Net Deep Semantic Segmentation

### 8. Encoder-Decoder Feature Propagation & Instance Normalization
$$\mathbf{x}^{(l)} = \text{LeakyReLU}_{\alpha=0.01}\left( \text{IN}\left( \mathbf{W}_2^{(l)} * \text{LeakyReLU}\left( \text{IN}\left( \mathbf{W}_1^{(l)} * \mathbf{x}^{(l-1)} \right) \right) \right) \right)$$
$$\text{IN}(\mathbf{v}) = \gamma \left( \frac{\mathbf{v} - \mu(\mathbf{v})}{\sqrt{\sigma^2(\mathbf{v}) + \epsilon}} \right) + \beta, \quad \mu(\mathbf{v}) = \frac{1}{HW}\sum_{i=1}^H \sum_{j=1}^W v_{i,j}, \quad \sigma^2(\mathbf{v}) = \frac{1}{HW}\sum_{i=1}^H \sum_{j=1}^W (v_{i,j}-\mu)^2$$

### 9. 6-Channel Softmax Class Posterior
$$P(C = c \mid x, y) = \frac{\exp(z_c(x, y))}{\sum_{k=1}^{6} \exp(z_k(x, y))}$$
Target Channels $c \in \{1: \text{TextRegion}, 2: \text{Marginalia}, 3: \text{GraphicRegion}, 4: \text{PageFrame}, 5: \text{Damage}, 6: \text{TextLine}\}$.

### 10. Multi-Scale Deep Supervision Compound Loss
$$\mathcal{L}_{\text{total}} = \sum_{s=0}^{2} w_s \left[ \lambda_1 \mathcal{L}_{\text{Focal}}^{(s)}(\hat{\mathbf{Y}}_s, \mathbf{Y}_s) + \lambda_2 \mathcal{L}_{\text{Dice}}^{(s)}(\hat{\mathbf{Y}}_s, \mathbf{Y}_s) \right], \quad w = [1.0, 0.5, 0.25]$$
$$\mathcal{L}_{\text{Focal}} = -\frac{1}{N} \sum_{i=1}^N \alpha_t (1 - p_{t,i})^\gamma \log(p_{t,i}), \quad \gamma = 2.0, \, \alpha_t = 0.25$$
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^N p_{t,i} y_{t,i} + \epsilon}{\sum_{i=1}^N p_{t,i}^2 + \sum_{i=1}^N y_{t,i}^2 + \epsilon}$$

---

## Stage 5: Multi-Column Gutter Parsing & Graphic Discrimination

### 11. Vertical Projection Gutter Partitioning
$$V_{\text{norm}}(x) = \frac{(V * G_{\sigma_c})(x)}{\max_x (V * G_{\sigma_c})(x)} < 0.02 \quad \text{for continuous gap } \Delta x > \frac{W_R}{15}$$
Reading order topological sorting: $\mathcal{C}_1 \prec \mathcal{C}_2 \prec \dots \prec \mathcal{C}_k$.

### 12. GraphicRegion Illustration Filter
$$\rho_{\text{ink}} = \frac{1}{H_c W_c} \sum_{y=0}^{H_c-1} \sum_{x=0}^{W_c-1} B(y,x), \quad \nu_{\text{valley}} = \frac{N_{\text{valleys}}}{H_c / 100}$$
$$\text{Class}(\mathcal{C}) = \text{GraphicRegion} \iff (\rho_{\text{ink}} > 0.30 \land \nu_{\text{valley}} < 1.5) \lor (\rho_{\text{ink}} > 0.25 \land N_{\text{valleys}} < 3)$$

---

## Stage 6: Shirorekha Headline Slicing & Akshara Valley Tracking

### 13. Horizontal Headline Ablation
$$P_H(y) = \sum_{x=0}^{W_L-1} B_{\text{line}}(y,x), \quad y^* = \arg\max_{y \in [0, 0.45 H_L]} P_H(y) \quad \text{s.t. } P_H(y^*) > 0.25 W_L$$
$$B_{\text{ablated}}(y,x) = \begin{cases} 0 & \text{if } |y - y^*| \le \tau, \quad \tau = \max(2, \lfloor 0.06 H_L \rfloor) \\ B_{\text{line}}(y,x) & \text{otherwise} \end{cases}$$

### 14. 1D Gaussian Valley Cut-Point Detection
$$S_V(x) = (P_V * G_\sigma)(x) = \sum_{k=-w}^w P_V(x-k) \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{k^2}{2\sigma^2}}$$
$$\frac{\partial S_V}{\partial x} = 0, \quad \frac{\partial^2 S_V}{\partial x^2} > 0, \quad S_V(x_k^*) < \theta_{\text{valley}} = 0.25 \cdot \left(\frac{\text{Peak}_L + \text{Peak}_R}{2}\right)$$

---

## Stage 7: Physical Damage & Binder Hole Isolation

### 15. Isoperimetric Circularity Formulation
$$\Psi(\mathcal{C}) = \frac{4\pi \cdot \text{Area}(\mathcal{C})}{[\text{Perimeter}(\mathcal{C})]^2}$$
$$\mathcal{C} \in \mathcal{R}_{\text{damage}} \iff 500 \le \text{Area}(\mathcal{C}) \le 8000 \text{ px} \quad \land \quad \Psi(\mathcal{C}) > 0.85 \quad \land \quad 0.5 \le \frac{\text{Width}(\mathcal{C})}{\text{Height}(\mathcal{C})} \le 2.0$$

---

## Stage 8: Multilingual Recurrent Sequence Modeling (CRNN-CTC OCR)

### 16. Bidirectional LSTM Context Sequence
$$\mathbf{h}_t = \left[ \overrightarrow{\text{LSTM}}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1}); \, \overleftarrow{\text{LSTM}}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1}) \right] \in \mathbb{R}^{2 D_{\text{hidden}}}$$
$$\mathbf{y}_t = \text{softmax}(\mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o) \in \mathbb{R}^{|\Sigma| + 1}$$

### 17. Connectionist Temporal Classification (CTC) Loss & Beam Search
$$P(\mathbf{Y} \mid \mathbf{X}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{Y})} \prod_{t=1}^T y_{\pi_t}^t$$
$$\hat{\mathbf{Y}} = \arg\max_{\mathbf{Y}} \left[ \log P_{\text{CTC}}(\mathbf{Y} \mid \mathbf{X}) + \alpha \log P_{\text{LM}}(\mathbf{Y}) + \beta |\mathbf{Y}| \right]$$

---

## Stage 9: Polygon Simplification & Metric Evaluation

### 18. Ramer-Douglas-Peucker (RDP) Polygon Vector Reduction
$$d_\perp(p_i, \overline{p_1 p_n}) = \frac{|(y_n - y_1)x_i - (x_n - x_1)y_i + x_n y_1 - y_n x_1|}{\sqrt{(y_n - y_1)^2 + (x_n - x_1)^2}} \le \epsilon = 0.005 \cdot \text{ArcLength}(\mathcal{P})$$

### 19. Quantitative Levenshtein Error Rates & Human Effort Model
$$\text{CER} = \frac{\sum_{i=1}^M D_{\text{Levenshtein}}(S_{\text{pred}}^{(i)}, S_{\text{gt}}^{(i)})}{\sum_{i=1}^M \text{Length}(S_{\text{gt}}^{(i)})}, \quad \text{WER} = \frac{\sum_{i=1}^M D_{\text{Levenshtein}}(W_{\text{pred}}^{(i)}, W_{\text{gt}}^{(i)})}{\sum_{i=1}^M |W_{\text{gt}}^{(i)}|}$$
$$E = 50 \cdot |\Delta \mathcal{R}| + \sum_{i=1}^K \left[ (1 - \text{IoU}(\mathcal{P}_i, \mathcal{P}_i^{\text{gt}})) \cdot 100 + 0.5 \cdot |\Delta V_i| \right]$$
