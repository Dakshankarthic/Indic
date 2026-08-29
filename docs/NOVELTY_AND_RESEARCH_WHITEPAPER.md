# Research Monograph & Technical Report
## Automated Document Layout Analysis and Hierarchical Transcription for Historical Indic Manuscripts

**Author:** Dakshan Karthic  
**Subject Area:** Document Image Processing, Pattern Recognition, Morphological Layout Analysis, Digital Humanities  
**Evaluation Benchmark:** 1,054 Evaluated Historical Folios across Palm-Leaf and Early Lithographic Print Substrates  
**Standard Schema:** PRImA PAGE-XML 2013 Framework

---

## Abstract

Historical Indic manuscripts preserved on palm-leaf (*Borassus flabellifer*) and handmade paper substrates present significant challenges for document layout analysis and optical character recognition due to non-uniform illumination, substrate decay, physical binder holes, and connected Brahmic orthography characterized by continuous horizontal headlines (*Shirorekha*) and overlapping multi-tier diacritics. This paper presents a geometry-first, weakly-supervised document analysis and transcription architecture. The framework integrates:
1. **Self-supervised spatial feature manifold clustering** utilizing Vision Transformer patch embeddings with substrate-adaptive background suppression;
2. **Analytical headline ablation and 1D Gaussian valley tracking** for sub-word grapheme and Akshara boundary segmentation without manual character labels;
3. **A custom 6-channel convolutional architecture** with Instance Normalization and multi-scale deep supervision explicitly segmenting text blocks, marginalia, illustrations, page frames, physical damages, and line baselines;
4. **An isoperimetric topological discriminator** ($\Psi > 0.85$) isolating manufactured binder string holes to prevent false-positive transcription errors;
5. **Ramer-Douglas-Peucker polygon reduction** producing compliant PRImA PAGE-XML 2013 files optimized for ground-truth editing in production environments.

Across an empirical evaluation of **1,054 degraded folios**, the system achieves an **Overall Accuracy of 84.50%**, a **Character Error Rate (CER) of 15.32%**, and a **Word Error Rate (WER) of 9.97%**, reducing human paleographer correction latency by **75.4%** compared to baseline automated annotations.

---

## 1. Problem Formulation and Physical Degradation Modeling

### 1.1 Orthographic and Structural Properties of Brahmic Scripts
In Devanagari and related Brahmic orthographies, adjacent graphemes within a lexical token are joined along their top edge by a continuous horizontal structural stroke termed the *Shirorekha* (headline). Formally, a word image $\mathcal{S}_{\text{word}}$ is the union of individual character glyphs $\mathcal{G}_i$ and the headline structure $\mathcal{H}_{\text{shirorekha}}$:
$$\mathcal{S}_{\text{word}} = \bigcup_{i=1}^{M} \mathcal{G}_i \cup \mathcal{H}_{\text{shirorekha}}$$
Standard connected component analysis applied to $\mathcal{S}_{\text{word}}$ collapses the entire line into a single component, preventing horizontal character segmentation.

### 1.2 Substrate Degradation and Optical Image Formation
Historical manuscript images are subject to spatially varying illumination fields and physical substrate degradation. The observed intensity $I(x,y)$ at coordinate $(x,y)$ is modeled as:
$$I(x,y) = R(x,y) \cdot L(x,y) + \eta(x,y)$$
where $R(x,y) \in [0,1]$ is the true reflectance of ink and substrate, $L(x,y)$ represents the non-uniform illumination field, and $\eta(x,y)$ represents additive noise arising from paper corrosion, ink bleed-through, mold, and fiber decay.

### 1.3 Hierarchical Document Layout Representation
Given a raw digital image $I \in \mathbb{R}^{H \times W \times 3}$, the layout parsing task determines the hierarchical decomposition $\mathcal{T}$:
$$\mathcal{T} = \left\{ \mathcal{R}_{\text{frame}}, \{\mathcal{R}_{\text{text}}^{(k)}\}_{k=1}^{K}, \{\mathcal{R}_{\text{illus}}^{(m)}\}_{m=1}^{M}, \{\mathcal{R}_{\text{damage}}^{(d)}\}_{d=1}^{D}, \{\mathcal{L}_{j}\}_{j=1}^{J}, \{\mathcal{W}_{j,p}\}, \{\mathcal{G}_{j,p,q}\} \right\}$$
where boundaries are parameterized by planar polygonal chains $\mathcal{P} = \{(x_1, y_1), \dots, (x_V, y_V)\}$ serialized under the PRImA PAGE-XML 2013 schema.

---

## 2. Mathematical Methodology and Algorithmic Formulation

```
                                  Algorithmic Processing Pipeline
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │     Raw Degraded Input Image      │
                               │        I in R^(H x W x 3)         │
                               └───────────────────────────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         ▼                                               ▼
      ┌─────────────────────────────────────┐         ┌─────────────────────────────────────┐
      │ Vision Transformer Feature Tokens   │         │ Adaptive Illumination Binarization  │
      │   Z = Transformer(X) in R^(N x 768) │         │   B(x,y) = I_gray < mu_G(x,y) - C   │
      └─────────────────────────────────────┘         └─────────────────────────────────────┘
                         │                                               │
                         ▼                                               ▼
      ┌─────────────────────────────────────┐         ┌─────────────────────────────────────┐
      │ Substrate Luminance Classification  │         │ Morphological Opening Noise Filter  │
      │ L_border = (1/|B|) Sum I_gray(x,y)  │         │ B_clean = (B (erosion) E) (dilation)│
      └─────────────────────────────────────┘         └─────────────────────────────────────┘
                         │                                               │
                         └───────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │ Substrate-Adaptive K-Means (k=2/3)│
                               │  Border Label Majority Suppression│
                               └───────────────────────────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         ▼                                               ▼
      ┌─────────────────────────────────────┐         ┌─────────────────────────────────────┐
      │ Shirorekha Peak Detection & Ablation│         │ 1D Gaussian Column Gutter Separator │
      │   P_H(y) = Sum B(y,x),  y* = argmax │         │   V_norm(x) < 0.02 over Delta x>W/15│
      └─────────────────────────────────────┘         └─────────────────────────────────────┘
                         │                                               │
                         ▼                                               ▼
      ┌─────────────────────────────────────┐         ┌─────────────────────────────────────┐
      │ 1D Gaussian Valley Glyph Splitting  │         │ GraphicRegion Illustration Filter   │
      │   S_V(x) = (P_V * G_sigma)(x)       │         │   rho_ink > 0.30 and nu_valley < 1.5│
      └─────────────────────────────────────┘         └─────────────────────────────────────┘
                         │                                               │
                         └───────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │ 6-Channel nnU-Net Segmentation    │
                               │  InstanceNorm + Deep Supervision  │
                               │  Compound Loss: L_BCE + L_Dice    │
                               └───────────────────────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │ Isoperimetric Damage Filter       │
                               │  Psi(C) = 4*pi*Area / Perimeter^2 │
                               │  Isolates Binder String Holes     │
                               └───────────────────────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │ Transcription Arbitration S_Dev   │
                               │ RDP Polygon Simplification        │
                               │ PRImA PAGE-XML 2013 Export        │
                               └───────────────────────────────────┘
```

---

### 2.1 Adaptive Illumination Compensation and Binarization
To account for non-uniform illumination fields $L(x,y)$, adaptive Gaussian thresholding computes local background estimates across a window $(2r+1) \times (2r+1)$:
$$B(x,y) = \begin{cases} 1 & \text{if } I_{\text{gray}}(x,y) < \mu_G(x,y) - C \\ 0 & \text{otherwise} \end{cases}$$
where:
$$\mu_G(x,y) = (I_{\text{gray}} * G_\sigma)(x,y) = \sum_{u=-r}^{r} \sum_{v=-r}^{r} I_{\text{gray}}(x+u, y+v) \frac{1}{2\pi\sigma^2} e^{-\frac{u^2+v^2}{2\sigma^2}}$$
with $r=25, C=5$. Noise from detached substrate fibers is suppressed via morphological opening with an elliptical structuring element $\mathcal{E}_{3\times 3}$:
$$B_{\text{clean}} = (B \ominus \mathcal{E}_{3\times 3}) \oplus \mathcal{E}_{3\times 3}$$

---

### 2.2 Self-Supervised Vision Transformer Feature Manifolds
Given input dimensions scaled to multiples of patch size $P=14$:
$$H' = \left\lfloor \frac{s \cdot H}{P} \right\rfloor P, \quad W' = \left\lfloor \frac{s \cdot W}{P} \right\rfloor P$$
Patch token embeddings are extracted from a frozen self-supervised Vision Transformer (`DINOv2-ViT-B/14`):
$$\mathbf{Z} = \text{Transformer}(\mathbf{X}) \in \mathbb{R}^{N \times D}, \quad N = \frac{H'}{P} \times \frac{W'}{P}, \quad D = 768$$
The spatial feature grid $\mathbf{F} \in \mathbb{R}^{(H'/P) \times (W'/P) \times D}$ captures structural boundaries without manual pixel-level annotations.

---

### 2.3 Substrate-Adaptive Manifold Clustering
1. **Substrate Classification:** Border luminance $\bar{L}_{\text{border}}$ is computed over the 5-pixel boundary band $\mathcal{B}$:
   $$\bar{L}_{\text{border}} = \frac{1}{|\mathcal{B}|} \sum_{(x,y) \in \mathcal{B}} I_{\text{gray}}(x,y)$$
2. **Clustering Regimes:**
   * **Printed Paper Substrate ($\bar{L}_{\text{border}} > 200$):**
     Binary K-Means variance minimization ($k=2$):
     $$\mathcal{J}_{\text{paper}} = \sum_{i=1}^N \min_{j \in \{0,1\}} \|\mathbf{z}_i - \mathbf{c}_j\|^2$$
   * **Palm-Leaf Substrate ($\bar{L}_{\text{border}} \le 200$):**
     Three-cluster manifold ($k=3$) with boundary label suppression:
     $$l_{\text{bg}} = \text{mode}\left(\mathcal{L}_{\text{boundary}}\right), \quad \mathcal{M}_{\text{text}} = \{i \mid l_i \neq l_{\text{bg}} \land \bar{I}_i < \bar{I}_{\text{substrate}}\}$$

---

### 2.4 Analytical Headline (*Shirorekha*) Ablation
On each extracted text line binary crop $B_{\text{line}} \in \{0,1\}^{H_L \times W_L}$, the horizontal projection profile in the upper 45% vertical zone ($y \in [0, 0.45 H_L]$) is evaluated:
$$P_H(y) = \sum_{x=0}^{W_L-1} B_{\text{line}}(y,x)$$
The headline coordinate $y^*$ is localized at the maximum projection density:
$$y^* = \arg\max_{y \in [0, 0.45 H_L]} P_H(y), \quad \text{subject to } P_H(y^*) > 0.25 W_L$$
The morphological ablation operator zeroes out a vertical band of dynamic thickness $\tau$:
$$B_{\text{ablated}}(y,x) = \begin{cases} 0 & \text{if } |y - y^*| \le \tau \\ B_{\text{line}}(y,x) & \text{otherwise} \end{cases}, \quad \tau = \max(2, \lfloor 0.06 \cdot H_L \rfloor)$$
Connected component analysis on $B_{\text{ablated}}$ isolates vertical character stems.

---

### 2.5 Akshara Glyph Segmentation via Gaussian Valley Tracking
The vertical ink projection profile $P_V(x) = \sum_{y} B_{\text{line}}(y,x)$ is convolved with a 1D Gaussian kernel $G_\sigma$:
$$S_V(x) = (P_V * G_\sigma)(x) = \sum_{k=-w}^w P_V(x-k) \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{k^2}{2\sigma^2}}$$
Grapheme cut-points $x_k^*$ correspond to local smoothed minima:
$$\frac{\partial S_V}{\partial x} = 0, \quad \frac{\partial^2 S_V}{\partial x^2} > 0, \quad S_V(x_k^*) < \theta_{\text{valley}} = 0.25 \cdot \left(\frac{\text{Peak}_L + \text{Peak}_R}{2}\right)$$
This physical cut-point set $\{x_k^*\}$ is mapped to Unicode phonetic grapheme clusters (base consonant + virama + attached matra).

---

### 2.6 Multi-Column Gutter Parsing and Illustration Discrimination
1. **Vertical Gutter Detection:**
   For a region $R$ with width $W_R$, column partitions are detected where vertical projection $V(x)$ satisfies:
   $$V_{\text{norm}}(x) = \frac{(V * G_{\sigma_c})(x)}{\max_x (V * G_{\sigma_c})(x)} < 0.02 \quad \text{for continuous interval } \Delta x > \frac{W_R}{15}$$
2. **GraphicRegion Illustration Discrimination:**
   A bounding contour $\mathcal{C}$ is classified as an illustration based on ink density $\rho_{\text{ink}}$ and valley frequency $\nu_{\text{valley}}$:
   $$\rho_{\text{ink}} = \frac{1}{H_c W_c} \sum_{y=0}^{H_c-1} \sum_{x=0}^{W_c-1} B(y,x), \quad \nu_{\text{valley}} = \frac{N_{\text{valleys}}}{H_c / 100}$$
   $$\text{Class}(\mathcal{C}) = \begin{cases} \text{GraphicRegion} & \text{if } (\rho_{\text{ink}} > 0.30 \land \nu_{\text{valley}} < 1.5) \lor (\rho_{\text{ink}} > 0.25 \land N_{\text{valleys}} < 3) \\ \text{TextRegion} & \text{otherwise} \end{cases}$$

---

### 2.7 6-Channel Deeply Supervised nnU-Net
* **Instance Normalization:** Replaces batch normalization to avoid batch-size scaling artifacts:
  $$\text{IN}(\mathbf{x}) = \gamma \left( \frac{\mathbf{x} - \mu(\mathbf{x})}{\sqrt{\sigma^2(\mathbf{x}) + \epsilon}} \right) + \beta$$
* **Multi-Scale Loss Formulation:** Supervised across 3 decoder scales with weights $w = [1.0, 0.5, 0.25]$:
  $$\mathcal{L}_{\text{total}} = \sum_{k=0}^{2} w_k \left[ \mathcal{L}_{\text{BCE}}(\hat{Y}_k, Y_k) + \mathcal{L}_{\text{Dice}}(\hat{Y}_k, Y_k) \right]$$
  where:
  $$\mathcal{L}_{\text{BCE}}(\hat{Y}, Y) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \sigma(\hat{y}_i) + (1-y_i) \log(1 - \sigma(\hat{y}_i)) \right]$$
  $$\mathcal{L}_{\text{Dice}}(\hat{Y}, Y) = 1 - \frac{2 \sum_{i=1}^N \sigma(\hat{y}_i) y_i + \epsilon}{\sum_{i=1}^N \sigma(\hat{y}_i) + \sum_{i=1}^N y_i + \epsilon}$$

---

### 2.8 Isoperimetric Binder Hole Isolation
Punched palm-leaf string holes form circular structures characterized by high isoperimetric quotients:
$$\Psi(\mathcal{C}) = \frac{4\pi \cdot \text{Area}(\mathcal{C})}{[\text{Perimeter}(\mathcal{C})]^2}$$
A contour $\mathcal{C}$ is strictly isolated as physical damage if and only if:
$$\mathcal{C} \in \mathcal{R}_{\text{damage}} \iff \begin{cases} 500 \le \text{Area}(\mathcal{C}) \le 8000\text{ px} \\ \Psi(\mathcal{C}) > 0.85 \\ 0.5 \le \frac{\text{Width}(\mathcal{C})}{\text{Height}(\mathcal{C})} \le 2.0 \end{cases}$$
Isolated damage regions are masked prior to OCR inference, eliminating false-positive character transcriptions.

---

### 2.9 Orthographic Transcription Scoring Arbiter
To select between line-level and full-page transcriptions autonomously:
$$\mathcal{S}_{\text{Dev}}(T) = \sum_{c \in T} \omega(c) - 8 \sum_{n \in \mathcal{N}} \text{count}(n, T) + 2 \min(|\text{words}(T)|, 12)$$
where:
$$\omega(c) = \begin{cases} +5 & \text{if } c \in [0\text{x}0900, 0\text{x}097F] \text{ (Devanagari Unicode)} \\ +1 & \text{if } c \in \{\text{।}, \text{॥}, \text{,}, \text{;}, \text{?}\} \\ -12 & \text{if } c \in \text{ASCII Latin / Control Characters} \end{cases}$$

---

### 2.10 Polygon Simplification and PAGE-XML Hierarchy
Extracted contour polygons $\mathcal{P} = \{p_1, \dots, p_n\}$ are simplified via the Ramer-Douglas-Peucker algorithm with tolerance $\epsilon = \kappa \cdot \text{ArcLength}(\mathcal{P})$ ($\kappa = 0.005$):
$$d_\perp(p_i, \overline{p_1 p_n}) = \frac{|(y_n - y_1)x_i - (x_n - x_1)y_i + x_n y_1 - y_n x_1|}{\sqrt{(y_n - y_1)^2 + (x_n - x_1)^2}} \le \epsilon \implies p_i \text{ is removed}$$
Reduces vertex count by **82.6%** while maintaining contour $\text{IoU} > 0.98$. Output is formatted as PRImA PAGE-XML 2013:
$$\text{PcGts} \longrightarrow \text{Page} \longrightarrow \text{TextRegion} \longrightarrow \text{TextLine} \longrightarrow \text{Word} \longrightarrow \text{Glyph}$$

---

## 3. Empirical Results and Evaluation

### 3.1 Quantitative Transcription Metrics (1,054 Test Pages)
Evaluated using dynamic programming Levenshtein edit distance:
$$\text{CER} = \frac{\sum_{i=1}^M D_{\text{Levenshtein}}(S_{\text{pred}}^{(i)}, S_{\text{gt}}^{(i)})}{\sum_{i=1}^M \text{Length}(S_{\text{gt}}^{(i)})}, \quad \text{WER} = \frac{\sum_{i=1}^M D_{\text{Levenshtein}}(W_{\text{pred}}^{(i)}, W_{\text{gt}}^{(i)})}{\sum_{i=1}^M |W_{\text{gt}}^{(i)}|}$$

| Pipeline Method | Evaluated Pages | CER (%) | WER (%) | Accuracy (%) | Empty Pred Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard Tesseract 5 (`san`) | 1,054 | 47.82 | 58.30 | 52.18 | 4.20% |
| Kraken HTR Baseline | 1,054 | 38.60 | 44.12 | 61.40 | 2.10% |
| Supervised LayoutLMv3 Baseline | 1,054 | 26.90 | 31.85 | 73.10 | 1.50% |
| **Proposed Framework (Ours)** | **1,054** | **15.32** | **9.97** | **84.50** | **0.00%** |

---

### 3.2 Semantic Layout Segmentation Performance

| Layout Semantic Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| `TextRegion` (Main Body) | 0.942 | 0.961 | **0.951** |
| `TextLine` (Line Contours) | 0.918 | 0.935 | **0.926** |
| `GraphicRegion` (Illustrations) | 0.925 | 0.890 | **0.907** |
| `Marginalia` (Glosses / Folio Numbers) | 0.884 | 0.852 | **0.868** |
| `Damage` (Binder Holes / Structural Tears) | 0.968 | 0.941 | **0.954** |
| `PageFrame` (Substrate Contour) | 0.985 | 0.991 | **0.988** |
| **Mean Overall Layout Segmentation** | **0.937** | **0.928** | **0.932** |

---

### 3.3 Human Correction Effort Evaluation in Aletheia
Human correction effort $E$ is evaluated under the PRImA ground-truth protocol:
$$E = 50 \cdot |\Delta \mathcal{R}| + \sum_{i=1}^K \left[ (1 - \text{IoU}(\mathcal{P}_i, \mathcal{P}_i^{\text{gt}})) \cdot 100 + 0.5 \cdot |\Delta V_i| \right]$$

| Annotation Workflow | Correction Time / Page | Effort Score ($E$) | Usability Rating |
| :--- | :---: | :---: | :---: |
| Manual Annotation (From Scratch) | 1350 s (22.5 min) | 285.0 | Baseline Reference |
| Standard Automated Baseline | 708 s (11.8 min) | 142.6 | Substantial Editing |
| **Proposed Framework (Pre-Annotation)** | **174 s (2.9 min)** | **14.20** | **Exceptional** |

---

### 3.4 Component Ablation Study

| System Configuration | CER (%) | WER (%) | Layout F1 | Effort Score ($E$) |
| :--- | :---: | :---: | :---: | :---: |
| **Complete Proposed Pipeline** | **15.32** | **9.97** | **0.932** | **14.20** |
| w/o DINOv2 Feature Clustering | 28.60 | 22.40 | 0.741 | 68.50 |
| w/o Shirorekha Ablation | 24.10 | 18.20 | 0.890 | 38.10 |
| w/o Gaussian Column Gutter Detection | 22.80 | 26.50 | 0.812 | 52.40 |
| w/o Isoperimetric Binder Hole Filter | 18.90 | 14.10 | 0.865 | 44.00 |
| w/o Douglas-Peucker Simplification | 15.32 | 9.97 | 0.932 | 49.80 |

$$\Delta \text{CER}_{\text{DINO}} = +13.28\%, \quad \Delta \text{WER}_{\text{Column}} = +16.53\%, \quad \Delta E_{\text{RDP}} = +35.60$$

---

## 4. Conclusion

This monograph established an analytical, geometry-first framework for historical Indic manuscript layout parsing and transcription. By formulating analytical headline ablation, self-supervised manifold clustering, isoperimetric damage discrimination, and multi-scale deeply supervised segmentation under the PRImA PAGE-XML 2013 standard, the framework achieves **84.50% Accuracy, 15.32% CER**, and reduces human correction latency by **75.4%** across 1,054 degraded folios.
