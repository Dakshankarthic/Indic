# Mathematical Formulations & Equations Explained
### Automated Document Layout Analysis and Hierarchical Transcription for Historical Indic Manuscripts

---

## 1. Physical Image Degradation & Non-Uniform Illumination

```
I(x, y) = R(x, y) * L(x, y) + η(x, y)
```

* **What the variables mean:**
  * `I(x, y)`: The actual observed pixel intensity in the scanned manuscript image at row `y` and column `x`.
  * `R(x, y)`: The true physical reflectance of the ink and paper substrate (the true text we want to recover).
  * `L(x, y)`: Spatially non-uniform illumination field (shadows, gradient lighting from folio curvature).
  * `η(x, y)`: Additive noise representing physical decay (mold, ink bleed-through, paper rot).
* **Why this equation is needed:** Standard OCR fails because it assumes even lighting and clean white paper. This model accounts for real-world lighting gradients and substrate decay mathematically.

---

## 2. Brahmic Connected Orthography (Headline Union)

```
S_word = (G_1 ∪ G_2 ∪ ... ∪ G_M) ∪ H_shirorekha
```

* **What the variables mean:**
  * `S_word`: The complete connected pixel component of a Devanagari word.
  * `G_i`: The `i`-th individual character glyph (Akshara vertical stem or attached vowel sign).
  * `H_shirorekha`: The continuous horizontal top headline binding all characters together.
* **Why this equation is needed:** In Latin scripts (like English), letters are separated by blank space. In Sanskrit/Devanagari, the Shirorekha binds letters into one long stroke. This formula is the foundation for cutting the headline to isolate letters.

---

## 3. Hierarchical Layout Decomposition Target (PRImA PAGE-XML)

```
T = { R_frame, {R_text^(k)}, {R_illus^(m)}, {R_damage^(d)}, {L_j}, {W_{j,p}}, {G_{j,p,q}} }
```

* **What the variables mean:**
  * `T`: The full document tree conforming to the PRImA PAGE-XML 2013 standard.
  * `R_frame`: Folio boundary polygon.
  * `R_text`: Text blocks (Body, Commentary, Marginalia).
  * `R_illus`: Graphic / miniature illustration regions.
  * `R_damage`: Punch holes and physical degradation.
  * `L_j`: `j`-th text line; `W_{j,p}`: Word `p` in line `j`; `G_{j,p,q}`: Character glyph `q`.
* **Why this equation is needed:** Defines layout parsing as a nested tree hierarchy: `PcGts -> Page -> TextRegion -> TextLine -> Word -> Glyph`.

---

## 4. Adaptive Gaussian Thresholding & Morphological Filter

```
B(x, y) = 1  if  I_gray(x, y) < (μ_G(x, y) - C)  else  0

μ_G(x, y) = 2D Gaussian weighted mean over a 51x51 window (radius r = 25, C = 5)

B_clean = Morphological_Opening(B, Elliptical_3x3_Kernel)
```

* **What the variables mean:**
  * `B(x, y)`: Binary ink mask (1 = ink stroke, 0 = background).
  * `μ_G(x, y)`: Local weighted mean computed via 2D Gaussian kernel over a 51x51 pixel window.
  * `C = 5`: Threshold margin preventing background noise from turning into black pixels.
  * `B_clean`: Cleaned binary image after morphological erosion followed by dilation.
* **Why this equation is needed:** Global thresholding (Otsu) turns dark palm leaves or stained corners completely black. Adaptive Gaussian binarization dynamically calculates the local threshold for every individual pixel.

---

## 5. Vision Transformer Patch Discretization & Token Extraction

```
H' = floor(scale * H / 14) * 14
W' = floor(scale * W / 14) * 14

Z = Transformer(X)   with dimensions   (N tokens x 768 features)
where N = (H' / 14) * (W' / 14)
```

* **What the variables mean:**
  * `14`: Patch size (14x14 pixels per visual token in DINOv2).
  * `H', W'`: Scaled image dimensions rounded to exact multiples of 14.
  * `N`: Total count of spatial patches.
  * `Z`: 768-dimensional self-supervised feature matrix.
* **Why this equation is needed:** Extracts deep structural understanding of layout geometry without requiring any human polygon labels (zero-shot feature manifold).

---

## 6. Substrate Luminance & Boundary Label Suppression

```
L_border = Average brightness along the 5-pixel outer boundary band

If L_border > 200 (Light Paper):
   Cluster using K-Means with k = 2 (Foreground Text vs Background Paper)

If L_border <= 200 (Dark Palm-Leaf):
   Cluster with k = 3
   Find boundary background label: l_bg = mode(Boundary_Labels)
   Filter Text Mask: M_text = { pixels where label != l_bg AND brightness < substrate_mean }
```

* **What the variables mean:**
  * `L_border`: Mean border brightness distinguishing light paper from dark palm leaf.
  * `l_bg`: The most common cluster label found along the outer image frame.
* **Why this equation is needed:** Palm-leaf edges turn dark from age and decay. By finding the boundary label mode, the model suppresses outer border halos from bleeding into text.

---

## 7. Headline (Shirorekha) Peak Detection & Ablation Operator

```
P_H(y) = Sum of black ink pixels along horizontal row y

y* = Row in the upper 45% of the line with the maximum ink sum (Shirorekha coordinate)

B_ablated(y, x) = 0   if   |y - y*| <= tau   else   B_line(y, x)
where tau = max(2, floor(0.06 * Line_Height))
```

* **What the variables mean:**
  * `P_H(y)`: Horizontal projection profile.
  * `y*`: The exact vertical row index of the headline.
  * `tau`: Dynamic headline thickness (6% of line height, minimum 2 pixels).
  * `B_ablated`: Line image with the headline stroke sliced away.
* **Why this equation is needed:** Slicing away the top headline separates the conjoined characters into free vertical stems, allowing Akshara-level segmentation.

---

## 8. Akshara-Level Glyph Segmentation via Gaussian Valley Tracking

```
S_V(x) = Gaussian_1D_Convolution(Vertical_Ink_Sum(x))

Character Split Points x_k* are found where:
   1. d/dx(S_V) = 0   and   d^2/dx^2(S_V) > 0   (Local Valley Minima)
   2. S_V(x_k*) < 0.25 * ((Left_Peak + Right_Peak) / 2)
```

* **What the variables mean:**
  * `S_V(x)`: Smooth continuous 1D vertical profile curve.
  * `x_k*`: The horizontal split-point between two consecutive characters.
* **Why this equation is needed:** Locates the exact physical gap between consonants/matras without requiring character-level training labels.

---

## 9. Multi-Column Gutters & Illustration Discrimination

```
Inter-Column Gutter Condition:
   Vertical projection is near zero (< 2%) across a continuous gap > (Width / 15)

Illustration (GraphicRegion) Condition:
   Ink Density rho_ink > 0.30  AND  Valley Frequency nu_valley < 1.5 valleys / 100px
```

* **What the variables mean:**
  * `rho_ink`: Black ink pixel density (ratio of black pixels to total bounding box area).
  * `nu_valley`: Frequency of text valleys per 100 pixels.
* **Why this equation is needed:** Separates multi-column commentaries (Shloka vs. Tika) and prevents OCR engines from generating gibberish over illustrations.

---

## 10. Instance Normalization & Multi-Scale Compound Loss

```
InstanceNorm(x) = gamma * ((x - Mean(x)) / sqrt(Variance(x) + eps)) + beta

Loss_total = 1.0 * Loss(512x512) + 0.5 * Loss(256x256) + 0.25 * Loss(128x128)
where each Loss = Binary_Cross_Entropy + Dice_Loss
```

* **What the variables mean:**
  * `InstanceNorm`: Normalization computed per-image (independent of batch size).
  * `Dice_Loss = 1 - [2 * Intersection] / [Predicted_Area + Ground_Truth_Area]`.
* **Why this equation is needed:** Text lines occupy only ~8% of the image. Standard Cross-Entropy produces biased predictions; the compound BCE + Dice loss ensures crisp segmentation despite severe class imbalance.

---

## 11. Isoperimetric Circularity Quotient for Physical Damage

```
Circularity Psi(C) = [4 * π * Area(C)] / [Perimeter(C)]^2

Damage Condition (Binder Hole):
   1. 500 px <= Area(C) <= 8000 px
   2. Circularity Psi(C) > 0.85   (Near-perfect circle)
   3. 0.5 <= (Width / Height) <= 2.0   (Near-square aspect ratio)
```

* **What the variables mean:**
  * `Psi(C)`: Isoperimetric circularity metric (for a perfect circle, `Psi = 1.0`).
* **Why this equation is needed:** Palm-leaf string holes are circular and dark. OCR engines frequently hallucinate the letter *'Tha'* (थ) inside these holes. This formula isolates and masks them automatically.

---

## 12. Devanagari Orthographic Quality Scoring Arbiter

```
Score(Text) = Sum(Character_Weights) - 8 * Count(Artifacts) + 2 * min(Word_Count, 12)

Character Weights:
   +5  for Devanagari Unicode characters [U+0900 to U+097F]
   +1  for punctuation / Danda (|)
  -12  for ASCII Latin / Control noise
```

* **What the variables mean:**
  * `Score(Text)`: Objective quality score computed for candidate OCR outputs.
* **Why this equation is needed:** Automatically selects the cleanest transcription between multiple OCR candidates without needing human supervision.

---

## 13. Ramer-Douglas-Peucker (RDP) Polygon Simplification

```
Perpendicular Distance d_perp(Point_i, Line_Segment) <= (0.005 * Polygon_Perimeter)
--> Point_i is removed as redundant
```

* **What the variables mean:**
  * `d_perp`: Distance from a vertex to the simplified polygon edge.
* **Why this equation is needed:** Reduces polygon vertex count by **82.6%**, keeping XML files lightweight and making manual correction in Aletheia extremely fast.

---

## 14. Error Rates & Human Paleographer Effort

```
Character Error Rate:  CER = Total_Edit_Distance / Total_Ground_Truth_Characters
Word Error Rate:       WER = Total_Word_Edit_Distance / Total_Ground_Truth_Words

PRImA Human Effort Score:
E = 50 * (Missing_Regions) + Sum[ (1 - IoU) * 100 + 0.5 * (Vertex_Edits) ]
```

* **What the variables mean:**
  * `Edit_Distance`: Minimum insertions, deletions, substitutions (Levenshtein distance).
  * `E`: PRImA Human Effort Score (penalizes missing regions, boundary errors, and vertex editing).
* **Why this equation is needed:** Measures both raw transcription fidelity (**CER = 15.32%, WER = 9.97%**) and actual human scholar time saved (reduces editing time from 22.5 min/page down to 2.9 min/page -> **75.4% time reduction**).
