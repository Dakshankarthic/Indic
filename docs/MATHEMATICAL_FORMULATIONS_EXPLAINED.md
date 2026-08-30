# Comprehensive Mathematical Formulations & Algorithms
### Automated Document Layout Analysis and Hierarchical Transcription for Historical Indic Manuscripts

---

# Section 1: Self-Supervised Vision Transformer (DINOv2) Mechanics

## 1.1 Patch Embedding & 2D Position Encoding
```
x_0 = [ x_cls;  x_p^1 * E;  x_p^2 * E;  ...;  x_p^N * E ] + E_pos
where:
  E in R^((P^2 * C) x D)   (Linear projection matrix, P = 14, C = 3, D = 768)
  E_pos in R^((N+1) x D)   (Learnable 2D positional encodings)
```
* **Variables:**
  * `P = 14`: Patch dimension (14x14 pixels per visual token).
  * `N = (H'/14) * (W'/14)`: Total count of patches across the manuscript folio.
  * `x_cls`: Global classification token capturing overall illumination.

---

## 1.2 Multi-Head Self-Attention (MHSA) Spatial Manifold Extraction
```
Q = X * W_Q,   K = X * W_K,   V = X * W_V

Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V

MultiHead(X) = [ head_1; head_2; ...; head_12 ] * W_O   (h = 12 heads, d_k = 64)
```
* **Variables:**
  * `(Q * K^T) / sqrt(d_k)`: Scaled dot-product pairwise affinity matrix measuring structural correlation between distant patches.
  * `softmax(...)`: Normalizes attention weights across the entire page (summing to 1 across rows).
  * `h = 12`: 12 distinct attention heads tracking character lines, margins, headlines, and damage independently.

---

## 1.3 Spatial Feature Grid & Cosine Similarity Affinity
```
F = Reshape( Z_{1:N}, (H'/14, W'/14, 768) )

A_{ij} = (z_i · z_j) / ( ||z_i|| * ||z_j|| ) = cos(θ_{ij})
```
* **Variables:**
  * `Z`: 768-dimensional token representations from the final Transformer block.
  * `A_{ij}`: Cosine similarity between feature vector `z_i` and `z_j`.
  * Text patches have high similarity (`A_{ij} > 0.85`), while blank substrate patches cluster at `A_{ij} < 0.20`.

---

# Section 2: 6-Channel nnU-Net Deep Semantic Segmentation

## 2.1 Feature Propagation & Instance Normalization
```
x^(l) = LeakyReLU( IN( W_2 * LeakyReLU( IN( W_1 * x^(l-1) ) ) ) )

InstanceNorm(v) = gamma * ( (v - Mean(v)) / sqrt(Variance(v) + eps) ) + beta
where Mean and Variance are computed across the (H, W) spatial dimensions of each folio.
```
* **Why InstanceNorm:** Normalizes each folio independently, preventing distortion across varying resolutions and dimensions.

---

## 2.2 6-Channel Softmax Probability Distribution
```
P(Class = c | x, y) = exp( z_c(x, y) ) / Sum_{k=1}^6 exp( z_k(x, y) )

The 6 Target Classes:
  c = 1: TextRegion (Main body of Sanskrit verses)
  c = 2: Marginalia (Margin glosses, header titles, folio numbers)
  c = 3: GraphicRegion (Woodblock illustrations and paintings)
  c = 4: PageFrame (Outer physical substrate perimeter)
  c = 5: Damage (Punched string binder holes and tears)
  c = 6: TextLine (Individual line baselines and polygons)
```

---

## 2.3 Multi-Scale Deep Supervision Compound Loss
```
Loss_total = Sum_{s=0}^2 w_s [ lambda_1 * Loss_Focal(s) + lambda_2 * Loss_Dice(s) ]
where w = [1.0, 0.5, 0.25] across scales 512x512, 256x256, 128x128.

Loss_Focal = - (1/N) * Sum [ alpha_t * (1 - p_t)^gamma * log(p_t) ]   (gamma = 2.0, alpha_t = 0.25)

Loss_Dice = 1 - [ (2 * Sum p_t * y_t + eps) / (Sum p_t^2 + Sum y_t^2 + eps) ]
```
* **Why this loss:** Eliminates extreme class imbalance (text lines occupy only ~8% of the folio area).

---

# Section 3: Optical Character Recognition (OCR & HTR) Mechanics

## 3.1 Recurrent Feature Sequence Formulation (CRNN)
```
x_t = CNN( Line_Crop[:, t] ),   for t = 1, 2, ..., T

h_t = [ Forward_LSTM(x_t, h_{t-1});  Backward_LSTM(x_t, h_{t+1}) ] in R^(2 * D_hidden)

y_t = softmax( W_out * h_t + b_out ) in R^(|Vocabulary| + 1)
```
* **Variables:**
  * `T`: Horizontal time steps across the line crop.
  * `h_t`: Bidirectional LSTM state capturing characters before and after the current position.
  * `Vocabulary`: 128 Devanagari consonants, matras, viramas, numerals + 1 blank token `eps`.

---

## 3.2 Connectionist Temporal Classification (CTC) Conditional Probability
```
Path Probability:       P(pi | X) = Product_{t=1}^T y_{pi_t}^t

Sequence Probability:   P(Y | X) = Sum_{pi in B^(-1)(Y)} P(pi | X)

CTC Training Loss:      Loss_CTC = - ln P(Y* | X) = - ln Sum_{pi in B^(-1)(Y*)} Product_{t=1}^T y_{pi_t}^t
```
* **Variables:**
  * `pi`: Frame-level character alignment path.
  * `B`: Collapse operator removing consecutive duplicate characters and blank tokens:
    `B( "क" eps "क" "क" eps "ष" ) = "क" "क" "ष"`
  * `Y*`: Ground-truth Sanskrit text string.

---

## 3.3 Beam Search Decoding with Devanagari Language Model
```
Y_hat = argmax_Y [ log P_CTC(Y | X) + alpha * log P_LM(Y) + beta * |Y| ]

where P_LM(Y) = Product_{i=1}^M P(word_i | word_{i-1}, word_{i-2}, ..., word_{i-n+1})
```
* **Variables:**
  * `Y_hat`: Optimal decoded Sanskrit transcription.
  * `alpha = 0.65`: Language Model weight enforcing Sanskrit grammar and legal root stems.
  * `beta = 1.20`: Word insertion bonus preventing over-shortening of words.
