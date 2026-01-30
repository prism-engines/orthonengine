# PRISM/ORTHON Engine Reference

> **"Failure is loss of coherent structure — geometrically, dynamically, topologically, and causally."**

This document catalogs all 116 primitives and 21 engines in the PRISM computation library, with their mathematical foundations, physical interpretations, and role in the four-pillar structural health framework.

**Final tally after PR #14 (FINAL):**
- **Primitives:** 116 (Y1-Y9)
- **Engines:** 21 (Y10-Y14)
- **SQL Reports:** 52
- **Total Capabilities:** 189

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [The Four Pillars](#the-four-pillars)
3. [Primitives (Y1-Y9)](#primitives-y1-y9)
   - [Y1: Individual Primitives](#y1-individual-primitives-35)
   - [Y2: Pairwise Primitives](#y2-pairwise-primitives-20)
   - [Y3: Matrix Primitives](#y3-matrix-primitives-10)
   - [Y4: Embedding Primitives](#y4-embedding-primitives-4)
   - [Y5: Topology Primitives](#y5-topology-primitives-5)
   - [Y6: Network Primitives](#y6-network-primitives-11)
   - [Y7: Dynamical Primitives](#y7-dynamical-primitives-10)
   - [Y8: Test Primitives](#y8-test-primitives-12)
   - [Y9: Information Primitives](#y9-information-primitives-9)
4. [Engines (Y10-Y13)](#engines-y10-y13)
   - [Y10: Structure Engines](#y10-structure-engines-5-engines-8-sql)
   - [Y11: Physics Engines](#y11-physics-engines-4-engines-7-sql)
   - [Y12: Dynamics Engines](#y12-dynamics-engines-4-engines-6-sql)
   - [Y13: Advanced Engines](#y13-advanced-engines-4-engines-10-sql)
   - [Y14: Statistics Engines](#y14-statistics-engines-4-engines-12-sql)
5. [Output Schema](#output-schema)
6. [Interpretation Guide](#interpretation-guide)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAW SENSOR DATA                                │
│                         (observations.parquet)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRIMITIVES (Y1-Y9)                                  │
│                        116 raw computations                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │ Y1: Indiv│ │ Y2: Pair │ │ Y3: Matrix│ │Y4: Embed │ │Y5: Topo  │         │
│  │   (35)   │ │   (20)   │ │   (10)   │ │   (4)    │ │   (5)    │         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
│  │Y6: Network│ │Y7: Dynam │ │ Y8: Test │ │ Y9: Info │                      │
│  │   (11)   │ │   (10)   │ │   (12)   │ │   (9)    │                      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ENGINES (Y10-Y14)                                  │
│                    21 composed interpretations                              │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌──────────────┐│
│  │ Y10: Structure │ │ Y11: Physics   │ │ Y12: Dynamics  │ │ Y13: Advanced││
│  │   5 engines    │ │   4 engines    │ │   4 engines    │ │   4 engines  ││
│  │   8 SQL        │ │   7 SQL        │ │   8 SQL        │ │  10 SQL      ││
│  └────────────────┘ └────────────────┘ └────────────────┘ └──────────────┘│
│                           ┌──────────────────┐                             │
│                           │ Y14: Statistics  │                             │
│                           │    4 engines     │                             │
│                           │   12 SQL         │                             │
│                           └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORTHON INTERPRETATION                                │
│                     Four-Pillar Health Assessment                           │
│         ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│         │ GEOMETRY │ │ DYNAMICS │ │ TOPOLOGY │ │   INFO   │               │
│         │Coherence │ │ Lyapunov │ │  Betti   │ │ Entropy  │               │
│         │ Coupling │ │   RQA    │ │Persistence│ │ Causality│               │
│         └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Four Pillars

| Pillar | Question | Key Metrics | Healthy | Failing |
|--------|----------|-------------|---------|---------|
| **Geometry** | How are signals related? | Coherence, effective dimension, coupling | Stable coupling | Decoupling or over-coupling |
| **Dynamics** | How stable is the system? | Lyapunov, RQA, attractors | λ < 0 (stable) | λ > 0 (chaotic) |
| **Topology** | What is the shape of behavior? | Betti numbers, persistence | β₀=1, β₁=1 (clean cycle) | β₀>1 (fragmented) |
| **Information** | Who drives whom? | Transfer entropy, hierarchy | Clear hierarchy | Circular causation |

---

## Primitives (Y1-Y9)

### Y1: Individual Primitives (35)

Per-signal computations that characterize univariate time series behavior.

#### Statistical Moments

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `mean` | $\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$ | Central tendency | Baseline reference |
| `std` | $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2}$ | Spread/variability | Volatility indicator |
| `var` | $\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2$ | Variance | Energy measure |
| `skew` | $\gamma_1 = \frac{E[(X-\mu)^3]}{\sigma^3}$ | Asymmetry | Distribution shape |
| `kurtosis` | $\gamma_2 = \frac{E[(X-\mu)^4]}{\sigma^4} - 3$ | Tail heaviness | Extreme event likelihood |
| `rms` | $x_{rms} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$ | Root mean square | Energy content |
| `crest_factor` | $CF = \frac{|x|_{max}}{x_{rms}}$ | Peak-to-RMS ratio | Impulsiveness |
| `peak_to_peak` | $x_{pp} = x_{max} - x_{min}$ | Total range | Operating envelope |

#### Trend Analysis

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `trend_slope` | $m = \frac{\sum(x_i - \bar{x})(t_i - \bar{t})}{\sum(t_i - \bar{t})^2}$ | Linear drift rate | Degradation velocity |
| `trend_intercept` | $b = \bar{x} - m\bar{t}$ | Baseline level | Starting point |
| `trend_r2` | $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$ | Trend fit quality | Trend confidence |
| `detrend_std` | $\sigma_{detrended}$ | Residual variability | Noise after trend removal |

#### Long-Range Dependence

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `hurst_rs` | $E[R(n)/S(n)] = Cn^H$ | Rescaled range Hurst | Long-range memory |
| `hurst_dfa` | $F(n) \sim n^H$ via DFA | Detrended fluctuation | Memory with nonstationarity |
| `hurst_wavelet` | $H = \frac{1}{2}(\beta + 1)$ from wavelet spectrum | Wavelet-based Hurst | Multi-scale memory |

**Interpretation:**
- H = 0.5: Random walk (no memory)
- H > 0.5: Persistent (trends continue)
- H < 0.5: Anti-persistent (mean-reverting)

#### Entropy & Complexity

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `sample_entropy` | $SampEn = -\ln\frac{A}{B}$ | Regularity/predictability | System complexity |
| `permutation_entropy` | $H_p = -\sum p(\pi)\ln p(\pi)$ | Ordinal pattern distribution | Nonlinear complexity |
| `approximate_entropy` | $ApEn = \phi^m(r) - \phi^{m+1}(r)$ | Self-similarity | Regularity at scale |
| `spectral_entropy` | $H_s = -\sum P(f)\ln P(f)$ | Spectral uniformity | Frequency complexity |
| `svd_entropy` | $H_{svd} = -\sum \sigma_i \ln \sigma_i$ | Singular value distribution | Embedding complexity |

**Interpretation:**
- High entropy: Complex, unpredictable
- Low entropy: Regular, predictable
- Sudden changes: Regime transitions

#### Autocorrelation

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `acf_lag1` | $\rho_1 = \frac{E[(X_t - \mu)(X_{t-1} - \mu)]}{\sigma^2}$ | First-order dependence | Short-term memory |
| `acf_decay_rate` | $\rho_k \sim e^{-k/\tau}$ | Exponential decay constant | Memory timescale |
| `acf_zero_crossing` | First k where $\rho_k < 0$ | Decorrelation lag | Independence scale |
| `partial_acf_lag1` | $\phi_{11}$ from Yule-Walker | Direct lag-1 effect | AR(1) coefficient |

#### Spectral Features

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `dominant_freq` | $f_{dom} = \arg\max_f |X(f)|^2$ | Primary oscillation | Characteristic frequency |
| `spectral_centroid` | $f_c = \frac{\sum f \cdot P(f)}{\sum P(f)}$ | Spectral center of mass | "Brightness" |
| `spectral_bandwidth` | $BW = \sqrt{\frac{\sum (f-f_c)^2 P(f)}{\sum P(f)}}$ | Spectral spread | Frequency variability |
| `spectral_slope` | $P(f) \sim f^{-\beta}$ | Power law exponent | Self-similarity (1/f noise) |
| `spectral_rolloff` | $f_{85\%}$ where $\sum_{f<f_r} P(f) = 0.85 \sum P(f)$ | High-frequency content | Energy distribution |
| `band_power_low` | $\int_{f_1}^{f_2} P(f) df$ | Low-frequency energy | Slow oscillations |
| `band_power_mid` | $\int_{f_2}^{f_3} P(f) df$ | Mid-frequency energy | Process dynamics |
| `band_power_high` | $\int_{f_3}^{f_4} P(f) df$ | High-frequency energy | Fast oscillations/noise |

#### Nonlinear Features

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `zero_crossing_rate` | $ZCR = \frac{1}{N}\sum \mathbb{1}[x_t x_{t-1} < 0]$ | Sign change frequency | Oscillation rate |
| `mean_crossing_rate` | $MCR = \frac{1}{N}\sum \mathbb{1}[(x_t - \bar{x})(x_{t-1} - \bar{x}) < 0]$ | Mean crossing frequency | Deviation frequency |
| `turning_points` | Count of local extrema | Peak/trough density | Signal roughness |

---

### Y2: Pairwise Primitives (20)

Signal-pair computations for relationship analysis.

#### Correlation Measures

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `pearson_corr` | $\rho_{XY} = \frac{Cov(X,Y)}{\sigma_X \sigma_Y}$ | Linear dependence | Basic coupling |
| `spearman_corr` | $\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$ | Rank correlation | Monotonic relationship |
| `kendall_tau` | $\tau = \frac{C - D}{\binom{n}{2}}$ | Concordance | Ordinal association |
| `partial_corr` | $\rho_{XY|Z}$ | Correlation controlling for Z | Direct relationship |
| `cross_corr_max` | $\max_\tau R_{XY}(\tau)$ | Peak cross-correlation | Maximum coupling |
| `cross_corr_lag` | $\arg\max_\tau R_{XY}(\tau)$ | Lag at peak | Lead/lag relationship |

#### Information-Theoretic

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `mutual_info` | $I(X;Y) = \sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$ | Shared information | Nonlinear dependence |
| `normalized_mi` | $NMI = \frac{I(X;Y)}{\sqrt{H(X)H(Y)}}$ | Normalized MI | Comparable across pairs |
| `conditional_entropy` | $H(Y|X) = -\sum p(x,y)\log p(y|x)$ | Remaining uncertainty | Predictability |

#### Distance Measures

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `dtw_distance` | $DTW(X,Y) = \min_{\pi} \sum d(x_{\pi_x(k)}, y_{\pi_y(k)})$ | Warped distance | Shape similarity |
| `euclidean_dist` | $d = \sqrt{\sum (x_i - y_i)^2}$ | L2 distance | Point-wise difference |
| `cosine_similarity` | $\cos\theta = \frac{X \cdot Y}{||X|| ||Y||}$ | Angular similarity | Direction alignment |

#### Coherence & Phase

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `coherence` | $C_{xy}(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f)S_{yy}(f)}$ | Frequency-domain correlation | Spectral coupling |
| `phase_lag` | $\phi = \arg(S_{xy}(f))$ | Phase difference | Temporal offset |
| `instantaneous_phase_sync` | $R = |\frac{1}{N}\sum e^{i(\phi_x(t) - \phi_y(t))}|$ | Phase locking | Synchronization |

#### Causality (Basic)

| Primitive | Equation | Measures | Why Use It |
|-----------|----------|----------|------------|
| `granger_f_stat` | $F = \frac{(RSS_r - RSS_u)/p}{RSS_u/(n-2p-1)}$ | Granger F-statistic | Predictive causality |
| `granger_p_value` | p-value from F-test | Statistical significance | Causality confidence |
| `ccm_score` | Convergent cross-mapping | Nonlinear causality | State-space causality |
| `optimal_lag` | $\arg\min_\tau AIC(\tau)$ | Best prediction lag | Causal timescale |

---

### Y3: Matrix Primitives (10)

Full observation matrix computations across all signals.

| Primitive | Equation | Measures | Why Use It | Pillar |
|-----------|----------|----------|------------|--------|
| `covariance_matrix` | $\Sigma_{ij} = Cov(X_i, X_j)$ | Signal covariances | Relationship structure | Geometry |
| `correlation_matrix` | $R_{ij} = \frac{\Sigma_{ij}}{\sigma_i \sigma_j}$ | Normalized covariances | Coupling strength | Geometry |
| `eigenvalues` | $\Sigma v = \lambda v$ | Principal variances | Mode energies | Geometry |
| `eigenvectors` | Principal directions | Mode shapes | Coupling patterns | Geometry |
| `effective_dimension` | $d_{eff} = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}$ | Participation ratio | Complexity measure | Geometry |
| `coherence_ratio` | $\frac{\lambda_1}{\sum \lambda_i}$ | Dominance of first mode | Collective behavior | Geometry |
| `condition_number` | $\kappa = \frac{\lambda_{max}}{\lambda_{min}}$ | Matrix ill-conditioning | Numerical stability | Geometry |
| `trace` | $tr(\Sigma) = \sum \lambda_i$ | Total variance | System energy | Geometry |
| `determinant` | $det(\Sigma) = \prod \lambda_i$ | Generalized variance | Volume measure | Geometry |
| `frobenius_norm` | $||\Sigma||_F = \sqrt{\sum \sigma_{ij}^2}$ | Matrix magnitude | Overall coupling | Geometry |

---

### Y4: Embedding Primitives (4)

Phase space reconstruction via Takens' theorem.

| Primitive | Equation | Measures | Why Use It | Pillar |
|-----------|----------|----------|------------|--------|
| `embedding_delay` | $\tau = \arg\min_\tau MI(x(t), x(t+\tau))$ | Optimal time delay | Independence scale | Dynamics |
| `embedding_dimension` | $d_E$ via FNN or Cao's method | Minimum embedding | Attractor dimension | Dynamics |
| `delay_vector` | $\vec{x}(t) = [x(t), x(t-\tau), ..., x(t-(d-1)\tau)]$ | Reconstructed state | Phase space point | Dynamics |
| `reconstruction_quality` | $Q = 1 - \frac{FNN(d_E)}{FNN(1)}$ | Embedding goodness | Unfolding quality | Dynamics |

**Takens' Theorem:**
For a dynamical system with attractor dimension $d_A$, an embedding in $d_E \geq 2d_A + 1$ dimensions preserves topological properties.

---

### Y5: Topology Primitives (5)

Persistent homology and topological data analysis.

| Primitive | Equation | Measures | Why Use It | Pillar |
|-----------|----------|----------|------------|--------|
| `betti_0` | $\beta_0$ = # connected components | Fragmentation | Attractor structure | Topology |
| `betti_1` | $\beta_1$ = # 1-dimensional holes | Loops/cycles | Periodic structure | Topology |
| `betti_2` | $\beta_2$ = # 2-dimensional voids | Cavities | Higher structure | Topology |
| `persistence_diagram` | Birth-death pairs $(b_i, d_i)$ | Feature lifetimes | Topological robustness | Topology |
| `persistence_entropy` | $H_p = -\sum p_i \ln p_i$ where $p_i = \frac{d_i - b_i}{\sum(d_j - b_j)}$ | Persistence distribution | Topological complexity | Topology |

**Interpretation:**
- β₀ = 1: Single connected attractor (healthy)
- β₀ > 1: Fragmented attractor (critical)
- β₁ = 1: Limit cycle (healthy oscillation)
- β₁ = 0: Collapsed dynamics (fixed point)
- β₁ > 2: Complex/chaotic dynamics

---

### Y6: Network Primitives (11)

Graph-theoretic measures on signal interaction networks.

| Primitive | Equation | Measures | Why Use It | Pillar |
|-----------|----------|----------|------------|--------|
| `degree_centrality` | $C_D(v) = \frac{deg(v)}{n-1}$ | Connection count | Hub identification | Information |
| `betweenness_centrality` | $C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$ | Path intermediary | Bridge signals | Information |
| `closeness_centrality` | $C_C(v) = \frac{n-1}{\sum_u d(v,u)}$ | Average distance | Global influence | Information |
| `eigenvector_centrality` | $Ax = \lambda x$ | Influence propagation | Important neighbors | Information |
| `clustering_coefficient` | $C_i = \frac{2e_i}{k_i(k_i-1)}$ | Local density | Clique formation | Information |
| `global_efficiency` | $E = \frac{1}{n(n-1)}\sum_{i \neq j} \frac{1}{d_{ij}}$ | Network integration | Communication efficiency | Information |
| `modularity` | $Q = \frac{1}{2m}\sum_{ij}[A_{ij} - \frac{k_i k_j}{2m}]\delta(c_i, c_j)$ | Community structure | Signal grouping | Information |
| `assortativity` | $r = \frac{\sum_{jk}jk(e_{jk} - q_j q_k)}{\sigma_q^2}$ | Degree correlation | Network organization | Information |
| `average_path_length` | $L = \frac{1}{n(n-1)}\sum_{i \neq j} d_{ij}$ | Typical separation | Information spread | Information |
| `network_density` | $\rho = \frac{2m}{n(n-1)}$ | Edge fraction | Coupling density | Information |
| `rich_club_coefficient` | $\phi(k) = \frac{2E_{>k}}{N_{>k}(N_{>k}-1)}$ | Hub interconnection | Elite structure | Information |

---

### Y7: Dynamical Primitives (10)

Lyapunov exponents, recurrence quantification, and attractor properties.

#### Lyapunov Exponents

| Primitive | Equation | Measures | Why Use It | Pillar |
|-----------|----------|----------|------------|--------|
| `lyapunov_max` | $\lambda_{max} = \lim_{t \to \infty} \frac{1}{t} \ln \frac{||\delta x(t)||}{||\delta x(0)||}$ | Maximum divergence rate | Chaos indicator | Dynamics |
| `lyapunov_spectrum` | $\{\lambda_1, \lambda_2, ..., \lambda_n\}$ | Full exponent set | Attractor dimension | Dynamics |
| `kaplan_yorke_dim` | $D_{KY} = j + \frac{\sum_{i=1}^j \lambda_i}{|\lambda_{j+1}|}$ | Fractal dimension | Attractor complexity | Dynamics |

**Interpretation:**
- λ_max < 0: Stable (trajectories converge)
- λ_max ≈ 0: Marginal (limit cycle)
- λ_max > 0: Chaotic (sensitive dependence)

#### Recurrence Quantification Analysis (RQA)

| Primitive | Equation | Measures | Why Use It | Pillar |
|-----------|----------|----------|------------|--------|
| `recurrence_rate` | $RR = \frac{1}{N^2}\sum_{i,j} R_{ij}$ | State recurrence | System regularity | Dynamics |
| `determinism` | $DET = \frac{\sum_{l=l_{min}}^N l \cdot P(l)}{\sum_{l=1}^N l \cdot P(l)}$ | Diagonal line fraction | Predictability | Dynamics |
| `laminarity` | $LAM = \frac{\sum_{v=v_{min}}^N v \cdot P(v)}{\sum_{v=1}^N v \cdot P(v)}$ | Vertical line fraction | Laminar states | Dynamics |
| `trapping_time` | $TT = \frac{\sum_{v=v_{min}}^N v \cdot P(v)}{\sum_{v=v_{min}}^N P(v)}$ | Average vertical length | State persistence | Dynamics |
| `rqa_entropy` | $ENTR = -\sum P(l) \ln P(l)$ | Diagonal length distribution | Complexity | Dynamics |

#### Attractor Properties

| Primitive | Equation | Measures | Why Use It | Pillar |
|-----------|----------|----------|------------|--------|
| `correlation_dimension` | $D_2 = \lim_{r \to 0} \frac{\ln C(r)}{\ln r}$ | Grassberger-Procaccia | Attractor fractal dim | Dynamics |

---

### Y8: Test Primitives (12)

Statistical hypothesis tests for time series properties.

| Primitive | Equation | Measures | Why Use It | Output |
|-----------|----------|----------|------------|--------|
| `adf_statistic` | ADF test statistic | Unit root presence | Stationarity test | p-value |
| `adf_pvalue` | p-value from ADF | Significance | Stationarity confidence | p-value |
| `kpss_statistic` | KPSS test statistic | Trend stationarity | Stationarity around trend | p-value |
| `kpss_pvalue` | p-value from KPSS | Significance | Trend-stationarity confidence | p-value |
| `ljung_box_stat` | $Q = n(n+2)\sum_{k=1}^h \frac{\hat{\rho}_k^2}{n-k}$ | Autocorrelation presence | White noise test | p-value |
| `ljung_box_pvalue` | p-value from Ljung-Box | Significance | Independence test | p-value |
| `jarque_bera_stat` | $JB = \frac{n}{6}(S^2 + \frac{(K-3)^2}{4})$ | Normality test | Distribution shape | p-value |
| `jarque_bera_pvalue` | p-value from JB | Significance | Normality confidence | p-value |
| `levene_stat` | Levene's test statistic | Variance equality | Homoscedasticity | p-value |
| `levene_pvalue` | p-value from Levene | Significance | Variance stability | p-value |
| `runs_test_stat` | Runs test statistic | Randomness | Sequence independence | p-value |
| `runs_test_pvalue` | p-value from runs test | Significance | Randomness confidence | p-value |

---

### Y9: Information Primitives (9)

Entropy measures and causal inference.

#### Transfer Entropy

| Primitive | Equation | Measures | Why Use It | Pillar |
|-----------|----------|----------|------------|--------|
| `transfer_entropy_xy` | $TE_{X \to Y} = \sum p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) \log \frac{p(y_{t+1}|y_t^{(k)}, x_t^{(l)})}{p(y_{t+1}|y_t^{(k)})}$ | Information flow X→Y | Causal influence | Information |
| `transfer_entropy_yx` | $TE_{Y \to X}$ | Information flow Y→X | Reverse causality | Information |
| `net_transfer_entropy` | $nTE = TE_{X \to Y} - TE_{Y \to X}$ | Net information flow | Dominant direction | Information |
| `effective_te` | Shuffled-corrected TE | Bias-corrected flow | True causal strength | Information |

#### Complexity Measures

| Primitive | Equation | Measures | Why Use It | Pillar |
|-----------|----------|----------|------------|--------|
| `lempel_ziv_complexity` | LZ76 complexity | Algorithmic randomness | Compressibility | Information |
| `kolmogorov_complexity` | Approximated K(x) | Descriptive complexity | Information content | Information |
| `multiscale_entropy` | $MSE(\tau) = SampEn(y^{(\tau)})$ | Scale-dependent entropy | Complexity across scales | Information |
| `fisher_information` | $F = E[(\frac{\partial}{\partial \theta} \ln p)^2]$ | Parameter sensitivity | Estimation precision | Information |
| `active_information` | $A = I(X_{t-1}; X_t)$ | Past-present MI | Memory utilization | Information |

---

## Engines (Y10-Y13)

Engines compose primitives into interpretable analyses with associated SQL views.

### Y10: Structure Engines (5 engines, 8 SQL)

**Purpose:** Analyze signal relationships and system structure.

#### Engines

| Engine | Inputs | Outputs | Measures |
|--------|--------|---------|----------|
| `covariance_engine` | observations | covariance_matrix, eigenvalues | Signal coupling structure |
| `eigenvalue_engine` | covariance_matrix | eigenvalues, eigenvectors, eff_dim | Modal decomposition |
| `koopman_engine` | observations | koopman_modes, koopman_eigenvalues | Linear approximation of nonlinear dynamics |
| `spectral_engine` | observations | power_spectrum, dominant_freqs | Frequency content |
| `wavelet_engine` | observations | wavelet_coeffs, scale_energy | Time-frequency decomposition |

#### SQL Views

| SQL File | View Created | Purpose |
|----------|--------------|---------|
| `10_covariance_analysis.sql` | `v_covariance_summary` | Coupling matrices per window |
| `11_eigenvalue_decomposition.sql` | `v_eigen_health` | Modal health assessment |
| `12_effective_dimension.sql` | `v_dimension_tracking` | Complexity evolution |
| `13_coherence_analysis.sql` | `v_coherence_state` | Coupling interpretation |
| `14_spectral_analysis.sql` | `v_spectral_health` | Frequency health |
| `15_koopman_modes.sql` | `v_koopman_stability` | Linear stability |
| `16_wavelet_decomposition.sql` | `v_wavelet_energy` | Scale-dependent energy |
| `17_structure_summary.sql` | `v_structure_health` | Unified structure score |

---

### Y11: Physics Engines (4 engines, 7 SQL)

**Purpose:** Compute physical quantities from sensor data.

#### Engines

| Engine | Inputs | Outputs | Measures |
|--------|--------|---------|----------|
| `energy_engine` | velocity, mass | kinetic_energy, potential_energy, total_energy | System energetics |
| `mass_engine` | density, flow_rate | mass_flow, accumulation | Material balance |
| `momentum_engine` | velocity, mass | momentum, angular_momentum | Dynamic balance |
| `constitutive_engine` | stress, strain | youngs_modulus, viscosity | Material properties |

#### SQL Views

| SQL File | View Created | Purpose |
|----------|--------------|---------|
| `20_energy_balance.sql` | `v_energy_health` | Energy conservation |
| `21_mass_balance.sql` | `v_mass_health` | Mass conservation |
| `22_momentum_balance.sql` | `v_momentum_health` | Momentum conservation |
| `23_constitutive_analysis.sql` | `v_material_health` | Material consistency |
| `24_thermodynamic_health.sql` | `v_thermo_health` | L4 thermodynamic state |
| `25_conservation_violations.sql` | `v_conservation_alerts` | Balance violations |
| `26_physics_summary.sql` | `v_physics_health` | Unified physics score |

---

### Y12: Dynamics Engines (4 engines, 6 SQL)

**Purpose:** Characterize dynamical system behavior.

#### Engines

| Engine | Inputs | Outputs | Measures |
|--------|--------|---------|----------|
| `lyapunov_engine` | embedded_states | lyapunov_max, lyapunov_spectrum | Stability/chaos |
| `attractor_engine` | embedded_states | correlation_dim, kaplan_yorke_dim | Attractor properties |
| `recurrence_engine` | embedded_states | RR, DET, LAM, TT, ENTR | Recurrence structure |
| `bifurcation_engine` | time_series | bifurcation_points, regime_changes | Stability transitions |

#### SQL Views

| SQL File | View Created | Purpose |
|----------|--------------|---------|
| `30_dynamics_stability.sql` | `v_dynamics_summary` | Stability classification |
| `31_regime_transitions.sql` | `v_regime_changes` | Transition detection |
| `32_basin_stability.sql` | `v_basin_stability` | Basin analysis |
| `33_birth_certificate.sql` | `v_birth_prognosis` | Early-life prediction |
| `34_rqa_analysis.sql` | `v_rqa_health` | Recurrence health |
| `35_dynamics_summary.sql` | `v_dynamics_health` | Unified dynamics score |

---

### Y13: Advanced Engines (4 engines, 10 SQL) — PR #13 FINAL

**Purpose:** Causal inference, topology, emergence detection, and unified health assessment.

> **This completes the PRISM build:** causal network analysis, topological data analysis, information-theoretic emergence detection, and a master integration engine that combines all metrics into unified health assessment.

#### Engines

| # | Engine | Inputs | Key Outputs | Purpose |
|---|--------|--------|-------------|---------|
| 130 | `causality_engine` | observations | granger_f, transfer_entropy, hierarchy, n_feedback_loops, top_driver, top_sink, bottleneck | Causal network analysis via Granger + TE |
| 131 | `topology_engine` | embedded_states | betti_0, betti_1, betti_2, persistence_entropy, fragmentation, topology_change | Persistent homology for attractor analysis |
| 132 | `emergence_engine` | observations | mutual_information, redundancy, unique_1, unique_2, synergy, synergy_ratio | PID-based emergence/synergy detection |
| 133 | `integration_engine` | all_parquets | health_score (0-100), risk_level, recommendation, primary_concern | **Master engine** — unified health assessment |

#### Engine Details

##### 130: Causality Engine

Computes causal networks using Granger causality and Transfer Entropy.

**Equations:**
$$\text{Granger: } F_{i \to j} = \frac{(RSS_{reduced} - RSS_{full}) / p}{RSS_{full} / (n - 2p - 1)}$$

$$\text{Transfer Entropy: } T_{X \to Y} = \sum p(y_{t+1}, y_t^k, x_t^l) \log \frac{p(y_{t+1} | y_t^k, x_t^l)}{p(y_{t+1} | y_t^k)}$$

$$\text{Hierarchy: } H = 1 - \frac{\text{reciprocal edges}}{\text{total edges}}$$

**Outputs:** edges parquet + network parquet

##### 131: Topology Engine

Computes persistent homology metrics for attractor analysis.

**Equations:**
$$\beta_0 = \text{connected components}, \quad \beta_1 = \text{loops/holes}, \quad \beta_2 = \text{voids}$$

$$H_{persist} = -\sum_i \frac{p_i}{\sum p} \log \frac{p_i}{\sum p}$$

$$W_q(D_1, D_2) = \inf_\gamma \left( \sum_{i} ||x_i - \gamma(x_i)||_\infty^q \right)^{1/q}$$

**Detects:** Attractor fragmentation, structural changes, topology evolution

##### 132: Emergence Engine

Computes synergy, redundancy, unique information via Partial Information Decomposition (PID).

**Equations:**
$$I(X_1, X_2; Y) = \underbrace{R}_{\text{redundancy}} + \underbrace{U_1}_{\text{unique}_1} + \underbrace{U_2}_{\text{unique}_2} + \underbrace{S}_{\text{synergy}}$$

$$\text{Emergence ratio} = \frac{S}{R + U_1 + U_2 + S}$$

**Identifies:** Multi-signal interactions that pairwise analysis misses

##### 133: Integration Engine (Master)

Combines all metrics into unified health assessment.

**Equations:**
$$\text{Health} = 100 \times (1 - \text{Risk})$$

$$\text{Risk} = w_s \cdot S_{stability} + w_p \cdot S_{predictability} + w_{ph} \cdot S_{physics} + w_t \cdot S_{topology} + w_c \cdot S_{causality}$$

**Default weights:** $w_s = 0.25, w_p = 0.20, w_{ph} = 0.25, w_t = 0.15, w_c = 0.15$

**Outputs:**
- `health_score`: 0-100 (higher = healthier)
- `risk_level`: LOW / MODERATE / HIGH / CRITICAL
- `primary_concern`: What's wrong
- `recommendation`: What to do

| Risk Level | Health Score | Action |
|------------|--------------|--------|
| LOW | ≥ 80 | Normal operation |
| MODERATE | 60-79 | Monitor closely, plan maintenance |
| HIGH | 40-59 | Schedule maintenance within 1 week |
| CRITICAL | < 40 | IMMEDIATE INSPECTION REQUIRED |

#### SQL Views (10 reports)

| # | SQL File | View Created | Purpose |
|---|----------|--------------|---------|
| 164 | `30_causality_reports.sql` | `causality_significant_edges` | Strong causal links |
| 165 | `30_causality_reports.sql` | `causality_network_summary` | Network structure overview |
| 166 | `30_causality_reports.sql` | `causality_feedback_alerts` | Feedback loop warnings |
| 167 | `31_topology_reports.sql` | `topology_summary` | Topological state |
| 168 | `31_topology_reports.sql` | `topology_fragmentation_alerts` | Attractor fragmentation |
| 169 | `32_emergence_reports.sql` | `emergence_high_synergy` | Multi-signal emergence |
| 170 | `32_emergence_reports.sql` | `emergence_redundant` | Redundant sensors |
| 171 | `33_integration_reports.sql` | `health_dashboard` | Unified health view |
| 172 | `33_integration_reports.sql` | `health_critical_alerts` | High/critical risk entities |
| 173 | `33_integration_reports.sql` | `health_fleet_summary` | Fleet-wide statistics |

---

### Y14: Statistics Engines (4 engines, 12 SQL) — PR #14 FINAL

**Purpose:** Fleet-wide analytics, baseline computation, anomaly scoring, and report generation.

> **These engines operate ACROSS entities and windows** to provide comparative analytics, establish baselines, detect fleet-wide patterns, and generate executive summaries.

#### Why Statistics Engines?

| Question | Requires | Engine |
|----------|----------|--------|
| "Is this value normal?" | Baseline from fleet/history | `baseline_engine` |
| "Which entities are outliers?" | Fleet-wide comparison | `anomaly_engine` |
| "Who's performing best/worst?" | Rankings and clustering | `fleet_engine` |
| "Give me a one-page summary" | Report generation | `summary_engine` |

#### Engines

| # | Engine | Purpose | Key Outputs |
|---|--------|---------|-------------|
| 134 | `baseline_engine` | Compute fleet/entity baselines for all metrics | mean, std, percentiles, n_samples |
| 135 | `anomaly_engine` | Score deviations from baseline (z-scores) | z_score, percentile_rank, is_anomaly, anomaly_severity |
| 136 | `fleet_engine` | Rankings, clusters, cohort analysis | health_rank, cluster, health_tier, critical_events |
| 137 | `summary_engine` | Generate executive reports | report_section, content, priority |

#### Engine Details

##### 134: Baseline Engine

Computes fleet-wide and per-entity baselines for all metrics.

**Equations:**
$$\mu_{baseline} = \frac{1}{n}\sum_{i=1}^{n} x_i \quad \text{(first } n \text{ windows)}$$

$$\sigma_{baseline} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}$$

$$CV = \frac{\sigma}{\mu} \quad \text{(coefficient of variation)}$$

**Outputs:** Baseline statistics for every metric across fleet or per-entity

##### 135: Anomaly Engine

Scores deviations from baseline using z-scores.

**Equations:**
$$z = \frac{x - \mu_{baseline}}{\sigma_{baseline}}$$

| Severity | Z-Score | Action |
|----------|---------|--------|
| CRITICAL | \|z\| ≥ 3.0 | Immediate inspection |
| WARNING | \|z\| ≥ 2.0 | Investigate |
| ELEVATED | \|z\| ≥ 1.5 | Monitor |
| NORMAL | \|z\| < 1.5 | Continue |

##### 136: Fleet Engine

Rankings, clustering, and cohort analysis across all entities.

**Equations:**
$$\text{Rank}_i = \text{order}(\bar{h}_i) \quad \text{(by avg health)}$$

$$\text{Cluster} = k\text{-means}(\text{health}, \text{volatility}, \text{critical\_events})$$

**Health Tiers:**
| Tier | Avg Health |
|------|------------|
| HEALTHY | ≥ 80 |
| MODERATE | 60-79 |
| AT_RISK | 40-59 |
| CRITICAL | < 40 |

##### 137: Summary Engine

Generates executive summaries and reports.

**Output Sections:**
1. Executive Summary (fleet overview, KPIs)
2. Critical Alerts (entities needing attention)
3. Top Concerns (most common issues)
4. Anomaly Summary (deviations from normal)
5. Entity Rankings (best/worst performers)
6. Recommendations (actionable advice)
7. Trends (improving/degrading)

**Output Formats:** Parquet, Markdown, Text

#### SQL Views (12 reports)

| # | SQL File | View Created | Purpose |
|---|----------|--------------|---------|
| 174 | `40_baseline_reports.sql` | `baseline_overview` | All baseline statistics |
| 175 | `40_baseline_reports.sql` | `baseline_high_variance` | Metrics with high variability |
| 176 | `40_baseline_reports.sql` | `baseline_entity_vs_fleet` | Entity deviations from fleet |
| 177 | `41_anomaly_reports.sql` | `anomaly_current` | Current anomalies |
| 178 | `41_anomaly_reports.sql` | `anomaly_by_entity` | Anomaly counts per entity |
| 179 | `41_anomaly_reports.sql` | `anomaly_by_metric` | Anomaly counts per metric |
| 180 | `41_anomaly_reports.sql` | `anomaly_timeline` | Anomalies over time |
| 181 | `42_fleet_reports.sql` | `fleet_rankings` | Entity rankings |
| 182 | `42_fleet_reports.sql` | `fleet_clusters` | Cluster summary |
| 183 | `42_fleet_reports.sql` | `fleet_tier_distribution` | Health tier breakdown |
| 184 | `43_summary_reports.sql` | `summary_kpis` | Executive KPIs |
| 185 | `43_summary_reports.sql` | `summary_health_distribution` | Health score histogram |

---

## Output Schema

### Parquet Files

| File | Index | Contents |
|------|-------|----------|
| `observations.parquet` | entity_id, signal_id, I | Raw sensor data |
| `vector.parquet` | entity_id, signal_id, window | Per-signal primitives (Y1) |
| `pairs.parquet` | entity_id, signal_a, signal_b, window | Pairwise primitives (Y2) |
| `geometry.parquet` | entity_id, window | Matrix primitives (Y3) + Structure engines (Y10) |
| `dynamics.parquet` | entity_id, window | Embedding (Y4) + Dynamical (Y7) + Dynamics engines (Y12) |
| `topology.parquet` | entity_id, window | Topology primitives (Y5) + engines (Y13) |
| `information_flow.parquet` | entity_id, window | Network (Y6) + Information (Y9) + engines (Y13) |
| `physics.parquet` | entity_id, window | Physics engines (Y11) + ORTHON interpretation |
| `health.parquet` | entity_id, window | Integration engine (Y13) health assessment |
| `baseline.parquet` | metric_source, metric_name, entity_id | Baseline statistics (Y14) |
| `anomaly.parquet` | entity_id, window, metric | Anomaly scores (Y14) |
| `fleet_rankings.parquet` | entity_id | Entity rankings and clusters (Y14) |
| `summary.parquet` | report_section | Executive report sections (Y14) |

---

## Interpretation Guide

### Health Score Mapping

| Score Range | Status | Color | Action |
|-------------|--------|-------|--------|
| 0.8 - 1.0 | Healthy | Green | Monitor |
| 0.6 - 0.8 | Watch | Yellow | Investigate |
| 0.4 - 0.6 | Warning | Orange | Plan maintenance |
| 0.0 - 0.4 | Critical | Red | Immediate action |

### Cross-Pillar Patterns

| Geometry | Dynamics | Topology | Information | Diagnosis |
|----------|----------|----------|-------------|-----------|
| Coherence ↑ | λ < 0 | β₀=1, β₁=1 | Hierarchical | Healthy system |
| Coherence ↑↑ | λ → 0 | β₁ → 0 | Hierarchy ↑ | Rigidification (turbofan pattern) |
| Coherence ↓ | λ > 0 | β₀ > 1 | Circular | Fragmentation (bearing pattern) |
| Coupling ↓ | λ fluctuating | β₁ ↑ | Feedback ↑ | Instability onset |

### The Orthon Signal

The **Orthon Signal** (Ø) is detected when:
1. hd_slope > threshold (baseline deviation)
2. Confirmed by at least 2 pillars
3. Trajectory shows consistent direction

```
Ø = TRUE when:
  (hd_slope > μ + 2σ) AND
  (cross_layer_agreement > 0.6) AND
  (trajectory_consistency > 3 windows)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01 | Initial release with geometry, dynamics |
| 2.0 | 2025-01 | Added topology, information pillars |
| 3.0 | 2025-01 | Full primitive library (116) + engines (18) |
| 4.0 | 2025-01 | PR #13: Added causality, topology, emergence, integration engines |
| **5.0** | **2025-01** | **PR #14: FINAL** — Added statistics engines (baseline, anomaly, fleet, summary). Final: 116 primitives, 21 engines, 52 SQL reports (189 capabilities) |

---

## The Complete Picture

```
                    ┌─────────────────────────────────────┐
                    │         INTEGRATION ENGINE          │
                    │   Health Score │ Risk │ Recommend   │
                    └───────────────┬─────────────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬─────────────────┐
        │               │           │           │                 │
┌───────▼───────┐ ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐ ┌─────────▼─────────┐
│   STRUCTURE   │ │  PHYSICS  │ │DYNAMICS│ │ CAUSALITY │ │ TOPOLOGY/EMERGENCE│
│ Covariance    │ │ Energy    │ │Lyapunov│ │ Granger   │ │ Betti numbers     │
│ Eigenvalues   │ │ Mass      │ │RQA     │ │ TE Network│ │ Synergy           │
│ Koopman/DMD   │ │ Momentum  │ │CSD     │ │ Feedback  │ │ Redundancy        │
│ Spectral      │ │ Constitut.│ │Attractor│ │           │ │                   │
└───────────────┘ └───────────┘ └───────┘ └───────────┘ └───────────────────┘
        │               │           │           │                 │
        └───────────────┴───────────┼───────────┴─────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │      116 PRIMITIVES           │
                    │  Individual│Pairwise│Matrix   │
                    │  Embedding│Topology│Network   │
                    │  Dynamical│Tests│Information  │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │         RAW SIGNALS           │
                    │    entity_id × timestamp ×    │
                    │         signal values         │
                    └───────────────────────────────┘
```

**Input:** Time series signals
**Output:**
- **Health Score** (0-100)
- **Risk Level** (LOW/MODERATE/HIGH/CRITICAL)
- **Primary Concern** (what's wrong)
- **Recommendation** (what to do)

Plus full traceability through all intermediate metrics.

**PRISM is complete. Systems lose coherence before they fail. Now you can see it.**

---

*Document version: 5.0*
*Last updated: January 2025*
*Authors: Jason Rudder, Claude*

---

## Complete PR List

| PR | Name | Primitives | Engines | SQL |
|----|------|------------|---------|-----|
| 1 | Individual Primitives | 35 | - | - |
| 2 | Pairwise Primitives | 20 | - | - |
| 3 | Matrix Primitives | 10 | - | - |
| 4 | Embedding Primitives | 4 | - | - |
| 5 | Topology Primitives | 5 | - | - |
| 6 | Network Primitives | 11 | - | - |
| 7 | Dynamical Primitives | 10 | - | - |
| 8 | Test Primitives | 12 | - | - |
| 9 | Information Primitives | 9 | - | - |
| 10 | Structure Engines | - | 5 | 8 |
| 11 | Physics Engines | - | 4 | 7 |
| 12 | Dynamics Engines | - | 4 | 8 |
| 13 | Advanced Engines | - | 4 | 10 |
| **14** | **Statistics Engines** | - | **4** | **12** |
| | **TOTAL** | **116** | **21** | **52** |

**Now you have baselines, anomaly scoring, fleet rankings, and executive summaries.**
