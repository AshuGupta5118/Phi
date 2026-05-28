# Phi: Advanced Architectural & Facial Golden Ratio Analyzer

Phi is a high-precision computer vision engine built to analyze geometric structure and anatomical harmony against the Golden Ratio ($\Phi \approx 1.6180339887$). 

Unlike surface-level analysis tools that get fooled by generic shapes or accidental outer bounding box symmetry (the "potato face" paradox), Phi relies on intricate internal landmark tracking and a high-order statistical ensemble model. It aggressively penalizes structural variance to evaluate true, consistent geometric harmony.

---

## 🚀 Key Features

* **Dual Processing Pipelines:** Seamlessly toggles between high-precision facial landmark mapping and localized structural/contour chaos checking.
* **Anatomical Landmarking:** Utilizes MediaPipe to track 468 precise spatial points, evaluating classical vertical facial triads and horizontal pentads.
* **Resilient Structural Fallback:** Analyzes image canvas framing, outer subject contours, and point distribution symmetry when no human face is detected.
* **Advanced Mathematical Scoring:** Discards naive arithmetic averages in favor of a robust ensemble formula incorporating Geometric Means, Root Mean Square Error (RMSE), and variance penalties.
* **Visual Dashboard Engine:** Generates high-fidelity visual analysis maps overlaid with dynamic Golden Spirals and real-time structural telemetry.

---

## 📐 The Mathematics Behind Phi

A simple arithmetic mean can easily mask flaws. If an object has a couple of features that accidentally match the golden ratio by coincidence, an average pulls the final score up. Phi prevents this by analyzing data structure across three distinct mathematical axes:

### 1. Root Mean Square Error (RMSE) Deviation
Instead of checking how close a ratio is to $\Phi$, the algorithm measures the strict distance (error) of each individual ratio from $\Phi$, squares those errors to heavily penalize large deviations, averages them, and takes the square root:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (R_i - \Phi)^2}$$

The squaring mechanism aggressively punishes single disproportionate features, dragging down the final score of flawed geometry.

### 2. Geometric Mean (The Harmony Metric)
The geometric mean measures how well all dimensions scale together in fluid proportion. 

$$\text{Geometric Mean} = \left( \prod_{i=1}^{n} C_i \right)^{\frac{1}{n}} = \sqrt[n]{C_1 \times C_2 \times \dots \times C_n}$$

Because the inputs are multiplied, **if any single internal feature scores close to 0%, the entire product collapses toward zero.** This guarantees that objects missing key internal features (like a potato) cannot cheat the algorithm.

### 3. Standard Deviation Variance Penalty ($\sigma$)
True structural harmony requires consistency across all metrics. Phi calculates the standard deviation ($\sigma$) of all feature closeness scores to measure data instability. The final ensemble score is calculated as:

$$\text{Final Score} = \left(0.5 \times \text{Geom Mean}\right) + \left(0.5 \times \text{RMSE Closeness}\right) - \left(0.25 \times \sigma\right)$$

Wildly inconsistent ratios trigger a steep variance penalty, suppressing the final score.

---

## 🧠 Algorithmic Workflow


```text
[ Input Image ] ──> [ MediaPipe Face Mesh Detection ]
│
├──> (Face Found) ──> [ Facial Anatomy Mode ]
│                       • Vertical Triad Segments
│                       • Horizontal Pentad / Ocular Span
│                       • Intercanthal Space Allocation
│
└──> (No Face)    ──> [ Structural Fallback Mode ]
                        • Canvas Aspect Ratio Bounding
                        • Main Subject Contour W/H
                        • Edge Point Distribution Symmetry
│
▼
[ Ensemble Statistical Suite ]
• Calculate RMSE & Geometric Mean
• Extract Standard Deviation (σ)
• Apply Variance Penalties
│
▼
[ Output Dashboard & Analytics ]
```

---

## 🛠️ Installation & Setup

### Prerequisites
Make sure you have Python 3.8+ installed along with the required computer vision and mathematical libraries.

```bash
pip install opencv-python mediapipe numpy
```

### Repository Structure

```text
├── golden_ratio_calculator.py  # Main execution engine
├── test_img1.png               # Structural analysis test image 1 (Canvas/Architecture)
├── test_img2.png               # Facial anatomy test image 2 (Standard Profile)
├── test_img3.png               # Abstract/Chaos test image 3 (Asymmetry Stress Test)
└── README.md                   # Project documentation
```

### Running the Analyzer

To execute the script against the default test imaging suite, run:

```bash
python golden_ratio_calculator.py
```

The engine will log an advanced anatomical or structural report directly to your console and output a rendered high-precision PNG dashboard asset (e.g., `test_img1_golden_anatomy_analysis.png`) displaying the structural vector meshes.

---

## 📊 Evaluation Metrics

Final composite scores are systematically graded across strict anatomical thresholds:

* **$\ge$ 88.00%** | 🌟 Exceptional, true geometric facial harmony.
* **75.00% - 87.99%** | ✨ Strong structural alignment with the Golden Ratio.
* **50.00% - 74.99%** | 🔶 Moderate or localized facial symmetry.
* **< 50.00%** | 🥔 Low overall structural harmony (Potato Threshold).

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.