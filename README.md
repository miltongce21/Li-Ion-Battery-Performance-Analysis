# Li-Ion-Battery-Performance-Analysis
# Machine Learning-Assisted Classification and Data-Driven Discovery of High-Performance Lithium-Ion Battery Materials

Stability and performance analysis of lithium-ion battery cathode candidates using exploratory data analysis, a weighted scoring index, and a Random Forest classifier.

## Problem Statement

Can machine learning predict the stability of lithium-ion battery materials from their formation energy, band gap and crystal structure, and how can performance and stability be analysed from an existing dataset?

## Objectives

- Screen candidate materials using thermodynamic and electronic criteria.
- Rank materials with a weighted **performance score**.
- Compare stability and conductivity across crystal systems.
- Study correlations between material properties.
- Train a **Random Forest classifier** to predict phase stability.

## Dataset

The project reads a CSV file (`lithium-ion batteries.csv`) in which each row is a material. Columns used:

## Methodology

### 1. Material screening
Materials are kept only if they satisfy all of:
- Formation energy < 0
- E above hull < 0.05 eV
- 0 < Band gap < 1.5 eV

### 2. Performance score
A weighted index combining stability and electronic properties:

```
performance_score = 0.5 * (-Formation Energy)
                  + 0.3 * (0.05 - E Above Hull)
                  + 0.2 * (2 - Band Gap)
```

The weights (0.5, 0.3, 0.2) are domain-knowledge based assumptions: formation energy is weighted highest because of its direct link to thermodynamic stability, followed by phase stability, then band gap. They can be tuned for a specific research focus or optimised with ML methods. The top 10 materials are reported.

### 3. Exploratory analysis
- Thermodynamically stable materials (formation energy < 0)
- Phase-stable materials (E above hull = 0)
- Highly conductive materials (band gap < 0.5 eV)
- Metallic materials (band gap = 0) and insulators (band gap > 3 eV)
- Mean stability and band gap grouped by crystal system
- Correlation analysis of formation energy, E above hull, band gap, density and volume

### 4. Visualisations
- Scatter plot: formation energy vs band gap, coloured by crystal system
- Bar chart: performance score by material
- Pie chart: crystal system distribution
- Histograms (with KDE): band gap, formation energy, E above hull
- Box plot: band gap by crystal system
- Correlation heatmap
- Random Forest feature importance

### 5. Stability prediction (Random Forest)
- **Target:** `Stability = 1` if E above hull < 0.05 eV, otherwise `0`
- **Features:** Formation Energy, Band Gap, Nsites, Density, Volume
- **Split:** 80% train / 20% test (`random_state=42`)
- **Model:** `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Evaluation:** accuracy and classification report (precision, recall, F1)

`E Above Hull` is deliberately excluded from the features, since it defines the target label.

## Key Findings

- Crystal system has little effect on thermodynamic stability (average formation energies are similar) but a clear effect on band gap.
- Triclinic materials tend to have lower band gaps (better conductivity), while orthorhombic materials have higher band gaps.
- Orthorhombic materials show the lowest average E above hull in this dataset.
- Formation energy and band gap are negatively correlated (about -0.44); formation energy and E above hull are positively correlated (about 0.27).
- Most materials have formation energies between roughly -2.8 and -2.2 eV, and a group of about 30 materials has a near-zero band gap (metal-like).

## Tech Stack

- Python 3
- pandas
- matplotlib
- seaborn
- scikit-learn

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install pandas matplotlib seaborn scikit-learn
```

## Usage

1. Place `lithium-ion batteries.csv` in the project folder.
2. Update the file path in the script to a relative path, for example:
   ```python
   df = pd.read_csv("lithium-ion batteries.csv")
   ```
3. Run the script:
   ```bash
   python main.py
   ```

The script prints the filtered tables and model metrics, and displays all plots.

## Project Structure

```
.
├── main.py                       # Full analysis and ML pipeline
├── lithium-ion batteries.csv     # Dataset
└── README.md
```

## Limitations and Future Work

- The performance score weights are assumed, not learned; they could be optimised or validated against experimental data.
- Stability labels may be imbalanced; consider stratified splits, cross-validation and class-weighting.
- Add hyperparameter tuning and compare with other models (Gradient Boosting, XGBoost, SVM).
- Include chemical composition features (e.g. element fractions) and electrochemical properties such as voltage and capacity.
- Add voltage, capacity and ionic diffusion data for a fuller picture of battery performance.

## Author

Milton — [GitHub](https://github.com/<miltoncme21>)
