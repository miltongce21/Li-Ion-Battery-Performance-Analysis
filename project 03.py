# Machine Learning-Assisted classification and Data-Driven Discovery of High-Performance Lithium-Ion Battery Materials : Stability and Performance Analysis.
# Problem statement: Can ML predict the stability of Lithium-ion batteries using their formation energy, band gap and Crystal structure and How to analyze the performance and stability from a existing dataset?
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Pycharm\\lithium-ion batteries.csv")
print("Original dataframe:")
print(df.head())
print(df.isnull().sum())
# Filtering the materials with respect to different properties.
# Formation energy should be negative, E above hull should be near zero, Band gap should be minimal for better performance.
required_data=df[(df["Formation Energy (eV)"]<0)&(df["E Above Hull (eV)"]<0.05)&(df["Band Gap (eV)"]>0)&(df["Band Gap (eV)"]<1.5)].copy()
print("Dataframe according to applied conditions:")
print(required_data.head())
required_data["performance_score"] = (
    (-required_data["Formation Energy (eV)"] * 0.5) + # Formation energy has the highest influence in stability.
    ((0.05 - required_data["E Above Hull (eV)"]) * 0.3) + # E above hull has moderate influence
    ((2 - required_data["Band Gap (eV)"]) * 0.2)  # Band gap has the lowest impact.
)
Best_materials=required_data.sort_values("performance_score", ascending=False).head(10)
print("Best performance materials:")
print(Best_materials[["Formula","Formation Energy (eV)","E Above Hull (eV)","Band Gap (eV)","performance_score"]])
# performance score is a weighted index combining stability and electronic properties.
# weights (0.5, 0.3, 0.2) are applied based on domain knowledge and signify relative importance of each feature.
# formation energy is given the highest importance due to its direct relationship to thermodynamic stability.
# E above hull indicates phase stability and band gap signify electronic conductivity.
# These weights are assumed approximately and can be adjusted based on research focus or optimized using ML methods.
## Thermodynamic stability analysis.
Thermodynamically_stabled_materials=df[df["Formation Energy (eV)"]<0]
stable_materials=Thermodynamically_stabled_materials.sort_values(by='Formation Energy (eV)',ascending=False).head(5)
print("Thermodynamically stabled materials:")
print(stable_materials)
# More negative formation energy values indicate more thermodynamically stable materials.Li2Co2SiO6 is thermodynamically more stable.
## Phase stability according to E Above Hull.
stable_phase_materials=df[df["E Above Hull (eV)"]==0]
stable_materials=stable_phase_materials.sort_values(by="E Above Hull (eV)",ascending=False).head(5)
print("Phase stable materials:")
print(stable_materials)
# LiFe(SiO3)2 has more phase stability.
## Materials with better electronic conductivity.
Better_Electronic_conductive=df[df["Band Gap (eV)"]<0.5]
conductive_materials=Better_Electronic_conductive.sort_values(by="Band Gap (eV)",ascending=False).head(5)
print("Materials with better electronic conductivity:")
print(conductive_materials)
# Lower band gap signifies better electronic conductive.LiMnSiO4 is the most conductive materials.
## Materials with metal like conductivity
Metallic_conductivity=df[df["Band Gap (eV)"]==0] # For Metallic materials band gap is zero.
print("materials with Metallic conductivity:")
print(Metallic_conductivity)
## Electrical insulator materials
Electrical_insulator=df[df["Band Gap (eV)"]>3] # Band gap is greater than 3 eV for insulator materials.
print("Electrical_insulator materials:")
print(Electrical_insulator)
## Stability according to crystal system analysis.
crystal_system=df.groupby("Crystal System")[["Formation Energy (eV)","E Above Hull (eV)"]].mean()
print("Stability with respect to crystal_system :")
print(crystal_system)
# Orthorhombic crystal system has lower E above Hull (eV). So It has higher phase stability.
# All crystal system has nearly same avg formation energy. SO, thermodynamic stability of all crystal systems nearly same.
## Conductivity vs Stability analysis.
Conductivity_vs_stability=df.groupby("Crystal System")[["Band Gap (eV)", "Formation Energy (eV)"]].mean()
print("Conductivity vs Stability analysis:")
print(Conductivity_vs_stability)
# Crystal system has less impact on stability.It has significant impact on conductivity.
# Triclinic structure has lower band gap as well as better conductivity.This structure is also thermodynamically more stable slightly.
# orthorhombic structure has higher band gap as well as lower conductivity.
## Analyzing properties correlation.
properties_corr=df[["Formation Energy (eV)", "E Above Hull (eV)", "Band Gap (eV)"]].corr()
print("Analyzing properties correlation:")
print(properties_corr)
# shows relation between the properties.
# Negative correlation between formation energy and band gap (-0.438429 ) means As formation energy increases, band gap decreases.
# positive correlation between formation energy and E above Hull (0.272422 ) means As formation energy increases, E above Hull increases.

## Data visualization
## Stability vs performance
sns.scatterplot(x="Formation Energy (eV)",y="Band Gap (eV)",hue="Crystal System",data=df)
plt.title("stability vs performance of the materials")
plt.xlabel("Formation Energy (eV)")
plt.ylabel("Band Gap (eV)")
plt.tight_layout()
plt.show()
# materials at Top-left side are more stable(less negative formation energy) but less conductive (larger band gap). Useful for solar cells and sensors.
# almost no dots is present on the far right side that indicates higher unstability of the materials to survive.
# Most of the monoclinic(Blue) and orthorhombic(orange) materials have lower conductivity.
# Green dots(triclinic) at the bottom are metal-like conductor due to nearly zero band gap. useful for electrical wires and electrode.
# Most of the materials are present between -2.8 to -2.2 range, having intermediate stability and can be tuned into desire materials.
## Best performance materials.
plt.figure(figsize=(8,5))
plt.bar(required_data["Formula"],required_data["performance_score"],color="teal")
plt.title("Performance score vs Materials")
plt.xlabel("Materials")
plt.xticks(rotation=60)
plt.ylabel("Performance score")
plt.tight_layout()
plt.show()
#  LiMn(SiO3)2 is the best performing material.
# Li3CoSiO5 is the worst performing material.
## Crystal system distribution of materials to find dominant crystal structure.
Crystal_structure_distribution=df["Crystal System"].value_counts()
plt.figure(figsize=(5,5))
plt.pie(Crystal_structure_distribution.values, labels=Crystal_structure_distribution.index , autopct='%1.1f%%',colors=['teal','coral','green'])
plt.title("Crystal System Distribution")
plt.show()
# Most of the materials have monoclinic crystal structure and few materials are triclinic.
# #Band Gap (eV) Distribution for conductivity analysis
plt.figure(figsize=(8,5))
sns.histplot(x="Band Gap (eV)", bins=20, kde=True, color="teal", data=df)
plt.title("Band Gap (eV) Distribution for conductivity analysis")
plt.xlabel("Band Gap (eV)")
plt.ylabel("Number of Materials")
plt.show()
# A group of nearly 30 materials has Nearly Zero band gap and show metal-like conductivity. Mainly used as conductors.
# The materials between the range 0.5 to 2.0 are the semiconductor materials. Used for electronics and solar cells.
# A dominant group (band gap > 2.50) consist of materials those are mainly the wideband gap semiconductors or Insulators.
## thermodynamic stability analysis
plt.figure(figsize=(8,5))
sns.histplot(x="Formation Energy (eV)", bins=20, kde=True, color="tan", data=df)
plt.title("Formation Energy (eV) Distribution for thermodynamic stability analysis")
plt.xlabel("Formation Energy (eV)")
plt.ylabel("Number of Materials")
plt.show()
# Materials on the far left (<-2.7 eV) in the histogram has the highest formation energy, they are thermodynamically more stable.
# Materials present on the far right ( >-2.3 eV) has lower thermodynamic stability, also less concentration of materials.
# The highest concentration of materials is denoted by the dominant peak. (at -2.6 eV)
## Phase stability analysis.
plt.figure(figsize=(8,5))
sns.histplot(x="E Above Hull (eV)", bins=20, kde=True, color="coral", data=df)
plt.title("E Above Hull (eV) Distribution for phase stability analysis")
plt.xlabel("E Above Hull (eV)")
plt.ylabel("Number of Materials")
plt.show()
# The materials with zero E above Hull in the far left has the highest phase stability, over 20 materials are present in this region.
# Phase stability drops beyond 1 eV and materials tends to decompose, concentration of materials is very few in that region.
# Majority of materials present below 1 eV and indicates metastable phases.
# The dominant peak at 0.08 eV shows that the highest no. of materials (Nearly 50) present there.
## Bandgap distribution according to crystal structure (boxplot).
plt.figure(figsize=(8,5))
sns.boxplot(x="Crystal System", y="Band Gap (eV)",color="coral", data=df)
plt.title("Band Gap distribution according to Crystal System")
plt.show()
# Triclinic materials have a 25% of data below 0.5 band gap, indicating this structure has higher concentration conductive or semiconductive materials.
# Lower crystal symmetry(Triclinic) can cause lower median bandgap value and higher symmetry(Orthorhombic) can have higher median bandgap value.
# Orthorhombic materials have the tightest middle 50% of data, indicating higher concentration of materials present in a short range.(1.5-3.0)
## correlation between material's features .
## shows strong and weak relations between features.
materials_features=["Formation Energy (eV)","E Above Hull (eV)","Band Gap (eV)","Density (gm/cc)","Volume"]
corr_matrix=df[materials_features].corr()
plt.figure(figsize=(8,5))
sns.heatmap(corr_matrix, annot=True)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# Formation energy vs density correlation has the strongest positive correlation(0.49), Indicates materials with higher formation energy tends to have higher density also.
# E above hull vs bandgap correlation has the strongest negative correlation(-0.47), Indicates materials with higher E above hull tends to have lower bandgap.
# wide bandgap materials are thermodynamically more stable for strongest correlation.
# Volume has less impact on other properties due to weak positive and negative correlation.
# df.to_excel("Project 01.xlsx",index=False) # saving the file.

## Thermodynamic stability prediction using Random Forest Classifier.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
X=df[["Formation Energy (eV)","Band Gap (eV)", "Nsites","Density (gm/cc)","Volume"]]
df["Stability"] = (df["E Above Hull (eV)"] < 0.05).astype(int)
y= df["Stability"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
random_forest= RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_predict=random_forest.predict(X_test)
# Visualization
plt.figure(figsize=(8,5))
plt.bar(X.columns, random_forest.feature_importances_, color="teal")
plt.xlabel("Feature Importance")
plt.xticks(rotation=80)
plt.tight_layout()
plt.show()

# Accuracy Measurement
print("Accuracy:", accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))

