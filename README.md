# Split-induced sibling data leakage inflates accuracy in medical image classification

This repository contains the official codebase and executed Jupyter notebooks for the study on **augmentation-induced sibling data leakage** in medical image classification.

Our study empirically demonstrates how random image-level dataset partitioning (the standard practice in many machine learning pipelines) allows near-identical augmented or anatomically adjacent copies of testing images to leak into the training set. This "sibling leakage" artificially inflates performance metrics, creating an illusion of high accuracy while crippling the model's true clinical generalization capabilities.

## Datasets Investigated
To ensure robustness and prove this is a universal problem across medical AI, we conducted a rigorous 2x2 factorial evaluation across three diverse medical imaging domains:

1. **Radiology (X-Ray):** `NIH ChestX-ray14` dataset (6,000 images, multiple images per patient)
2. **Dermatology (Skin):** `ISIC 2020 Melanoma` dataset (1,168 images, lesion siblings)
3. **Ophthalmology (Retina):** `Kermany OCT2017` dataset (1,200 images, adjacent B-scans)

## Four Targeted Analyses
Instead of relying on a single metric, this repository executes four targeted analyses on a shared, patient-disjoint test fold:

1. **Metric Inflation (McNemar's test):** Evaluates a comprehensive 11-metric panel (AUC, F1, Accuracy, etc.) to prove that the rigorous, patient-disjoint arm statistically outperforms the leaky arm only when isolated from leakage.
2. **Natural Class Imbalance:** Demonstrates how data leakage masks the severe impact of natural clinical imbalance (e.g., 2% malignant prevalence), falsely portraying models as robust.
3. **Leakage Severity Escalation:** Establishes categorical sibling thresholds (0%, 25%, 50%, 75%, 100%) to demonstrate how performance linearly inflates relative to the density of leaked siblings.
4. **XAI Audit (Border-Activation Ratio):** Uses Grad-CAM and LIME to mathematically quantify visual attention. We show that leaky models actively memorize augmentation borders and artifacts rather than true clinical pathology.

## Reproducibility and Executed Notebooks 

We provide the fully executed Jupyter Notebooks (`.ipynb`) used in the study. **These notebooks already contain the executed outputs, training logs, accuracy curves, and Grad-CAM/LIME explainability figures**, meaning you can review our results directly here on GitHub without running any code!

The included notebooks are:
*   `NIH_ChestXray_Leakage_Study.ipynb`
*   `ISIC_Melanoma_Leakage_Study.ipynb`
*   `OCT_Retinal_Leakage_Study.ipynb`

### How to Run (Optional)

If you wish to independently verify the findings by running the code yourself, follow these steps:

1. **Upload to Kaggle:**
   *   Create a new Kaggle Notebook.
   *   Go to **File -> Import Notebook** and select the `.ipynb` file for the domain you wish to test.

2. **Attach Datasets:**
   Add the following public Kaggle datasets depending on the notebook you are running:
   *   **NIH:** Add `nih-chest-xrays` (or `chest-xray-nihcc`)
   *   **ISIC:** Add `siim-isic-melanoma-classification`
   *   **OCT:** Add `kermany2018`

3. **Execute:**
   Ensure your Kaggle session is set to **GPU T4 x2** (Dual GPU) to accelerate the 5-fold cross-validation. Run all cells. The notebook will automatically:
   *   Install dependencies (like `grad-cam` and `lime`).
   *   Execute the 5-fold CV under both **Patient-Level Split (Rigorous)** and **Random-Split (Leaky)** protocols.
   *   Perform McNemar's Test for statistical significance and generate XAI figures.

## Requirements
If running locally (not on Kaggle), you will need the Python packages listed in `requirements.txt`. Note that the Kaggle environment generally comes pre-installed with most required packages (such as `torch` and `scikit-learn`), and the notebooks automatically run `!pip install grad-cam lime` internally.
