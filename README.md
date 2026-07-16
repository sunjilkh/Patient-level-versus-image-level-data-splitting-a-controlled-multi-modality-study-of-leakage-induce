# The Illusion of Accuracy: Quantifying Augmentation-Induced Data Leakage in Medical Image Classification

This repository contains the official codebase and executed results for the study **"The Illusion of Accuracy: Quantifying Augmentation-Induced Data Leakage in Medical Image Classification."** 

Our study empirically demonstrates how dataset augmentation—when improperly applied *before* train-test partitioning—creates "sibling leakage." This allows near-identical augmented copies of testing images to leak into the training set, artificially inflating performance metrics while destroying the model's true clinical generalization capabilities.

## Datasets Investigated
To ensure robustness against desk-rejection and to prove this is a universal problem across medical AI, we conducted a rigorous 2x2 factorial evaluation across three completely diverse medical imaging domains:

1. **Radiology (X-Ray):** `NIH ChestX-ray14` dataset (Effusion vs. Normal)
2. **Dermatology (Skin):** `ISIC 2020 Melanoma` dataset (Malignant vs. Benign)
3. **Ophthalmology (Retina):** `Kermany 2018 OCT2017` dataset (Disease vs. Normal)

## The Memorization Quotient (MQ)
We introduce the **Memorization Quotient (MQ)**, a novel mathematical metric calculated using cosine similarity in the `ResNet-50` embedding space. 
*   **MQ > 1** mathematically proves that the model relies on low-level, augmentation-invariant patient/camera signatures rather than actual clinical features to make predictions.

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
