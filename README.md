# LPBF CuCrZr — Process and Property Modelling

Runfang Lin · September 2026

Computational work from my Master of Professional Engineering capstone project
at Monash University, rebuilt and re-validated in September 2026. It covers
relative-density prediction for LPBF CuCrZr and multi-objective optimisation of
tensile strength against electrical conductivity, and serves as the starting
point for my PhD proposal on coupling build conditions with ageing response.

Technical notes on the evaluation protocols will be added to this file over the
coming week.

---

## What is here

### Density model

An XGBoost model predicting relative density from laser power, scan speed,
hatch spacing and layer thickness, trained on 81 records compiled from
published LPBF parameter studies. Model and hyperparameter selection use
cross-validation inside the training data; a group-based held-out set is
reserved for final evaluation only.

`Density_Predictor.py` is a small desktop tool that takes the four parameters
and returns a prediction, with no need to touch the training code.

### Property-optimisation workflow

Two independent Gaussian processes predict UTS and electrical conductivity from
combined build and ageing conditions, using 37 paired literature records plus
target-specific records handled separately by each submodel. Both carry
sample-specific observation noise, and their predictive uncertainty is
propagated into the optimisation stage.

Expected Hypervolume Improvement then ranks 106,624 candidate build–ageing
combinations by expected gain to the current UTS–EC front. The highest-ranked
candidate is 350 W, 800 mm/s, aged at 510 °C for 1 h, predicted at
513 ± 23 MPa and 81.8 ± 7.3 %IACS.

These are posterior predictions. The capstone had no access to an LPBF
platform, so no recommended condition was built or measured and nothing was fed
back into the models. The active-learning loop was never closed.

---

## What it can and cannot do

### Density model

Held-out performance is R² = 0.886 and RMSE = 0.61 percentage points. Six
repeated observations at identical parameters give a measurement repeatability
of about 0.31 percentage points, so the model sits at roughly twice the
measurement noise.

It is a screening tool for the high-density region rather than a regression
model across the full density range. In the held-out test, samples measured
between 94% and 97% received nearly identical predictions: those conditions
were withheld as whole parameter groups, and the training data hold little
comparable information there, so the model converged to a near-constant
response.

The gap between 0.61 and 0.31 matters near the 99% threshold. Prediction error
is not small relative to the distance separating a marginal condition from that
boundary, so a deterministic pass/fail screen overstates certainty. This is one
reason the proposed PhD work replaces the hard density filter with a
probabilistic manufacturability constraint.

### Property optimisation

The recommendations are candidate experiments, not validated performance
claims. The top-ranked candidate specifies 350 W, below every record in the
conductivity training data, and its predicted uncertainty is correspondingly
wide. The ranking prioritises experiments under the current model; it is not
evidence that those conditions will deliver the predicted properties. Closing
that loop — building, characterising, updating — is the purpose of the proposed
PhD work.

---

## Re-validation of the original results

Two parts of the original capstone workflow required correction.

### Density-model evaluation

The original capstone workflow reported a held-out R² of approximately 0.89 and 
an RMSE of 0.34 percentage points. **That value is not an independent performance
estimate and should not be used.**

The original workflow compared 93,750 candidate configurations against the
held-out data and kept whichever scored best. The nominal test set had
therefore participated in choosing the model, and the reported statistic
carries the resulting selection bias.

The rebuilt workflow confines model and hyperparameter selection to the
training data; the held-out set is used once, after selection is complete. On
the same 81 records this gives **R² = 0.886 and RMSE = 0.61 percentage
points** — roughly twice the error originally quoted.

Every result reported here comes from the re-validated workflow. The earlier
exploratory scripts generate none of the current outputs and are not included.

### Property-model anchor points

The original capstone run also reported approximately 85 %IACS for the recommended 
region. That run included three manually specified anchor points added to the
Gaussian-process training data, and I could not establish reliable provenance
for them during reconstruction. Rather than carry them as literature
observations without traceable evidence, I removed them.

The highest-ranked candidate changes from

- 499 ± 32 MPa UTS and 84.8 ± 7.3 %IACS

to

- 513 ± 23 MPa UTS and 81.8 ± 7.3 %IACS.

The recommended region is broadly unchanged — comparatively low laser power,
low volumetric energy density, roughly one hour of ageing — but the ageing
temperature shifts from about 530 °C to 510 °C.

---

## A finding that shaped the proposed work

The original capstone used a three-parameter density model that omitted hatch
spacing. Re-testing showed the omission discarded physically meaningful
information.

Hatch spacing spans 0.08 to 0.22 mm across the 81 records. Removing it makes 29
of them indistinguishable in the model's input space despite differing measured
densities. In one case, two records with identical retained inputs but hatch
spacings of 0.12 and 0.22 mm differ by 2.6 percentage points. A three-input
model must assign both the same representation and return their average.

Under the corrected protocol the three-parameter model reaches a mean
outer-fold R² of 0.702 ± 0.096 against 0.790 ± 0.113 for the four-parameter
model, with the feature set chosen before the held-out data were touched. The
rebuilt workflow therefore uses all four variables.

This clarifies two choices in the proposal. It fixes hatch spacing
experimentally after platform qualification rather than assuming it does not
matter. And because prediction uncertainty is substantial relative to the 99%
threshold itself, the proposed work uses a probabilistic manufacturability
constraint instead of a binary filter.

---

## Relationship to the proposed PhD work

The capstone treated LPBF as a direct process-to-property optimisation problem.
The proposal extends it in three ways: it separates build condition from ageing
treatment and asks whether different manufacturable build states respond
differently to matched ageing schedules; it places measured microstructural
states between process parameters and final properties instead of asking a
surrogate to infer all metallurgical behaviour from processing variables; and it
treats manufacturability, property prediction and experiment selection
probabilistically.

The intended chain is build parameters → initial microstructural state → ageing
response → aged state → UTS and conductivity → posterior Pareto frontier → next
experiment. This repository is the computational precursor to that framework,
not a completed experimental study.

---

## Repository structure

| Path | Description |
| --- | --- |
| `Density Model/Density_Predictor.py` | Desktop tool: four parameters in, predicted density out |
| `Density Model/Density_Model.py` | Trains and evaluates the density model |
| `Density Model/Density_Data.csv` | The 81 density records |
| `Density Model/Density_Results/` | Fitted model, metrics, figures |
| `Pareto Model/Data/` | Paired and target-specific literature records |
| `Pareto Model/Preprocessing/generate_test_data.py` | Builds the 106,624-point candidate grid |
| `Pareto Model/Pareto_Model/Pareto_Model.py` | Gaussian-process models and EHVI ranking |
| `Pareto Model/Output/Pareto_Result/` | Ranked candidates, figures, run summary |
| `requirements.txt` | Pinned package versions |

The clearest single output is
`Pareto Model/Output/Pareto_Result/pareto_candidates_real_units.png`, showing
the paired literature observations, the predicted candidate space and the five
highest-ranked conditions.

---

## Running it

Tested on Windows 64-bit with Python 3.14.2. Package versions are pinned in
`requirements.txt`; other recent Python versions may work but were not used to
produce the archived results.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Density model.** Double-click `Density Model/Run_Density_Model.bat` to train,
then `Run_Density_Predictor.bat` to open the predictor.

**Property workflow.** Generate the candidate grid first, then rank it:

```
cd "Pareto Model/Preprocessing"
python generate_test_data.py
cd "../Pareto_Model"
python Pareto_Model.py
```

The candidate grid and the full ranked output are excluded from version control
— together they exceed 18 MB and both are reproduced by the scripts above. The
virtual environment is excluded for the same reason, and because it is
platform-specific. `requirements.txt` pins exact versions, which matters here:
the predictor checks the installed xgboost and scikit-learn versions against
those recorded at training time and refuses to load the model if they differ.
