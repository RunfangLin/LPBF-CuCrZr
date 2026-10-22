# LPBF CuCrZr — Process and Property Modelling

Runfang Lin · September 2026

Computational work from my Master of Professional Engineering capstone project
at Monash University, rebuilt and re-validated in September 2026. It covers
relative-density prediction for LPBF CuCrZr and multi-objective optimisation of
tensile strength against electrical conductivity, and serves as the starting
point for my PhD proposal on coupling build conditions with ageing response.

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
candidate is 350 W, 800 mm/s, aged at 525 °C for 1 h, predicted at
493 ± 25 MPa and 83.1 ± 7.4 %IACS.

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

The original capstone workflow reported a held-out R² of approximately 0.89