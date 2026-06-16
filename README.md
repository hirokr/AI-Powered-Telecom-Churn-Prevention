# AI-Powered Telecom Churn Prevention

An end-to-end machine-learning and business proposal for identifying telecom
customers at risk of churn and prioritizing targeted retention actions.

The project uses 100,000 anonymized customer records from Company A. It combines
exploratory data analysis, business-driven feature engineering, gradient
boosting, customer-risk ranking, and an illustrative retention-impact model.

## Results

The final model blends:

- 65% engineered LightGBM
- 35% native-categorical CatBoost ensemble

Evaluation uses a stratified 60/20/20 train, validation, and held-out test split.

| Metric | Test result |
|---|---:|
| ROC-AUC | **0.7023** |
| PR-AUC | **0.6949** |
| Accuracy at threshold 0.50 | **64.3%** |
| Top-20% churn capture | **30.1%** |
| Top-20% risk lift | **1.51x** |

The proposed PoC scores customers weekly and targets the highest-risk 20% with
actions matched to the dominant risk signal, such as handset upgrades, plan-fit
reviews, or proactive service recovery.

## Deliverables

- [`Company_A_Telecom_Churn_Proposal.ipynb`](Company_A_Telecom_Churn_Proposal.ipynb):
  fully executed, standalone analysis notebook
- [`Company_A_Churn_Proposal.pdf`](Company_A_Churn_Proposal.pdf):
  final 15-slide submission report
- [`Company_A_Churn_Proposal.pptx`](Company_A_Churn_Proposal.pptx):
  editable presentation

## Repository Structure

```text
├── telecom/
│   ├── Client.csv
│   └── Record.csv
├── artifacts/
│   ├── advanced/              # Final model, metrics, and importance
│   └── improved/              # CatBoost ensemble artifacts
├── Company_A_Telecom_Churn_Proposal.ipynb
├── Company_A_Churn_Proposal.pdf
├── Company_A_Churn_Proposal.pptx
├── telecom_features.py
├── churn_ensemble.py
├── advanced_churn_model.py
├── train_models.py
├── train_improved_model.py
├── train_advanced_model.py
├── create_submission_notebook.py
├── create_presentation.py
└── requirements.txt
```

## Methodology

### Data preparation

`Client.csv` and `Record.csv` are validated and merged one-to-one using
`Customer_ID`. The identifier is removed before modeling.

Numeric missing values are median-imputed with missingness indicators.
Categorical values use explicit missing categories and encodings appropriate to
each model.

### Feature engineering

The pipeline creates 34 business-oriented features, including:

- current usage relative to three-month and lifetime averages
- equipment age relative to customer tenure
- call completion, failure, and voice-drop rates
- overage and recurring-revenue shares
- customer-care intensity
- household subscriber utilization

All features are calculated without using target statistics.

### Model development

The project compares linear and nonlinear baselines before training LightGBM and
CatBoost models. Model architecture and ensemble weights are selected using only
the validation partition. The test partition is evaluated once after selection.

### Business proposal

The model is used as a ranking system rather than an automatic customer
decision. The recommended 90-day PoC includes randomized control groups to
measure incremental retention and campaign value.

Financial values in the report are transparent planning scenarios, not observed
campaign outcomes.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Notebook

```bash
jupyter lab Company_A_Telecom_Churn_Proposal.ipynb
```

The committed notebook already contains executed outputs. A full clean run
retrains all ensemble components and can take approximately 10-20 minutes,
depending on hardware.

## Reproduce the Script Pipeline

Train the models in sequence:

```bash
python train_models.py
python train_improved_model.py
python train_advanced_model.py
```

Generate the presentation:

```bash
python create_presentation.py
```

Generate a fresh unexecuted notebook:

```bash
python create_submission_notebook.py
```

## Key Limitations

- The supplied churn target is almost evenly balanced, unlike many production
  telecom portfolios.
- Feature importance describes predictive contribution, not causal impact.
- Retention effectiveness must be established through controlled experiments.
- Production use requires probability calibration, drift monitoring, fairness
  checks, and updated financial assumptions.

## References

- Deloitte, [2026 Telecommunications Industry Outlook](https://www.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions/2026/telecommunications-industry-outlook.html)
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*,
  NeurIPS 2017
- Prokhorenkova et al., *CatBoost: Unbiased Boosting with Categorical Features*,
  NeurIPS 2018

The dataset and assignment materials were supplied for the GCI World 2026 final
assignment.
