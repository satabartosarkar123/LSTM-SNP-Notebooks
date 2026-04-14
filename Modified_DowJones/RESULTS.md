# Results: Dow Jones Industrial Index Dataset
**30 Independent Runs | 100 Epochs | 8 Hidden Units | Adam (clipnorm=1.0)**

---

## Table: Modified Fuzzy LSTM-SNP — All 9 Architecture Variants

| # | Model Variant | RMSE Mean | RMSE Std | MSE Mean | MSE Std | NMSE Mean | NMSE Std | Best RMSE | Best MSE | Best NMSE |
|---|--------------|-----------|----------|----------|---------|-----------|----------|-----------|----------|-----------|
| 1 | MF-Gates-Only + DoG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | MF-Gates-Only + SG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | MF-Gates-Only + GB | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | MF-Gates + FuzzyInput + DoG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 5 | MF-Gates + FuzzyInput + SG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 6 | MF-Gates + FuzzyInput + GB | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 7 | MF-Gates + FuzzyOutput + DoG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | MF-Gates + FuzzyOutput + SG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 9 | MF-Gates + FuzzyOutput + GB | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## Gate Output Statistics (C-Gate & O-Gate across 30 runs)

| # | Model Variant | C-Gate Mean | C-Gate Std | C-Gate Min | C-Gate Max | O-Gate Mean | O-Gate Std | O-Gate Min | O-Gate Max |
|---|--------------|-------------|------------|------------|------------|-------------|------------|------------|------------|
| 1 | MF-Gates-Only + DoG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | MF-Gates-Only + SG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | MF-Gates-Only + GB | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | MF-Gates + FuzzyInput + DoG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 5 | MF-Gates + FuzzyInput + SG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 6 | MF-Gates + FuzzyInput + GB | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 7 | MF-Gates + FuzzyOutput + DoG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | MF-Gates + FuzzyOutput + SG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 9 | MF-Gates + FuzzyOutput + GB | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## Architecture Reference

| # | Architecture | Input Dim | Gate Function | Output Layer | Trainable Fuzzy Params |
|---|-------------|-----------|---------------|--------------|------------------------|
| 1 | MF-Gates-Only + DoG | 1 | DoG (biphasic) | Dense(1) | None |
| 2 | MF-Gates-Only + SG | 1 | SG (antisymmetric) | Dense(1) | None |
| 3 | MF-Gates-Only + GB | 1 | GB (non-neg. bell) | Dense(1) | None |
| 4 | MF-Gates + FuzzyInput + DoG | 2 | DoG | Dense(1) | None |
| 5 | MF-Gates + FuzzyInput + SG | 2 | SG | Dense(1) | None |
| 6 | MF-Gates + FuzzyInput + GB | 2 | GB | Dense(1) | None |
| 7 | MF-Gates + FuzzyOutput + DoG | 1 | DoG | FuzzyOutputLayer | 12 |
| 8 | MF-Gates + FuzzyOutput + SG | 1 | SG | FuzzyOutputLayer | 12 |
| 9 | MF-Gates + FuzzyOutput + GB | 1 | GB | FuzzyOutputLayer | 12 |

---
*Fill in table values after running each notebook (30 runs each).*
