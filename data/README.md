# Data documentation

Source: I-Cheng Yeh (2009), *Default of Credit Card Clients*,
[UCI dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
[doi:10.24432/C55S3H](https://doi.org/10.24432/C55S3H), licensed CC BY 4.0.
The workbook contains 30,000 Taiwanese credit-card records; read it with
`pd.read_excel(path, header=1, engine="xlrd")`.

| Field | Meaning / treatment |
| --- | --- |
| `ID` | Original observation identifier; preserved, never used as a model predictor. |
| `default` | Renamed from `default payment next month`; 1 = default, 0 = no default. |
| `SEX` | Source coding: 1 = male, 2 = female; protected attribute retained for later auditing, excluded from baseline predictors. |
| `LIMIT_BAL` | Credit limit, NT dollars. |
| `AGE` | Age in years. |
| `EDUCATION` | 1 graduate school, 2 university, 3 high school, 4 other; existing cleaning maps observed undocumented codes 0, 5, 6 to 4. |
| `MARRIAGE` | 1 married, 2 single, 3 other; existing cleaning maps observed undocumented code 0 to 3. |
| `PAY_0`, `PAY_2` … `PAY_6` | Repayment status, September through April 2005. Source defines -1 as paid duly and 1–9 as months of delay (9 = nine or more). Observed -2 and 0 lack definitions on the cited UCI page; preserve as separate categories without inventing meanings. |
| `BILL_AMT1` … `BILL_AMT6` | Bill statements, September through April 2005, NT dollars. |
| `PAY_AMT1` … `PAY_AMT6` | Previous payments, September through April 2005, NT dollars. |

No rows are removed. The unchanged raw file is the source of ID alignment.
Preparation preserves source row order and performs the existing category
consolidations. EDA age bands are descriptive and are not model predictors.

`split_metadata.csv` has one row per original ID and stores SEX, default,
original_split, duplicate_group_id, split, and partition_order. The group ID is
the lowest original ID among records with identical cleaned model predictors;
it is metadata, not a feature. `partition_order` is zero-based within the final
partition. `train_metadata.csv` and `test_metadata.csv` are sorted in model input
order. Prediction CSVs join to original observations by ID.

The 35 duplicate rows beyond first occurrences (70 records) refer to matching
all non-ID columns **including the outcome**. Predictor-only matching is broader:
84 groups / 177 records when excluding ID, SEX, and target. Conflicting outcomes
are retained. These matches may be legitimate distinct observations; the source
does not establish that they are erroneous copies. Group isolation is therefore
used to prevent exact predictor overlap without deleting records. See the
generated duplicate audit and stabilization report for crossing counts.
