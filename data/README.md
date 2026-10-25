# Data

The dataset is **not committed** to this repository (it is excluded by `.gitignore`). Place it here as:

```
data/credit_score_train.csv
```

## Expected format

- 100,000 rows: 12,500 customers × 8 monthly records (`Month` 1–8)
- 28 columns, already cleaned (no missing values):

| Column | Type | Used as |
|---|---|---|
| `ID`, `Customer_ID`, `Name`, `SSN` | id / text | dropped. `Customer_ID` is used to group the train/test split |
| `Month`, `Age` | numeric | feature |
| `Occupation`, `Credit_Mix`, `Payment_of_Min_Amount` | categorical | feature (one-hot) |
| `Annual_Income`, `Outstanding_Debt` | numeric | combined into `debt_to_income` |
| `Monthly_Inhand_Salary`, `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Num_of_Loan`, `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Changed_Credit_Limit`, `Num_Credit_Inquiries`, `Credit_Utilization_Ratio`, `Credit_History_Age`, `Total_EMI_per_month`, `Amount_invested_monthly`, `Monthly_Balance` | numeric | feature |
| `Type_of_Loan`, `Payment_Behaviour` | text | dropped |
| `Credit_Score` | `Good` / `Standard` / `Poor` | **target** |

## Source

The data comes from Kaggle. It is based on the **Credit score classification** dataset by Rohan Paris:
https://www.kaggle.com/datasets/parisrohan/credit-score-classification

The original Kaggle files contain messy values (for example `Credit_History_Age` as text and missing entries). This project uses a cleaned 100,000-row version with no missing values.
<!-- If you downloaded a cleaned version from another Kaggle page, replace the link above with that page. -->

To get it:
1. Sign in to Kaggle and download the dataset from the page above.
2. Save the training file as `data/credit_score_train.csv`.

Check the license on the Kaggle page before re-uploading the data anywhere.
