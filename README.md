# A Data-Driven Look at VC Fundraising

# Introduction

This project explores the venture capital (VC) fundraising landscape using a dataset of over **4,500 individual funds**. The data includes detailed information on each fund's characteristics—such as its age, investment stage, country focus, associated firm size, and more.

### The Central Question:
> **Can we predict how much capital a VC fund will raise based on its fund structure, firm characteristics, and strategic focus?**

Understanding which factors drive fundraising success is crucial for investors, founders, and policy-makers alike. Venture capital shapes the flow of innovation and access to opportunity—yet the fundraising process often seems opaque. This project aims to shed light on the patterns behind successful capital raises, using data-driven modeling to reveal which characteristics actually matter.

This analysis is particularly relevant to **limited partners (LPs)**—such as institutional investors, endowments, and high-net-worth individuals—who allocate capital across multiple venture firms. By examining how fund-level characteristics relate to fundraising outcomes, LPs can better assess a firm's track record and compare it against industry-wide patterns. This model may support more informed decision-making when evaluating which VC firms demonstrate consistent fundraising success.


## Dataset Overview

- **Total number of observations (rows):** 4820 (after data cleaning)
- **Source:** Compiled from FactSet, a financial data and software company  
- **Time span:** Fund launches from the 2008-2025 
- **Focus:** Fund-level data, not startup-level data


## Relevant Columns for Prediction

| Column Name              | Description |
|--------------------------|-------------|
| `Fund Amount Raised`     | **Target variable** — total capital raised by the fund (in millions) |
| `AUM (Current)`          | Assets under management for the firm managing the fund |
| `Fund Age`               | Age of the fund (in years) |
| `Firm # of Funds`        | Number of total funds the firm has launched |
| `Average Fund Size (MM)` | Current average fund size at the firm |
| `Fund Type`              | Investment stage focus (e.g., Early Stage, Buyout, Mixed) — multi-label |
| `Fund Country Focus`     | Geographic target of investments (e.g., US, Europe) |
| `Fund Status`            | Whether the fund is currently raising, inevesting or closed |
| `Fund Industry Focus`    | Industry sectors the fund invests in (e.g., healthcare, tech) — multi-label |

These columns were selected based on their potential predictive value and their availability **prior to fundraising**, to avoid data leakage.


# Data Cleaning and Exploratory Data Analysis

## Data Cleaning Steps

The raw dataset initially contained data on over **6,000 venture capital funds**, which required substantial cleaning and transformation to ensure consistency and usability. Key steps included handling missing values, standardizing existing features, and creating derived variables from timestamps.

- The original dataset included a `Fund Open Date` column, which was used to create a new variable: `Fund Age`. This feature calculates the number of years a fund has been active, allowing for a more interpretable and relevant metric when evaluating a fund's track record.

- The dataset also contained a column called `Fund Invests in Multiple Rounds`, referring to cases where a VC fund reinvests in startups it has already funded. This can be a strategic move to protect equity or support growth. However, this column had over **5,200 missing values out of 6,200**, making it too incomplete to be useful. As a result, it was dropped from the final dataset.

- I conducted initial visual analysis to understand the distribution of numerical variables. This revealed that several columns, including `AUM (Current)`, were **highly skewed** and contained significant outliers. Below are plots showing the boxplot and distribution of `AUM (Current)` before cleaning:

<iframe
 src="aum-raw-distribution.html"
 width="800"
 height="600"
 frameborder="0"
></iframe>

<iframe
 src="aum-raw-boxplot.html"
 width="800"
 height="600"
 frameborder="0"
></iframe>

- To address these extreme outliers, I applied **IQR-based filtering** instead of the more common 3-standard-deviation rule. Because the AUM distribution was **right-skewed**, the mean and standard deviation were distorted by a few extremely large values. The IQR method uses the interquartile range to identify and filter out outliers more robustly. This resulted in a distribution that better reflects the **central tendency** of most funds.


<iframe
 src="aum-clean-distribution.html"
 width="800"
 height="600"
 frameborder="0"
></iframe>

<iframe
 src="aum-clean-boxplot.html"
 width="800"
 height="600"
 frameborder="0"
></iframe>

- After these cleaning steps, the total number of observations was reduced to **4,820**, resulting in a cleaner and more reliable dataset for modeling and analysis.


### Preview of Cleaned Data

| Fund Name            | Fund Status | Fund Open Date | Fund Type   | Fund Country Focus | Fund Industry Focus                         | AUM (Current) | Firm # of Funds | Average Fund Size (MM) | Fund Amount Sought | Fund Amount Raised | Fund Age |
|----------------------|-------------|----------------|-------------|--------------------|----------------------------------------------|----------------|------------------|--------------------------|---------------------|---------------------|----------|
| 01 Advisors 01 Fund  | Divesting   | 2019-05-03     | Early Stage | United States      | Other                                        | 855.00         | 3.0              | 285.00                   | 200.00              | 135.00              | 6        |
| 01 Advisors 02 LP    | Investing   | 2021-01-20     | Later Stage | United States      | Internet Software/Services                   | 855.00         | 3.0              | 285.00                   | 325.00              | 325.00              | 4        |
| 01 Advisors 03 LP    | Investing   | 2022-03-24     | Early Stage | United States      | Packaged Software; Internet Software/Services| 855.00         | 3.0              | 285.00                   | 325.00              | 395.00              | 3        |
| 01fintech LP         | Investing   | 2022-01-01     | Buyout      | United States      | Other                                        | 61.90          | 1.0              | 61.90                    | 300.00              | 61.90               | 3        |
| 01vc Fund II LP      | Investing   | 2019-01-01     | Early Stage | China              | Other                                        | 14.57          | 3.0              | 14.57                    | 14.57               | 14.57               | 6        |



## Insights from Exploratory Data Analysis

Visual analysis of the dataset provided several key insights into the structure and characteristics of venture capital funds.

### 🌍 Fund Country Focus

Most funds in the dataset are based in the **United States** (2,377), followed by **India** with 213 funds. This trend aligns with expectations, as the U.S. has long been the epicenter of venture capital activity—particularly in regions like Silicon Valley and San Francisco.

<iframe
 src="fund-country-vc.html"
 width="800"
 height="600"
 frameborder="0"
></iframe>

### 🚀 Fund Type (Stage Focus)

The majority of funds in the dataset are focused on **Early Stage** investments. Some funds invest across multiple stages (e.g., Seed + Early + Late), but **Late Stage-only funds are relatively rare**. This reflects real-world dynamics: as investment rounds progress, the required check sizes grow substantially—often into the hundreds of millions or billions—making later-stage investing accessible to fewer firms.

<iframe
 src="fund-type-vc.html"
 width="800"
 height="600"
 frameborder="0"
></iframe>

### 📈 AUM vs. Number of Funds (Colored by Fund Age)

This scatter plot explores the relationship between the number of funds managed by a firm and its current assets under management (AUM). Each point represents a fund and is color-coded by its age.

<iframe
 src="funds_v_aum.html"
 width="800"
 height="600"
 frameborder="0"
></iframe>

There is a **slightly positive trend**: as the number of funds managed increases, AUM tends to rise as well. However, the relationship is **weak and noisy**, with wide variation in AUM even among firms managing the same number of funds.

Color gradients suggest that **fund age is not a strong determinant of AUM**. Older funds appear throughout the AUM spectrum, indicating that factors like investment strategy, fund type, or firm reputation may be more influential than time alone.


### Grouped Table: Fund Type vs. Average AUM

This grouped table highlights the **average assets under management (AUM)** for funds operating at different investment stages. Funds that span **multiple stages**—especially those combining early stage, later stage, and buyout strategies—tend to manage significantly higher capital.

This suggests that **broad-stage or hybrid investment approaches** are associated with larger fund sizes. Rather than specializing solely in early or late stage, many of the highest-AUM fund types include a blend of strategies, which may signal greater flexibility, experience, or appeal to institutional investors.

| Fund Type                                           | Average AUM (in Millions) |
|----------------------------------------------------|---------------------------:|
| Early Stage; Fund of Funds; Secondary; Buyout      | 3,167.71                   |
| Seed Stage; Early Stage; Later Stage; Secondary... | 2,102.77                   |
| Seed Stage; Early Stage; Later Stage; Fund of Funds| 1,787.27                   |
| Early Stage; Later Stage; LBO; MBO; Buyout         | 1,777.40                   |
| Early Stage; Real Estate                           | 1,764.08                   |
| ...                                                | ...                        |
| Early Stage; Mezzanine; Debt                       | 28.42                      |
| Early Stage; Secondary                             | 18.03                      |
| Seed Stage; Early Stage; Fund of Funds             | 11.46                      |
| Mezzanine; LBO; Real Estate                        | 11.43                      |
| Seed Stage; Early Stage; Infrastructure/Proj Fin   | 4.81                       |

This breakdown supports the broader finding that **larger funds often diversify across multiple stages**, likely due to the increased capital demands and longer investment horizons involved in managing a more flexible portfolio.


### Imputation

For this project, I made selective decisions about how to handle missing data, balancing data quality with time constraints and model interpretability.

A notable variable, `Fund Industry Focus`, was **excluded from modeling** due to several challenges:
- It had **multi-label values** (e.g., "Healthcare; Fintech; Consumer"), requiring multi-hot encoding
- It showed **extremely high cardinality** with hundreds of unique industry combinations
- Fewer than **1,000 out of 4,800 rows** had non-missing values, making imputation unreliable
- Creating a catch-all `'Other'` category resulted in too many rows falling into this group
- Feature engineering for this column would have required extensive parsing, cleaning, and transformation that wasn't feasible within the project's time constraints

As a result, I opted to **drop `Fund Industry Focus` entirely** from the modeling dataset, instead of doing imputation. This decision was supported by the fact that most of the dataset (~80%) was missing industry data anyway, meaning the exclusion would not significantly affect model performance.

# Framing a Prediction Problem

### Prediction Type:  
This is a **regression problem**, where the goal is to predict a **continuous numerical value**: the total capital raised by a VC fund.

### Response Variable:
The response variable is `Fund Amount Raised`. This was chosen because it serves as a direct measure of a fund’s **performance and credibility** from the perspective of **limited partners (LPs)**, such as institutional investors and endowments. 

Although the dataset included a column called `Fund Amount Sought`, we chose **not to use it** for prediction. From the perspective of an LP making a funding decision, the amount a fund *wants* to raise is often **not publicly known** at the time of evaluation. It is fund-specific and can be aspirational, rather than predictive of actual outcomes. For this reason, it was excluded to avoid leakage and to align with what would be known at the "time of prediction."

### Why This Question Matters:
This prediction model can help LPs assess **which fund characteristics are linked to stronger fundraising outcomes**, potentially guiding their decisions when committing capital. It also contributes to a broader understanding of what types of VC funds tend to succeed in raising capital.


### Evaluation Metric: R²

We use **R² (coefficient of determination)** as the primary evaluation metric for our regression model. R² measures the proportion of variance in the target variable (`Fund Amount Raised`) that is explained by the features in the model.

This metric was chosen over others because:
- It is **intuitive to interpret**: an R² closer to 1 indicates a better fit.
- It allows for **direct comparison between baseline and final models**.
- It captures overall model performance without being affected by the scale of the prediction errors, which makes it more interpretable than raw error-based metrics like MSE or MAE for our purpose.


### Time-of-Prediction Justification:
All predictor variables used in the model are features that would be **available to a limited partner evaluating a fund** at the time it is raising capital. These include fund-level descriptors like:
- Fund age
- AUM (Current)
- Fund Type (investment stage focus)
- Fund Country Focus
- Fund Status
- Firm’s track record (e.g. number of funds)
- Average Fund Size

We **intentionally excluded** any features, like 'Fund Amount Sought' that are outcomes of the fundraising process or would only be known after the fact.


# Baseline Model

## Model Information:
The baseline model is a **Lasso regression** model trained on an initial set of features, without any non-linear transformations or advanced feature engineering. It was designed to serve as a simple benchmark, which I iterated upon in the final model.

## Features Used:


#### Quantitative:
- `AUM (Current)`
- `Firm # of Funds`
- `Average Fund Size (MM)`
- `Fund Age`

These were treated as continuous numerical variables and scaled using `StandardScaler`.


#### Nominal Categorical: 
- `Fund Status`
- `Fund Country Focus`

These categorical variables were one-hot encoded using `OneHotEncoder(handle_unknown='ignore')`.  

The preprocessing pipeline used a `ColumnTransformer` to scale numeric features and encode categorical ones:

```python
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Fund Status', 'Fund Country Focus', 'Fund Type']),
    ('num', StandardScaler(), numeric_cols2)
])
```
These steps were combined into a scikit-learn pipeline with a Lasso regressor (alpha=0.1):
```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1))
])
```



### Evaluation Metric:
The model was evaluated using R², which measures the proportion of variance in `Fund Amount Raised` explained by the model. R² is intuitive, allows for direct comparison across models, and is standard for regression problems.

- Train R² = 0.6991
- Test R² = 0.7022

The R² values for the baseline model are quite close on both the training and test sets, which suggests that the model is not overfitting and has reasonably good generalization performance. However, this setup still uses relatively basic features and encodings.

### Summary:

The baseline model establishes a strong starting point using interpretable fund-level features. It avoids data leakage by only using information available at the time of prediction. Although it does not yet capture complex relationships in the data, it sets a fair benchmark for evaluating the added value of the final model's more advanced transformations.

To improve upon this baseline, the final model will:
- Use polynomial features to capture non-linear relationships
- Incorporate a multi-label encoder for Fund Type
- Apply hyperparameter tuning using GridSearchCV to optimize alpha and feature transformations
These steps aim to increase the model’s predictive power while maintaining interpretability


# Final Model

## Additional Feature Engineering

In the final model, I added new features that better capture **non-linear relationships** and more meaningfully represent fund characteristics. These features were added because they reflect how fundraising works in the real world—for example, a fund’s age and its AUM are likely to influence how much capital it can raise, and not necessarily in a straight-line way. I included these based on common-sense patterns in venture capital.

### New Features Added:

- **Polynomial features of `AUM (Current)` and `Fund Age`**  
  These two variables are central to the prediction task. `AUM` reflects the firm’s financial strength, while `Fund Age` captures credibility and track record. The relationship between these variables and fundraising success is likely **nonlinear**—for instance, a small increase in AUM might make a big difference for younger firms, but not for older ones. Polynomial transformations help model these interactions.

- **Multi-label encoded `Fund Type`**  
  Instead of treating `Fund Type` as a single-label category, the final model uses a **multi-hot encoding**, using `MultiLabelBinarizer()`, to capture multiple stage strategies (e.g., Seed + Later Stage). This better reflects real-world behavior, as many funds invest across different stages. 

  - **Reuse of key features from the baseline model**  
  Several informative features from the baseline model were retained in the final version, including:
  - `AUM (Current)`
  - `Fund Age`
  - `Firm # of Funds`
  - `Average Fund Size (MM)`
  - `Fund Country Focus`
  - `Fund Status`
  
  These were preserved because they provide core information about a fund’s financial size, track record, and geographic strategy. Numerical features were scaled using `StandardScaler`, while categorical variables (`Fund Country Focus`, `Fund Status`) were one-hot encoded to maintain interpretability and allow the model to flexibly represent regional or status-based differences.


## Model and Hyperparameter Tuning

The final model uses **Lasso Regression**, which reduces multicollinearity as well as unnecessary complexity. It performs **automatic feature selection** by shrinking irrelevant coefficients to zero. This improves interpretability and reduces overfitting.

### Hyperparameters
- `alpha`: Controls the strength of regularization.  
- `degree`: The degree of the polynomial features applied to `AUM` and `Fund Age`.

I used `GridSearchCV` with 5-fold cross-validation on the training set to identify the best model configuration. My parameter search spanned:

```python
param_grid = {
    'preprocessing__poly_feats__poly__degree': [1, 2, 3, 4, 5],
    'model__alpha': [0.5, 1, 3, 5, 10]
}
```

Best Hyperparameters found from GridSeach:
- `alpha`: 0.5  
- `polynomial degree`: 5


## Preprocessing Summary
All feature transformations were combined into a single pipeline using scikit-learn’s ColumnTransformer. Polynomial features were generated for AUM (Current) and Fund Age, while all numeric variables were scaled. Categorical features (Fund Status, Fund Country Focus) were one-hot encoded, and Fund Type was multi-hot encoded using a custom transformer to reflect its multi-label structure. This setup ensured clean, consistent preprocessing during training and cross-validation.

## Results

The final model achieved improved performance over the baseline model on both the training and test sets:

| Model         | Train R² | Test R² |
|---------------|----------|---------|
| Baseline      | 0.6991   | 0.7022  |
| **Final Model** | **0.7108**   | **0.7134**  |

While the improvement in R² is modest, it reflects meaningful gains in model fit. The final model captures slightly more variance in `Fund Amount Raised`, likely due to the added nonlinear interactions and richer categorical encodings.

This improvement came without overfitting: the training and test R² values remain closely aligned. The model generalizes well and benefits from regularization through Lasso, which helps reduce the impact of less informative features.

## Conclusion

This project allowed me to explore the fundraising landscape of venture capital (VC) firms through the lens of data. By building a regression model to predict how much capital a VC fund will raise, I gained a deeper understanding of the many fund-level and firm-level characteristics that contribute to fundraising success—such as AUM, fund age, geographic focus, and investment stage strategies. 

Beyond the modeling itself, this project introduced me to real-world venture dynamics and helped me appreciate how complex and multifaceted fund performance can be. It also deepened my understanding of key machine learning techniques, including feature engineering, regularization, and model evaluation, as well as how to think critically about the data generating process when designing predictive models.

### Next Steps

I’d love to explore additional factors that could improve prediction accuracy or offer deeper insights into fund success, such as:
- **Sentiment analysis** of fund press releases, portfolio news, or investor comments
- **More detailed industry focus**, breaking down sectors to see which strategies are over- or underperforming
- **Team composition and diversity**, including gender representation among general partners and investment committees
- **Longitudinal modeling**, tracking performance across fundraising cycles or fund vintages

There’s a lot of potential to build richer, more nuanced models that don’t just predict outcomes but also inform strategy for both VC firms and limited partners.

---

## References

Here are a few academic and industry sources that informed my understanding of VC fundraising and performance factors:

- Kaplan, S. N., & Schoar, A. (2005). [**Private equity performance: Returns, persistence, and capital flows**](https://doi.org/10.1016/j.jfineco.2004.05.003). *Journal of Finance*  
- Gompers, P., Kaplan, S. N., & Mukharlyamov, V. (2016). [**What do private equity firms say they do?**](https://www.nber.org/papers/w21133). *NBER Working Paper*  
- Lerner, J., & Nanda, R. (2020). [**Venture Capital’s Role in Financing Innovation: What We Know and How Much We Still Need to Learn**](https://doi.org/10.3386/w27492). *Harvard Business School/NBER*  
- PitchBook Data. [**VC Fundraising Trends**](https://pitchbook.com/news/reports)  
- NVCA Yearbook. [**National Venture Capital Association Yearbook (latest edition)**](https://nvca.org/research/nvca-yearbook/)




