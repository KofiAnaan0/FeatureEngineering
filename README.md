# Feature Engineering & Data Transformation

This section covers the **Feature Engineering** and **Data Transformation** steps applied in the project to prepare the dataset for analysis and model training.

## Overview

Feature engineering and data transformation are crucial steps in building a robust machine learning model. These processes help improve the performance of the model by creating new features, modifying existing ones, and ensuring the dataset is in the correct format for training.

Key steps in this notebook include:.
- Creating new features based on domain knowledge.
- Encoding categorical variables.
- Normalizing/Standardizing numerical features
- Transforming skewed distributions.

## Key Transformations
1. **Feature Creation:**
   - **Difference Between Off-Peak Prices in December and Preceding January**
     This feature was created to determine price changes across the entire year for each company. It helps to capture significant fluctuations in energy prices between December and the        preceding January, revealing potential seasonal patterns.
     
     ```python
     # Group off-peak prices by companies and month
      monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}).reset_index()
      
      # Get january and december prices
      jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
      dec_prices = monthly_price_by_id.groupby('id').last().reset_index()
      
      # Calculate the difference
      diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}), jan_prices.drop(columns='price_date'), on='id')
      diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
      diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
      diff = diff[['id', 'offpeak_diff_dec_january_energy','offpeak_diff_dec_january_power']]
      diff.head()
     ```

     ![offpeak](assets/)

     ```python
     # Merging the engineered feature
     df = pd.merge(df, diff, on='id')
     df.head()
     ```

     ![offpeak](assets/)

     - **Average Price Changes Across Different Periods**

       This feature provides more granularity by looking at the mean price differences between off-peak, peak, and mid-peak periods. It may reveal patterns across shorter timeframes             (monthly or seasonal fluctuations) that the December-January difference feature might miss.

       ```python
       # Aggregate average prices per period by company
         mean_prices = price_df.groupby(['id']).agg({
             'price_off_peak_var': 'mean', 
             'price_peak_var': 'mean', 
             'price_mid_peak_var': 'mean',
             'price_off_peak_fix': 'mean',
             'price_peak_fix': 'mean',
             'price_mid_peak_fix': 'mean'    
         }).reset_index()
       # Calculate the mean difference between consecutive periods
         mean_prices['off_peak_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices['price_peak_var']
         mean_prices['peak_mid_peak_var_mean_diff'] = mean_prices['price_peak_var'] - mean_prices['price_mid_peak_var']
         mean_prices['off_peak_mid_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices['price_mid_peak_var']
         mean_prices['off_peak_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices['price_peak_fix']
         mean_prices['peak_mid_peak_fix_mean_diff'] = mean_prices['price_peak_fix'] - mean_prices['price_mid_peak_fix']
         mean_prices['off_peak_mid_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices['price_mid_peak_fix']

       columns = [
          'id', 
          'off_peak_peak_var_mean_diff',
          'peak_mid_peak_var_mean_diff', 
          'off_peak_mid_peak_var_mean_diff',
          'off_peak_peak_fix_mean_diff', 
          'peak_mid_peak_fix_mean_diff', 
          'off_peak_mid_peak_fix_mean_diff'
      ]
      df = pd.merge(df, mean_prices[columns], on='id')
      df.head()

     ![offpeak](assets/)

     - **Tenure**

       How long a company has been a client of PowerCo.

       ```python
       df['tenure'] = ((df['date_end'] - df['date_activ'])/ np.timedelta64(1, 'D')/ 365.25).astype(int)
       df.groupby(['tenure']).agg({'churn': 'mean'}).sort_values(by='churn', ascending=False).head(3)
       ```

       ![offpeak](assets/)

       We can see that companies who have only been a client for 4 or less months are much more likely to churn compared to companies that have been a client for longer. Interestingly,         the difference between 4 and 5 months is about 4%, which represents a large jump in likelihood for a customer to churn compared to the other differences between ordered tenure           values. Perhaps this reveals that getting a customer to over 4 months tenure is actually a large milestone with respect to keeping them as a long term customer.

       This is an interesting feature to keep for modelling because clearly how long you've been a client, has a influence on the chance of a client churning.
   
2. **Encoding Categorical Variables:**
   - One-hot encoding and label encoding for categorical data.
   
3. **Feature Scaling:**
   - Standardization (Z-score scaling) and normalization (min-max scaling).
   
4. **Feature Creation:**
   - Creating interaction terms, polynomial features, or domain-specific transformations.

## Notebook

The detailed code and explanations for the feature engineering and data transformation steps can be found in the Jupyter Notebook:

[![Feature Engineering & Data Transformation Notebook](https://img.shields.io/badge/Notebook-Feature_Engineering-blue)](./link_to_your_notebook.ipynb)

You can also view the notebook directly in Google Colab using the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your_repo_link/your_notebook.ipynb)

## Libraries Used
- `pandas` for data manipulation.
- `scikit-learn` for transformations like encoding and scaling.
- `numpy` for numerical computations.

## Next Steps
Once the dataset is transformed, it is ready for the next step: **Model Training & Evaluation**. Check out the [model training notebook](./link_to_model_training_notebook.ipynb) for more details.

