## House Rent Prediction using Linear Regression
# Overview
This project predicts the monthly house rent based on features like BHK, Size, Bathroom, City, Furnishing Status, and Area Type using Linear Regression.
The dataset undergoes preprocessing, transformation, and feature encoding to build an interpretable and accurate regression model.

# Dataset
File: https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset?select=House_Rent_Dataset.csv

Streamlit: https://7-10-25-home-rent.streamlit.app/

# Key Features:
- BHK:Number of bedrooms
- Size:Total area (in sq. ft)
- Bathroom:Number of bathrooms
- City:Location of the property
- Furnishing Status:Furnishing condition (Furnished / Semi-Furnished / Unfurnished)
- Area Type:Built area type (Super Area / Carpet Area / Built Area)
- Rent:Monthly rent (Target variable)

# Data Preprocessing
- Power Transformation (Yeo-Johnson) applied on BHK and Size to stabilize variance and make data more Gaussian.
- Log Transformation applied on Rent and Bathroom to reduce skewness.
- Standard Scaling applied to normalize all numerical features.
- One-Hot Encoding used for categorical variables (City, Furnishing Status, Area Type).
- Dataset split into train (80%) and test (20%) using train_test_split.
  
# Model Development
1. Algorithm Used: LinearRegression from scikit-learn
2. Model trained on transformed and encoded features.
3. Coefficients and intercepts were extracted for interpretability.
4. Evaluation metrics used:R² Score,Root Mean Squared Error (RMSE)

# Visualization
Distribution of transformed variables (BHK, Size, Bathroom, Rent)
Histograms with KDE plots using Seaborn

<img width="540" height="540" alt="image" src="https://github.com/user-attachments/assets/9e368fcb-4578-44bb-8f1c-e1bbb5105601" />
<img width="540" height="540" alt="image" src="https://github.com/user-attachments/assets/a2af29cd-de81-4e35-89e7-f2bbaff994ce" />
<img width="540" height="540" alt="image" src="https://github.com/user-attachments/assets/8de15bfb-dea6-44b9-8ebe-164915f8d737" />
<img width="540" height="540" alt="image" src="https://github.com/user-attachments/assets/59c15a25-7507-471d-88e1-18bf3101d7c5" />

# Libraries Used
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

# How to Run
1. Clone the repository
- git clone <repo_name>
- cd <repo_name>
2. Install dependencies
- pip install -r requirements.txt
- pip install streamlit
  
# Stremlit Screenshot
<img width="580" height="580" alt="image" src="https://github.com/user-attachments/assets/e62132ae-dbaf-4600-b697-1524f2c775ea" />

# Conclusion
This project demonstrates the end-to-end process of:
- Data preprocessing
- Statistical transformation
- Linear Regression modeling
- Rent prediction with interpretability
- It highlights how regression techniques can be used effectively in real-estate analytics to forecast rental prices.
