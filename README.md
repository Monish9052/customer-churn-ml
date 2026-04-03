# Customer Churn Prediction

## About this project
This project is about predicting whether a customer will leave (churn) or not.  
I built this project to understand how machine learning works in a real-world scenario.

---

## What I did
- Loaded and explored the dataset  
- Cleaned the data (removed unwanted columns, handled missing values)  
- Converted text data into numerical format  
- Trained multiple machine learning models  
- Compared their performance  
- Selected the best model  
- Built a simple prediction system  

---

## Models used
- Logistic Regression  
- Random Forest  
- XGBoost  

Random Forest performed the best with around 94% accuracy.

---

## Project files
- Data_preprocessing.py → data cleaning  
- train_model.py → model training  
- predict.py → prediction script  
- model.pkl → saved model  
- requirements.txt → required libraries  

---

## How to run
1. Install libraries:
pip install -r requirements.txt

2. Train the model:
python train_model.py

3. Run prediction:
python predict.py

---

## Output
- 0 → Customer will not churn  
- 1 → Customer will churn  

---

## What I learned
- Data preprocessing  
- Machine learning model training  
- Model comparison  
- Using XGBoost  
- Building an end-to-end ML project  

---

## Conclusion
This project helped me understand the complete machine learning workflow from data preprocessing to prediction.
