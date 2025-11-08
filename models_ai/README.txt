How to create the trained model files

=== Logistic Regression Model (Vehicle Classification) ===

1) Create a virtual environment and install dependencies:
   python -m venv .venv
   .venv\\Scripts\\pip install --upgrade pip
   .venv\\Scripts\\pip install scikit-learn numpy pandas joblib

2) Train and save the model:
   .venv\\Scripts\\python models_ai/train_logreg.py

This will generate models_ai/logreg_model.pkl (pickle format).


=== Decision Tree Model (Restaurant Online Order Prediction) ===

1) Make sure you have the zomato.csv dataset file. Place it in one of these locations:
   - Ai_ML_Project/models_ai/zomato.csv
   - Ai_ML_Project/zomato.csv
   - Or in the same directory where you run the script

2) Install dependencies (if not already installed):
   .venv\\Scripts\\pip install scikit-learn numpy pandas joblib

3) Train and save the model:
   .venv\\Scripts\\python models_ai/train_dectree.py

This will generate:
   - models_ai/dectree_model.pkl (Decision Tree model)
   - models_ai/label_encoders.pkl (LabelEncoders for categorical features)
   - models_ai/scaler.pkl (StandardScaler for numerical features)

Note: The script will automatically look for zomato.csv in common locations.
If your dataset is elsewhere, modify the 'possible_paths' list in train_dectree.py

