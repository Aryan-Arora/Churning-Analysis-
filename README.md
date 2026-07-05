Bank Customer Churn Prediction

Exploratory data analysis and classification modeling on a bank customer churn dataset, using Python's data science and machine learning stack. Built as a hands-on learning project — every step is implemented to understand both the how and the why.

Dataset

Churn_Modelling.csv — Bank Customer Churn dataset (10,000 rows, 14 columns)
Source: Kaggle

Target variable: Exited (1 = customer churned, 0 = customer stayed)



Workflow

Each model follows a consistent pipeline:


Drop identifier columns (RowNumber, CustomerId, Surname)
Separate features (X) and target (y)
Train/test split (80/20)
Encode categorical features — Gender via LabelEncoder, Geography via one-hot encoding
Scale numeric features with StandardScaler (fit on train, transform on test)
Fit model on training data
Predict and evaluate on test data


Models Built

ModelDescriptionLogistic RegressionBaseline linear classifier, predicts churn probabilityDecision TreeRule-based splits, tuned with max_depth to control overfittingRandom ForestEnsemble of decision trees, aggregated via majority voteSVMFinds the maximum-margin boundary between churn classesKNNClassifies based on the majority class of nearest neighborsGradient BoostingSequential ensemble that corrects prior errors

Key Learnings


The accuracy trap: with ~80% of customers not churning, a naive "always predict no churn" model scores 80% accuracy — raw accuracy alone is misleading on imbalanced data.
Fit/transform discipline: encoders and scalers are fit only on training data to prevent data leakage.
Overfitting: an unrestricted Decision Tree scored lower on test data than a depth-limited one, despite fitting the training data more closely.
Model comparison: different models can disagree on the same prediction — comparing them surfaces trade-offs between interpretability, accuracy, and generalization.


Tools

Python 3.13 · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · PyCharm

Author

Aryan Arora
