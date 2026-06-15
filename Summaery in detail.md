1) General ML terms you will see
Parameters
These are the settings used to run the model.

Example: batch_size, epochs, learning_rate, test_size, n_estimators, max_depth
They do not measure performance directly.
They control how the model learns and how much data it sees.
Metrics
These are the numbers that tell you how well the model performed.

Example: accuracy, loss, AUC, R², RMSE, MSE
They are the results you compare across runs.
2) CEW experiment terms
Parameters used
img_size = 80

The input image is resized to 80 × 80 pixels.
Larger images carry more detail but cost more memory and training time.
The number 80 means the model sees a moderately sized image.
batch_size = 16

How many samples are processed together before updating the model weights.
Smaller batches make training more stable but slower.
16 is a standard moderate setting.
epochs = 20

How many times the model passes through the whole training set.
More epochs can improve learning, but too many may cause overfitting.
optimizer = adam

The algorithm that updates the model weights during training.
Adam is a popular optimizer because it adapts learning speed efficiently.
lr = 0.0001

Learning rate.
This controls how big each weight update is.
0.0001 is a small learning rate, which usually makes training more stable.
Evaluation terms used
test_accuracy = 0.8061

This means the model correctly classified about 80.61% of the test samples.
In simple terms: out of 100 new examples, it would get about 81 right.
A higher number is better.
test_loss = 0.4096

Loss measures how far the model’s predictions are from the correct answers.
Lower loss is better.
0.4096 is a moderate value, so the model is making some mistakes but is not completely off.
Interpretation
The CEW model is working and is now a usable baseline.
Around 80% accuracy is acceptable for a first version.
The model is not perfect, but it is learning the pattern reasonably well.
3) Blink experiment terms
Parameters used
best_model = Logistic Regression

The model that gave the best result among the tested options.
Logistic Regression is simple, fast, and often works well for binary classification.
models_compared = 3

The script tried 3 different classifiers.
This tells you the model was not chosen randomly; several candidates were compared.
rf_estimators = 200

Random Forest uses 200 decision trees.
More trees can improve stability, but also increase training time.
test_size = 0.3

30% of the data was reserved for testing.
The remaining 70% was used for training.
Evaluation terms used
best_accuracy = 0.8079

The best model got about 80.79% accuracy.
This means it correctly predicted about 81 out of 100 cases.
It is a solid baseline score.
best_auc = 0.6974

AUC stands for Area Under the ROC Curve.
It measures how well the model distinguishes between the two classes.
A value of 1.0 means perfect separation.
A value of 0.5 means random guessing.
0.6974 is better than random, but not excellent.
So the model has some discrimination power, but there is room for improvement.
best_threshold = 0.5031

This is the cutoff probability used to decide whether a sample is classified as positive or negative.
If the model outputs a probability above 0.5031, it predicts one class.
Below that, it predicts the other.
This threshold was selected to give the best trade-off for this run.
Interpretation
The blink classifier is decent but not outstanding.
It can detect blink-related patterns, but there is still margin for improvement.
AUC of 0.697 means the model is only moderately reliable.
4) NTHU experiment terms
Parameters used
rf_estimators = 200

Random Forest uses 200 trees.
More trees usually make predictions more stable.
test_size = 0.25

25% of the data was used for testing.
75% was used to train the model.
xgb_estimators = 300

XGBoost uses 300 boosting rounds.
Boosting means the model builds trees one after another to correct previous mistakes.
xgb_lr = 0.05

This is XGBoost’s learning rate.
A smaller value generally makes training more careful.
0.05 is a moderate setting.
Evaluation terms used
rf_accuracy = 0.8215

Random Forest accuracy = 82.15%.
This means the Random Forest model got roughly 82 out of 100 examples correct.
xgb_accuracy = 0.8272

XGBoost accuracy = 82.72%.
This is slightly better than the Random Forest result.
It means XGBoost identified the patterns a little more effectively.
Interpretation
Both models perform well.
XGBoost is slightly better than Random Forest in this run.
These are stronger results than the blink and CEW experiments.
5) Attention / MPIIGAZE experiment terms
Parameters used
n_estimators = 300

The model uses 300 decision trees in the boosting process.
More trees usually improve fit but can increase risk of overfitting.
max_depth = 6

Maximum depth of each tree.
Depth controls how complex the tree can become.
A depth of 6 is fairly moderate.
lr = 0.1

Learning rate for the boosting model.
This is fairly high, so the model updates more aggressively.
test_size = 0.2

20% of the data was reserved for testing.
80% was used for training.
Evaluation terms used
mse = 0.0

Mean Squared Error.
This measures the average squared difference between predicted and actual values.
A value of 0.0 means the model’s predictions are almost exactly matching the truth on this run.
Lower is better.
rmse = 0.0022

Root Mean Squared Error.
This is the square root of MSE.
It gives error in the same unit as the target variable.
0.0022 is extremely small, which means the model is making very tiny prediction errors.
r2_score = 0.9996

R² tells how much of the variance in the target is explained by the model.
1.0 means perfect explanation.
0.9996 means the model explains 99.96% of the variation.
This is an excellent score.
Interpretation
This is the strongest result among all four experiments.
The model appears to fit the data almost perfectly.
Because the value is so high, it is worth checking whether the test set is too easy or if there is some form of data leakage or overfitting.
6) Summary of the numbers
Accuracy
Used in CEW, blink, and NTHU.
It tells how many correct predictions the model made.
Example:
0.80 = 80%
0.82 = 82%
Higher is better
Loss
Used in CEW.
It measures prediction error.
Lower is better.
AUC
Used in blink.
Measures how well the model separates classes.
Higher is better.
0.5 = random
1.0 = perfect
MSE / RMSE
Used in attention.
Measure regression error.
Lower is better.
R²
Used in attention.
Measures how much of the outcome the model explains.
Closer to 1.0 is better.
7) Simple takeaway
CEW:

Accuracy around 80.6%
Good baseline, stable model
Blink:

Accuracy around 80.8%
AUC around 0.697
Useful but not strong enough yet
NTHU:

Accuracy around 82%+
Better than CEW and blink
Attention:

R² = 0.9996
Very strong result, but should be validated carefully for overfitting