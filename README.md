# Model Documentation  

To improve the baseline model, we extracted the features with the highest correlation to G3. Then we iterated on the model to find the subset of those attributes with the highest k-fold accuracy.   

## Feature Extraction  

We ran pandas' correlation matrix on all of the student data to explore the relationships between individual attributes and G3. After sorting the resulting Pearson correlation coefficients, we can see that the attributes with very strong correlation to G3 were G2 (0.9) and G1 (0.8). All other attributes had weak associations under a magnitude of 0.4. However, we decided to include failures as a potential attribute in our model in the next phase. Failures had the next highest correlation (-0.36), and this would help us to evaluate whether other attributes with similarly weak correlations were worth revisiting.  

```
df = pd.read_csv('data/student-mat.csv', sep=';')
G3_df = df['G3']
correlation_matrix = df.corrwith(G3_df).sort_values()
print(correlation_matrix.to_string())  
```  

**Results**  

Positive correlation to G3:  
* G2 (90%)  
* G1 (80%)  
* Medu (22%)  
* Fedu (15%)  
* studytime (10%)  
*    ...etc  

Negative correlation to G3:  
* failures (-36%)  
* age (-16%)  
* goout (-13%)  
* traveltime (-12%)  
*    ...etc  


## Model Iteration & Accuracy Testing  

### General Setup
We considered four common types of models: random forest classifier, basic decision tree, logistic regression, and naive bayes classifier. To evaluate the accuracy, of the model, we checked the f1 score, which is a combination of precision and recall commonly used to evaluate trained binary classifiers. We also checked the k-fold cross-validation accuracy of each type of classifier with 10 splits and shuffling on. This means we **randomly split** the data into training and testing sets ten ways and found the average accuracy. This should mitigate sampling biases and provide a clearer accuracy score than just dividing the dataset in half once.

### Baseline Model
The baseline model provided in the starter code yields these f1 scores and kfold accuracies when training:   

**F1 Score**  

+ Random Forest: 0.52
+ Decision Tree: 0.50
+ Logistic Regression: 0.00
+ Naive Bayes: 0.16

**K-fold Cross Validation Accuracy**  

+ Random Forest: 0.10
+ Decision Tree: 0.11
+ Logistic Regression: 0.00
+ Naive Bayes: 0.04

### Iterations
We first used G1 and G2 to predict G3. The scores are shown below. We can see that this is a huge improvement from the baseline for all types of models. The top three models have very similar scores, and logistic regression has the highest accuracy by a tiny margin.    

**F1 Score**  

+ Random Forest: 0.95
+ Decision Tree: 0.95
+ Logistic Regression: 0.94
+ Naive Bayes: 0.88

**K-fold Cross Validation Accuracy**  

+ Random Forest: 0.92
+ Decision Tree: 0.92
+ Logistic Regression: 0.93
+ Naive Bayes: 0.86

Just to be safe, we also checked the models using G1, G2, and failures to predict G3. 

**F1 Score**  

+ Random Forest: 0.97
+ Decision Tree: 0.97
+ Logistic Regression: 0.95
+ Naive Bayes: 0.84

**K-fold Cross Validation Accuracy**  

+ Random Forest: 0.92
+ Decision Tree: 0.93
+ Logistic Regression: 0.94
+ Naive Bayes: 0.83

We can see that logistic regression again has the highest accuracy, but a slightly lower F1 score. Furthermore, adding failures harmed its accuracy. Therefore, we will stick to logistic regression using just G1 and G2.  

## Final Model 
The final model uses logistic regression and the attributes G1 and G2 to predict if G3 is greater than 15 or not. G1 and G2 are the student's first period and second period grades, so it makes sense that these attributes are so highly correlated with the same student's final grade. As discussed above, the f1 score of this model is 0.95, and the k-fold cross validation value was 0.94. These are much higher (almost double) of the scores for the baseline model.  

