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
We considered three common types of models: random forest classifier, basic decision tree, and logistic regression. To evaluate the accuracy, of the model, we checked the overall f1 score, which is a combination of precision and recall commonly used to evaluate trained binary classifiers. We also checked the k-fold accuracy of each type of classifier with 10 splits and shuffling on. This means we **randomly split** the data into training and testing sets ten ways and found the average accuracy. This should mitigate sampling biases and provide a clearer accuracy score than just dividing the dataset in half.

### Baseline Model
The baseline model yields an f1 score of 0.55. The accuracies for each of the three model types were between 0.76 and 0.82. 

### Iterations
We first used G1 and G2 to predict G3. This yielded an f1 score of 0.96, average random-forest accuracy of 0.97, average decision-tree accuracy of 0.98, and average logistic-regression accuracy of 0.98. These are much higher than the baseline, without much room for improvement.  

Just to be safe, we also checked the models using G1, G2, and failures to predict G3. This yielded an f1 score of 0.97, average random-forest accuracy of 0.97, average decision-tree accuracy of 0.97, and average logistic-regression accuracy of 0.97. Since adding failures barely affected the model, we decided to just use G1 and G2 in the final model.  

## Final Model 
The final model uses G1 and G2 to predict if G3 is greater than 15 or not. G1 and G2 are the student's first period and second period grades, so it makes sense that these attributes are so highly correlated with the same student's final grade. As described above, the f1 score of this model is 0.96, which is much higher than the original f1 score of 0.55.

