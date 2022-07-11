# Databricks notebook source
# MAGIC %md
# MAGIC ## Using MLlib
# MAGIC 
# MAGIC The first concept to know is that MLlib is built to optimize iterations. To that end there are 3 important concepts to keep in mind: **Transformers**, **Estimators**, and **Pipelines**. Pipelines are how machine learning is 'automated' with MLlib, but first let's build the ideas of transformers and estimators. 
# MAGIC 
# MAGIC ### Transformers
# MAGIC A transfomer a method that takes the dataframe you are working on in to a new dataframe. 
# MAGIC 
# MAGIC You can call transformers with the `.transform()` method of transformer objects. 
# MAGIC 
# MAGIC ### Example:
# MAGIC * [Tokenizer](https://spark.apache.org/docs/latest/ml-features.html#tokenizer)
# MAGIC * [Normalizer](https://spark.apache.org/docs/latest/ml-features.html#normalizer)
# MAGIC 
# MAGIC *This example is taken from the Tokenizer documentation example*
# MAGIC 
# MAGIC ```
# MAGIC sentenceDataFrame = spark.createDataFrame([
# MAGIC     (0, "Hi I heard about Spark"),
# MAGIC     (1, "I wish Java could use case classes"),
# MAGIC     (2, "Logistic,regression,models,are,neat")
# MAGIC ], ["id", "sentence"])
# MAGIC 
# MAGIC tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
# MAGIC tokenized = tokenizer.transform(sentenceDataFrame)
# MAGIC tokenized.select("sentence", "words")\
# MAGIC     .withColumn("tokens", countTokens(col("words"))).show(truncate=False)
# MAGIC ```
# MAGIC ### Estimators
# MAGIC Estimators are a subsection of transformers. Estimators fit the data's parameters and returns a model of how to shape the data. You do this by calling the `.fit()` method of estimator objects to fit the estimator to the data and then call the `.transform()` method to transform the dataframe into the desired dataframe.
# MAGIC 
# MAGIC The main difference between **Transformers** and **Estimators** is estimators take and learn data parameters. Transformers do not. 
# MAGIC 
# MAGIC ### Examples:
# MAGIC * [OneHotEncoder](https://spark.apache.org/docs/latest/ml-features.html#onehotencoder)
# MAGIC * [StringIndexing](https://spark.apache.org/docs/latest/ml-features.html#stringindexer)
# MAGIC * [MinMaxScaler](https://spark.apache.org/docs/latest/ml-features.html#minmaxscaler)
# MAGIC 
# MAGIC [This website](https://spark.apache.org/docs/latest/ml-features.html) shows a comprehensive list of transformers, estimators, and their implementation that are available to use in MLlib. 
# MAGIC 
# MAGIC *This example is taken from the MinMaxScaler documentation example*
# MAGIC 
# MAGIC ```
# MAGIC dataFrame = spark.createDataFrame([
# MAGIC     (0, Vectors.dense([1.0, 0.1, -1.0]),),
# MAGIC     (1, Vectors.dense([2.0, 1.1, 1.0]),),
# MAGIC     (2, Vectors.dense([3.0, 10.1, 3.0]),)
# MAGIC ], ["id", "features"])
# MAGIC 
# MAGIC scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
# MAGIC 
# MAGIC scalerModel = scaler.fit(dataFrame)
# MAGIC 
# MAGIC scaledData = scalerModel.transform(dataFrame)
# MAGIC ```
# MAGIC 
# MAGIC ### Differences from Sklearn
# MAGIC Unlike Sklearn, MLlib requires that all features are in one column and the targets are in a separate column. Targets are also referred to as labels. 
# MAGIC ```
# MAGIC feat_cols = [avg, feat_engin, col_calcs, etc.]
# MAGIC assembler = VectorAssembler(inputCols = cols,outputCol = 'features')
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examples of Implementation

# COMMAND ----------

# MAGIC %md 
# MAGIC # Demonstrating MLlib

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression Model Building

# COMMAND ----------

#First thing, get the data. The data we will be using in the project will be the 'trees'.
#Running this box should automatically get you that data.

import statsmodels.api as sm
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
import pyspark.sql.functions as F

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

mtcars = sm.datasets.get_rdataset("trees", "datasets", cache=True).data
dat = pd.DataFrame(mtcars)
dat.head(3)


df=spark.createDataFrame(dat)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup DataFrame for ML

# COMMAND ----------

df.columns

# COMMAND ----------

pt = df.select("*").toPandas()
import pandas as pd
import matplotlib.pyplot as plt

plt.plot(pt['Girth'], pt['Volume'], 'o', color='black')
plt.xlabel('Girth', fontsize=14)
plt.ylabel('Volume', fontsize=14)
plt.grid(True)
m, b = np.polyfit(pt['Girth'], pt['Volume'], 1)

#add linear regression line to scatterplot 
plt.plot(pt['Girth'], m*pt['Girth']+b)
plt.show()


# COMMAND ----------

plt.plot(pt['Height'], pt['Volume'], 'o', color='black')
plt.xlabel('Height', fontsize=14)
plt.ylabel('Volume', fontsize=14)
plt.grid(True)
m, b = np.polyfit(pt['Height'], pt['Volume'], 1)

#add linear regression line to scatterplot 
plt.plot(pt['Height'], m*pt['Height']+b)
plt.show()

# COMMAND ----------

# Here, we are going to pick out which columns we want as our features

cols = ['Girth', 'Height']

assembler = VectorAssembler(inputCols = cols,
                           outputCol = 'features')

# COMMAND ----------

# For this package to work, we need to put all of our features into an array.
# So, what this code is doing, is taking all our defined features from our
# data and storing them as an array in 'output'

output = assembler.transform(df)
output.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Preparing the Data

# COMMAND ----------

# This is one way of putting together your "prepared/cleaned" data
final_df = output.select("features","Volume")
#final_df.display()

# COMMAND ----------

# Here you decide how to split your data
train_data, test_data = final_df.randomSplit([0.7, 0.3])

#train_data.display()
#test_data.display()

# COMMAND ----------

# In this chunck and the one below, you can use the describe to find out the size of your data and how the 2 samples compare.
# This dataset is rather small, but I would still say that the mean and std for both are close in value.
train_data.describe().show()

# COMMAND ----------

test_data.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Building the Regression Model

# COMMAND ----------

lm = LinearRegression(labelCol = "Volume")
model = lm.fit(train_data)
pd.DataFrame({"Coefficients":model.coefficients}, index = cols)

# COMMAND ----------

res = model.evaluate(test_data)
res.residuals.show()

# COMMAND ----------

unlabeled_data = test_data.select("features")
predictions = model.transform(unlabeled_data)
predictions.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Metrics

# COMMAND ----------

print("MAE: ", res.meanAbsoluteError)
print("MSE: ", res.meanSquaredError)
print("AMSE: ", res.rootMeanSquaredError)
print("R2: ", res.r2)
print("Adj R2: ", res.r2adj)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification Model Building

# COMMAND ----------

# MAGIC %md 
# MAGIC In this section we will contrast common sklearn classification models and model building techniques to the MLlib equivalent. 
# MAGIC 
# MAGIC 
# MAGIC __Random Forests__:
# MAGIC 
# MAGIC Sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
# MAGIC 
# MAGIC MLlib: https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier
# MAGIC 
# MAGIC __Decision Tree__:
# MAGIC 
# MAGIC Sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# MAGIC 
# MAGIC MLlib: https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ##### Data for Sklearn

# COMMAND ----------

from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

iris = sm.datasets.get_rdataset("iris", "datasets", cache=True).data

features = iris.drop(columns = ["Species"])
target = iris.filter(items = ["Species"])

X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(features, target, test_size=0.2, random_state=314)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ##### Data for MLlib

# COMMAND ----------

iris = sm.datasets.get_rdataset("iris", "datasets", cache=True).data
dat = pd.DataFrame(iris)
iris_mllib = spark.createDataFrame(dat)
iris_mllib = (iris_mllib
              .withColumnRenamed('Sepal.Length', 'Sepal_Length')
              .withColumnRenamed('Sepal.Width', 'Sepal_Width')
              .withColumnRenamed('Petal.Length', 'Petal_Length')
              .withColumnRenamed('Petal.Width', 'Petal_Width')
             )

# COMMAND ----------

# PySpark only works with numeric values. Every piece of data must be some form of numeric, even the target. 
from pyspark.ml.feature import StringIndexer

iris_mllib = (
    StringIndexer(
        inputCol='Species',
        outputCol='Species2',
        handleInvalid='keep')
    .fit(iris_mllib)
    .transform(iris_mllib)
    .drop('Species')
    .withColumnRenamed('Species2', 'Species'))

iris_mllib.show(2)

# COMMAND ----------

# Spark actually works to predict with a column with all the features smashed together into an array for each row of values in the dataset. 
from pyspark.ml.feature import VectorAssembler

# this next like of code make a new column with the feature values in each row and smashes them into an array and calls in 'features' 
assembler = VectorAssembler(
    inputCols=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'], 
    outputCol='features')

# now use .transform() to finish making the dataset
transformed_data = assembler.transform(iris_mllib)
transformed_data.show(2)

# COMMAND ----------

training_data, test_data = transformed_data.randomSplit([0.8,0.2])

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Random Forests

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Sklearn

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
sklearn_rf = RandomForestClassifier()

# COMMAND ----------

sklearn_rf.fit(X_train_sk, y_train_sk)
sklearn_rf_predictions = sklearn_rf.predict(X_test_sk)
result = accuracy_score(y_test_sk, sklearn_rf_predictions)

print("Sklearn Random Forest Accuracy:", round(result, 3))

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### MLlib

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

pyspark_rf = RandomForestClassifier(
    labelCol='Species', # this is how we set the target variable
    featuresCol='features') # this is how we set the feature variables

# COMMAND ----------

pyspark_rf_model = pyspark_rf.fit(training_data) # training the model
predictions = pyspark_rf_model.transform(test_data) # making predictions

# COMMAND ----------

# Evaluate our model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol='Species', 
    predictionCol='prediction', 
    metricName='accuracy')

accuracy = evaluator.evaluate(predictions)
print('MLlib Random Forest Accuracy = ', round(accuracy, 3)) 

# COMMAND ----------

# MAGIC %md 
# MAGIC [This site](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.MulticlassClassificationEvaluator.html#pyspark.ml.evaluation.MulticlassClassificationEvaluator.metricName) has possible metrics for MulticlassClassificationEvaluator. 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Decision Tree

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Sklearn

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
sklearn_dt = DecisionTreeClassifier()

# COMMAND ----------

sklearn_dt.fit(X_train_sk, y_train_sk)
sklearn_dt_predictions = sklearn_dt.predict(X_test_sk)
result = accuracy_score(y_test_sk, sklearn_dt_predictions)

print("Sklearn Decision Tree Accuracy:", round(result, 3))

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### MLlib

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
pyspark_dt = DecisionTreeClassifier(
    labelCol="Species", 
    featuresCol="features")

# COMMAND ----------

pyspark_dt_model = pyspark_dt.fit(training_data) # training the model
predictions = pyspark_dt_model.transform(test_data) # making predictions

# COMMAND ----------

# Evaluate our model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol='Species', 
    predictionCol='prediction', 
    metricName='accuracy')

accuracy = evaluator.evaluate(predictions)
print('MLlib Decision Tree Accuracy: ', round(accuracy, 3)) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clustering Example

# COMMAND ----------

### Import Libraries
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.ml.functions import vector_to_array

### Importing data
#R  data
women = sm.datasets.get_rdataset("women", "datasets", cache=True).data
dat = pd.DataFrame(women)
df1=spark.createDataFrame(dat)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Formatting the data for MLlib Clustering
# MAGIC This is the primary difference between MLlib and Scikit-Learn or some other machine learning libraries because the data has to be formatted very specifically to work with the MLlib clustering models. We need a column with our group that we're trying to predict and then we need one more column that has all of our features in a vector.

# COMMAND ----------

# Specifying our feature columns
feat_col = ['height','weight']
# Shoves all of our features into a single vector column
feature_vector = VectorAssembler(inputCols = feat_col, outputCol = 'features') ### Note for self: maybe explore this functionality before presentation --Hathway will probably ask what it does specifically
#Adding our features vector column to our original dataset
data = feature_vector.transform(df1)
final_data = feature_vector.transform(df1).select('features')

# COMMAND ----------

### Shows what our dataset looks like with the vector feature column before we drop the extra weight column
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Standardizing data 
# MAGIC This is especially important for K-Means with multiple feature columns as the primary goal of the algorithm is to calculate the euclidean distance between the current point and the centroid. If we have variables with significantly different magnitudes such as age vs salary, this has the potential to skew our model. 

# COMMAND ----------

scaler = StandardScaler(inputCol = 'features', outputCol = 'zfeatures', withStd = True, withMean = False)
scalerModel = scaler.fit(final_data)  ### Check more in detail what this is doing
cluster_input_data = scalerModel.transform(final_data)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Training/Evaluating the Model

# COMMAND ----------

### Training the model
kmeans = KMeans(featuresCol = 'zfeatures', k=2) #Where K is the number of clusters
model1 = kmeans.fit(cluster_input_data)

predictions = model1.transform(cluster_input_data)

# COMMAND ----------

### Seeing the centroids chosen for each cluster
centers = model1.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# COMMAND ----------

### Shows our cluster predictions for each row in our data 
cluster_output = model1.transform(cluster_input_data).select('zfeatures', 'prediction')
rows = cluster_output.collect()
df_predictions = spark.createDataFrame(rows)
display(df_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Finding Best Value for K
# MAGIC MLlib predominantly uses the Silhouette method to find the best value for K as the function to calculate Within Sum of Squares to make an elbow plot was deprecated in Spark 3.0.1 <br>
# MAGIC For the Silhouette score, it will return a number between - 1 and 1. The closer to 1 the Silhouette is, the better.

# COMMAND ----------

### Silhouettes
silhouettes = []
for k in range (2,16): #You can test as many k values as you would like however it gets significantly longer when doing so
    kmeans = KMeans(featuresCol = 'zfeatures', k=k)
    model2 = kmeans.fit(cluster_input_data)
    predictions = model2.transform(cluster_input_data)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    silhouettes.append(silhouette)

print(silhouettes)
print(max(silhouettes))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Graphing the Clusters
# MAGIC This seems to be slightly more complicated because you can only transform your predictions onto the data where all of your features are in a vector which cannot be graphed that way. However fortunately for us we can do this by using the vector_to_array function and then explode the array and go from there.

# COMMAND ----------

graph_df = (cluster_output.withColumn("features", vector_to_array("zfeatures"))).select(['features'] + ['prediction'])
graph_df=graph_df.withColumn('height', F.col('features').getItem(0))
graph_df=graph_df.withColumn('weight', F.col('features').getItem(1))   
graph_df = graph_df.select(['prediction'] + ['height'] + ['weight'])
display(graph_df)

# COMMAND ----------

pd_graph = graph_df.select("*").toPandas()
pd_graph['prediction'] = pd_graph['prediction'].astype(str)
#p9.ggplot(data = pd_graph, mapping=p9.aes(x="height",y="weight", color = "prediction"))+p9.geom_point()
sns.scatterplot(data=pd_graph, x="height", y="weight", hue = "prediction",palette="colorblind")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipelines

# COMMAND ----------

# libraries needed to implement pipeline
import statsmodels.api as sm
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# loading in iris data
iris = sm.datasets.get_rdataset("iris", "datasets", cache=True).data
dat = pd.DataFrame(iris)
iris_mllib = spark.createDataFrame(dat)
iris_mllib = (iris_mllib
              .withColumnRenamed('Sepal.Length', 'Sepal_Length')
              .withColumnRenamed('Sepal.Width', 'Sepal_Width')
              .withColumnRenamed('Petal.Length', 'Petal_Length')
              .withColumnRenamed('Petal.Width', 'Petal_Width')
             )
# split the data
training_data_a, test_data_a = iris_mllib.randomSplit([0.8,0.2])
# make sure your target column is called label
strings_indexed = StringIndexer(inputCol='Species',outputCol='label',handleInvalid='keep')
# breaking up what the features and targets are
feat_col = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
assembler = VectorAssembler(inputCols=feat_col, outputCol='features')
# put the model here 
pyspark_rf = RandomForestClassifier(
    labelCol ='label', # this is how we set the target variable
    featuresCol ='features')

# building the pipeline
class_pipe = Pipeline(stages = [strings_indexed, assembler, pyspark_rf])
# an estimator is being used
class_pipe_model = class_pipe.fit(training_data_a)
# 'transforming' predictions onto the test data
pred_pipe_model = class_pipe_model.transform(test_data_a)
# seeing the new predictions on the test data frame
# display(pred_pipe_model)

# if the label column is called something else
accuracy_dict = {'target':'label'}

print('Accuracy') 
print(round(MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pred_pipe_model), 4))
print('F1') 
print(round(MulticlassClassificationEvaluator(metricName="f1").evaluate(pred_pipe_model), 4))
print('Weighted Precision') 
print(round(MulticlassClassificationEvaluator(metricName="weightedPrecision").evaluate(pred_pipe_model),4))
print('Weighted Recall')
print(round(MulticlassClassificationEvaluator(metricName="weightedRecall").evaluate(pred_pipe_model), 4))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Hyperparameter Tuning

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

#establish the parameters to test. 
# Go to the link to look at the different hyperparameters
#https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html#pyspark.ml.classification.RandomForestClassifier.setParams
paramGrid = (ParamGridBuilder()
             .addGrid(pyspark_rf.maxDepth, [1,2, 3])
             .addGrid(pyspark_rf.maxBins, [4,8,16])
#              .addGrid(pyspark_rf.numTrees, [5,10,15,20,25,30])
             .build())

# create how to evaluate the best model
rf_eval = MulticlassClassificationEvaluator(metricName="accuracy")

# create the cross validator
rf_cv = CrossValidator(estimator=class_pipe, estimatorParamMaps=paramGrid, evaluator=rf_eval, numFolds=3, parallelism = 4)

# actually running the cross_validation and fitting the model to the training data
cv_rf_model = rf_cv.fit(training_data_a)

# testing how well the model ran by transforming the predictions onto the test data
cv_rf_pred = cv_rf_model.transform(test_data_a)

#showing the accuracy score
print('Accuracy') 
print(round(MulticlassClassificationEvaluator(metricName="accuracy").evaluate(cv_rf_pred), 4))
print('F1') 
print(round(MulticlassClassificationEvaluator(metricName="f1").evaluate(cv_rf_pred), 4))
print('Weighted Precision') 
print(round(MulticlassClassificationEvaluator(metricName="weightedPrecision").evaluate(cv_rf_pred),4))
print('Weighted Recall')
print(round(MulticlassClassificationEvaluator(metricName="weightedRecall").evaluate(cv_rf_pred), 4))
