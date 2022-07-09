import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from plotnine import *
import statsmodels.api as sm
import numpy as np
import pandas as pd
import plotnine as p9
import pyspark.sql.functions as F
from pyspark.sql import SparkSession as ss
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


mtcars = sm.datasets.get_rdataset("trees", "datasets", cache=True).data
dat = pd.DataFrame(mtcars)
dat.head(3)
spark = ss.builder.getOrCreate()

df= spark.createDataFrame(dat)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup DataFrame for ML

# COMMAND ----------

df.columns

# COMMAND ----------

pt = df.select("*").toPandas()
p9.ggplot(data = pt, mapping=p9.aes(x="Girth",y="Volume"))+p9.geom_point() + p9.geom_smooth(method = "lm")


# COMMAND ----------

p9.ggplot(data = pt, mapping=p9.aes(x="Height",y="Volume"))+p9.geom_point() + p9.geom_smooth(method = "lm")

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
#output.display()

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