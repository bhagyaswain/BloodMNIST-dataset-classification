from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os

import sys
outputFilePath = "Q3_output2.txt"
sys.stdout = open(outputFilePath, "w")

# Load the SparkSession
spark = SparkSession.builder \
    .appName("Q3") \
    .master("local[4]") \
    .config("spark.local.dir", os.environ.get('TMPDIR', '/tmp')) \
    .getOrCreate()

# Load your dataset
df = spark.read.csv("/users/acp22abj/com6012/acp22abj-COM6012/Data/HIGGS.csv", header=False, inferSchema=True)

# Define header names
header_names = ["label"] + ["feature{}".format(i) for i in range(1, 29)]

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Rename columns
for i, col_name in enumerate(header_names):
    df = df.withColumnRenamed("_c{}".format(i), col_name)

seed = 24280
# Split the data into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)

# Define the vector assembler
feature_cols = df.columns[1:]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Assemble features and label for training data
train_df_assembled = assembler.transform(train_df).select("features", "label")

# Assemble features and label for testing data
test_df_assembled = assembler.transform(test_df).select("features", "label")

train_df_assembled.cache()
test_df_assembled.cache()

# Define models with best parameters from step 1
rf_best = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=30, maxDepth=7, minInstancesPerNode=1)
gbt_best = GBTClassifier(labelCol="label", featuresCol="features", maxIter=150, maxDepth=6, stepSize=0.1)
mlp_best = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", layers=[28, 40, 2], blockSize=64, stepSize=0.03)

# Fit models on the training set
rf_model_full = rf_best.fit(train_df_assembled)
gbt_model_full = gbt_best.fit(train_df_assembled)
mlp_model_full = mlp_best.fit(train_df_assembled)

# Evaluate the models on the test set
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label", metricName="areaUnderROC")
predictions_rf_full = rf_model_full.transform(test_df_assembled)
predictions_gbt_full = gbt_model_full.transform(test_df_assembled)
predictions_mlp_full = mlp_model_full.transform(test_df_assembled)

auc_rf_full = evaluator.evaluate(predictions_rf_full)
auc_gbt_full = evaluator.evaluate(predictions_gbt_full)
auc_mlp_full = evaluator.evaluate(predictions_mlp_full)

# Print AUC for each model
print("Random Forest - AUC on full dataset:", auc_rf_full)
print("Gradient Boosting - AUC on full dataset:", auc_gbt_full)
print("Multilayer Perceptron - AUC on full dataset:", auc_mlp_full)

# Stop SparkSession (not necessary in this context as it's managed by the cluster)
spark.stop()
sys.stdout.close()
