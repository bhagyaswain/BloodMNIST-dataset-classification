from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os

import sys
outputFilePath = "Q3_output1.txt"
sys.stdout = open(outputFilePath, "w")

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Q3_1") \
    .master("local[4]") \
    .config("spark.local.dir", os.environ.get('TMPDIR', '/tmp')) \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Load your dataset
df = spark.read.csv("/users/acp22abj/com6012/acp22abj-COM6012/Data/HIGGS.csv", header=False, inferSchema=True)

# Define header names
header_names = ["label"] + ["feature{}".format(i) for i in range(1, 29)]

# Rename columns
for i, col_name in enumerate(header_names):
    df = df.withColumnRenamed("_c{}".format(i), col_name)

# Split the data into training and testing sets
seed = 24280
train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)

# Sample 1% of the data with class balancing from the training set
train_df_sampled = train_df.sampleBy("label", fractions={0: 0.01, 1: 0.01}, seed=seed)

# Select features and target column for the sampled training data
feature_cols = train_df_sampled.columns[1:]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df_assembled = assembler.transform(train_df_sampled).select("features", "label")

# Define models
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
gbt = GBTClassifier(labelCol="label", featuresCol="features")
mlp = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", layers=[len(feature_cols), 10, 2])

# Define parameter grids for each model
rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [30, 50]) \
    .addGrid(rf.maxDepth, [3, 7]) \
    .addGrid(rf.minInstancesPerNode, [1, 3]) \
    .build()

gbt_param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [100, 150]) \
    .addGrid(gbt.maxDepth, [4, 6]) \
    .addGrid(gbt.stepSize, [0.05, 0.1]) \
    .build()

mlp_param_grid = ParamGridBuilder() \
    .addGrid(mlp.layers, [[len(feature_cols), 40, 2], [len(feature_cols), 60, 2]]) \
    .addGrid(mlp.blockSize, [64, 128]) \
    .addGrid(mlp.stepSize, [0.03, 0.05]) \
    .build()

# Define evaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label", metricName="areaUnderROC")

# Perform cross-validation to find the best configuration of parameters for each model
cv_rf = CrossValidator(estimator=rf,
                       estimatorParamMaps=rf_param_grid,
                       evaluator=evaluator,
                       numFolds=5,
                       seed=seed)

cv_gbt = CrossValidator(estimator=gbt,
                        estimatorParamMaps=gbt_param_grid,
                        evaluator=evaluator,
                        numFolds=5,
                        seed=seed)

cv_mlp = CrossValidator(estimator=mlp,
                        estimatorParamMaps=mlp_param_grid,
                        evaluator=evaluator,
                        numFolds=5,
                        seed=seed)

# Fit models
cv_model_rf = cv_rf.fit(train_df_assembled)
cv_model_gbt = cv_gbt.fit(train_df_assembled)
cv_model_mlp = cv_mlp.fit(train_df_assembled)

# Retrieve best parameters for each model
best_params_rf = cv_model_rf.bestModel.extractParamMap()
best_params_gbt = cv_model_gbt.bestModel.extractParamMap()
best_params_mlp = cv_model_mlp.bestModel.extractParamMap()

# Print the best parameters for each model
print("Random Forest Classifier:")
print("numTrees:", best_params_rf[rf.numTrees])
print("maxDepth:", best_params_rf[rf.maxDepth])
print("minInstancesPerNode:", best_params_rf[rf.minInstancesPerNode])

print("Gradient-Boosted Tree Classifier:")
print("maxIter:", best_params_gbt[gbt.maxIter])
print("maxDepth:", best_params_gbt[gbt.maxDepth])
print("stepSize:", best_params_gbt[gbt.stepSize])

print("Multilayer Perceptron Classifier:")
print("layers:", best_params_mlp[mlp.layers])
print("blockSize:", best_params_mlp[mlp.blockSize])
print("stepSize:", best_params_mlp[mlp.stepSize])

# Stop SparkSession
spark.stop()

sys.stdout.close()