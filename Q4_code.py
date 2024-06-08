import sys
outputFilePath = "Q4_output.txt"
sys.stdout = open(outputFilePath, "w")

from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
import os
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, explode, split, avg, to_timestamp, from_unixtime


# Initialize SparkSession
spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Q4") \
        .config("spark.local.dir", os.environ.get('TMPDIR', '/tmp')) \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Load ratings data
ratings = spark.read.csv("Data/ratings.csv", header=True, inferSchema=True)
ratings = ratings.withColumn('timestamp', to_timestamp(from_unixtime(col('timestamp'))))

# Load movies data
movies_df = spark.read.csv("Data/movies.csv", header=True, inferSchema=True)


# Sort data by timestamp
ratings = ratings.orderBy("timestamp")

# Define the splits for training data (40%, 60%, 80%)
splits = [0.4, 0.6, 0.8]

# Define your student number (digits only)
student_number = 24280

evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

# Part B2: Movie Analysis within the Largest User Cluster
def movie_analysis(ratings, user_factors, largest_cluster, movies_df):

    # Filter for users in the largest cluster
    cluster_users = user_factors.filter(col("prediction") == largest_cluster).select(col("id").alias("userId"))
    
    # Join with ratings to get movies rated by these users
    cluster_ratings = ratings.join(cluster_users, "userId")

    # Group by movieId, calculate average rating and filter for high-rated movies
    movies_largest_cluster = cluster_ratings.groupBy("movieId").agg(avg("rating").alias("avg_rating"))

    # Filter for movies with an average rating >= 4
    top_movies = movies_largest_cluster.filter(col("avg_rating") >= 4)

    ## Print top 5 movies
    print(f"Top 5 Movies for Split Size {split_size * 100}%:")
    top_movies_list = top_movies.sort("avg_rating", ascending=False).head(5)
    for movie_row in top_movies_list:
        movie_id = movie_row["movieId"]
        movie_title = movies_df.filter(col("movieId") == movie_id).select("title").head()["title"]
        print(movie_title)

    ### Part B2: Movie Analysis within the Largest User Cluster

     #Join with movies DataFrame to get genre information
    movie_genres = top_movies.join(movies_df, top_movies.movieId == movies_df.movieId).select("genres")
    
    # Explode genres into separate rows
    movie_genres = movie_genres.withColumn("genre", explode(split(col("genres"), "[|]")))
    
    # Count occurrences of each genre and get top 10
    popular_genres = movie_genres.groupBy("genre").count().orderBy("count", ascending=False).head(10)
    return popular_genres

rmse_values = []
mse_values = []
mae_values = []
cluster_sizes_per_split = []

# Iterate over each split size
for split_size in splits:
    # Split data into training and testing sets based on timestamp
    split_point = int(split_size * ratings.count())
    training_data = ratings.limit(split_point)
    testing_data = ratings.subtract(training_data)
    
    # Define ALS settings for Setting 1
    als_setting_1 = ALS(userCol="userId", itemCol="movieId", seed=student_number, coldStartStrategy="drop")
    
    # Fit ALS model on training data with Setting 1
    model_1 = als_setting_1.fit(training_data)
    
    # Make predictions on testing data
    predictions_1 = model_1.transform(testing_data)
    
    # Compute metrics for Setting 1
    rmse_1 = evaluator_rmse.evaluate(predictions_1)
    mse_1 = evaluator_mse.evaluate(predictions_1)
    mae_1 = evaluator_mae.evaluate(predictions_1)

    rmse_values.append(rmse_1)
    mse_values.append(mse_1)
    mae_values.append(mae_1)
    
    print(f"Metrics for {split_size * 100}% training data with Setting 1:")
    print(f"RMSE: {rmse_1}, MSE: {mse_1}, MAE: {mae_1}")

    # Define ALS settings for Setting 2 (based on results from Setting 1)
    als_setting_2 = ALS(rank=30,userCol="userId", itemCol="movieId", seed=student_number, coldStartStrategy="drop")
    
    # Fit ALS model on training data with Setting 2
    model_2 = als_setting_2.fit(training_data)
    
    # Make predictions on testing data
    predictions_2 = model_2.transform(testing_data)
    
    # Compute metrics for Setting 2
    rmse_2 = evaluator_rmse.evaluate(predictions_2)
    mse_2 = evaluator_mse.evaluate(predictions_2)
    mae_2 = evaluator_mae.evaluate(predictions_2)

    rmse_values.append(rmse_2)
    mse_values.append(mse_2)
    mae_values.append(mae_2)
    
    print(f"Metrics for {split_size * 100}% training data with Setting 2:")
    print(f"RMSE: {rmse_2}, MSE: {mse_2}, MAE: {mae_2}")

    # Part B1: User Clustering
    kmeans = KMeans(k=25,seed = student_number,featuresCol="features")
    model = kmeans.fit(model_2.userFactors)

    predictions = model.transform(model_2.userFactors)

    # Find top five largest user clusters
    top_clusters = predictions.groupBy("prediction").count().orderBy("count", ascending=False).head(5)

    ## Print Top 5 Cluters for split size
    print(f"Top 5 Clusters for Split Size {split_size * 100}%:")
    for cluster in top_clusters:
        print(f"Cluster {cluster['prediction']}: {cluster['count']} users")

    ## Store size of top 5 clusters
    cluster_sizes = [cluster["count"] for cluster in top_clusters]
    cluster_sizes_per_split.append(cluster_sizes)

    # Extract the largest cluster prediction value (the first cluster from the top 5)
    largest_cluster_prediction =  predictions.groupBy("prediction").count().orderBy("count", ascending=False).first()["prediction"]
    
    top_genres = movie_analysis(ratings,predictions,largest_cluster_prediction,movies_df)
    # Print top genres
    print(f"Top 10 Genres for Split Size {split_size * 100}%:")
    for genre, count in top_genres:
        print(f"{genre}: {count}")


# Visualize metrics
plt.figure(figsize=(10, 6))
plt.plot(rmse_values, label='RMSE', marker='o')
plt.plot(mse_values, label='MSE', marker='o')
plt.plot(mae_values, label='MAE', marker='o')
plt.title('Metrics Comparison')
plt.xlabel('Split Size')
plt.ylabel('Value')
plt.xticks(range(len(splits)*2), [f"Split {i+1}, Setting {j+1}" for i in range(len(splits)) for j in range(2)], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('Q4_figA.png')

# Plot cluster sizes for each split size
plt.figure(figsize=(10, 6))
for i, cluster_sizes in enumerate(cluster_sizes_per_split):
    plt.plot(cluster_sizes, label=f'Split Size {splits[i] * 100}%')
plt.title('Cluster Sizes Comparison')
plt.xlabel('Cluster Index')
plt.ylabel('Size')
plt.legend()
plt.markers = ['o', '-', '*', 'x', '+']
plt.tight_layout()
plt.savefig('Q4_figB.png')


# Stop SparkSession
spark.stop()
