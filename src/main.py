from argparse import ArgumentParser
import time
import psutil
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, sum as spark_sum
from delta import configure_spark_with_delta_pip
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
import mlflow
import mlflow.spark

def get_args():
    parser = ArgumentParser(description="Steam Bans ML Pipeline")
    parser.add_argument('--input-path', '-i', default="/app/data/steambans.csv", help="Path to input CSV")
    parser.add_argument('--enable-cache', '-c', action='store_true', help="Enable caching for performance")
    return parser.parse_args()

def run_pipeline(input_path, enable_cache):
    # Initialize Spark with Delta Lake
    session_builder = (
        SparkSession.builder
        .appName("SteamBansAnalysis")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.memory", "5g")
        .config("spark.sql.shuffle.partitions", "150")
    )
    spark_session = configure_spark_with_delta_pip(session_builder).getOrCreate()
    spark_session.sparkContext.setLogLevel("WARN")
    mlflow.set_tracking_uri("file:/app/logs/mlflow")

    start_time = time.time()

    # Load raw data
    print("Loading raw data from CSV")
    raw_data = spark_session.read.option("header", "true").csv(input_path)

    # Define column types
    type_mappings = {
        "steam_level": "integer",
        "friends_count": "integer",
        "game_count": "integer",
        "total_playtime": "float",
        "cs2_playtime": "float",
        "vac_bans": "integer"
    }
    for col_name, col_type in type_mappings.items():
        raw_data = raw_data.withColumn(col_name, col(col_name).cast(col_type))

    # Bronze layer: Save raw data
    print("Writing to bronze layer")
    raw_data.repartition(2).write.format("delta").mode("overwrite").save("./data/bronze/")
    spark_session.sql("OPTIMIZE delta.`/app/data/bronze` ZORDER BY (vac_bans)")

    # Silver layer: Clean and transform data
    print("Processing for silver layer")
    processed_data = spark_session.read.format("delta").load("./data/bronze/").filter(col("steam_level") >= 0)
    processed_data = processed_data.dropDuplicates()

    # Handle missing values
    numeric_columns = ["steam_level", "friends_count", "game_count", "total_playtime", "cs2_playtime", "vac_bans"]
    for col_name in numeric_columns:
        mean_val = processed_data.select(spark_sum(col_name) / processed_data.count()).first()[0]
        processed_data = processed_data.fillna({col_name: mean_val})

    # Feature engineering: Create playtime ratio
    processed_data = processed_data.withColumn("cs2_playtime_ratio", 
                                              col("cs2_playtime") / (col("total_playtime") + 1.0))

    # Save to silver layer
    processed_data.write.format("delta").mode("overwrite").save("./data/silver/")

    # Analysis
    print("Dataset Overview:")
    processed_data.printSchema()
    print("Unique users by VAC ban status:")
    processed_data.groupBy("vac_bans").agg(countDistinct("personaname").alias("unique_users")).show()
    print("Total CS2 playtime by community visibility:")
    processed_data.groupBy("communityvisibilitystate").agg(spark_sum("cs2_playtime").alias("total_cs2_hours")).show()

    # Prepare for ML: Gold layer
    print("Preparing gold layer for ML")
    ml_data = processed_data.drop("personaname", "economy_ban", "communityvisibilitystate")
    visibility_indexer = StringIndexer(inputCol="community_visibility", outputCol="visibility_index")
    ml_data = visibility_indexer.fit(ml_data).transform(ml_data)

    # Save to gold layer
    ml_data.write.format("delta").mode("overwrite").save("./data/gold/")

    # ML Pipeline: Predict vac_bans
    feature_columns = ["steam_level", "friends_count", "game_count", "total_playtime", 
                      "cs2_playtime_ratio", "visibility_index"]
    feature_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
    ml_data = feature_assembler.transform(ml_data)

    train_set, test_set = ml_data.randomSplit([0.8, 0.2], seed=42)
    if enable_cache:
        train_set = train_set.cache().repartition(4)
        test_set = test_set.cache().repartition(4)

    # Train Random Forest model
    with mlflow.start_run(run_name="RandomForest_SteamBans"):
        rf_model = RandomForestClassifier(
            featuresCol="features",
            labelCol="vac_bans",
            numTrees=50,
            maxDepth=10,
            seed=42
        )
        trained_model = rf_model.fit(train_set)
        predictions = trained_model.transform(test_set)

        # Log model and parameters
        mlflow.spark.log_model(trained_model, "random_forest_model")
        mlflow.log_param("num_trees", rf_model.getNumTrees())
        mlflow.log_param("max_depth", rf_model.getMaxDepth())

        # Log metrics (example: accuracy from evaluator)
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        evaluator = MulticlassClassificationEvaluator(
            labelCol="vac_bans", predictionCol="prediction", metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        mlflow.log_metric("accuracy", accuracy)

    # Performance metrics
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.2f} MB")

    spark_session.stop()

if __name__ == "__main__":
    args = get_args()
    run_pipeline(args.input_path, args.enable_cache)