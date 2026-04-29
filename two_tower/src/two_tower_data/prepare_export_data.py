import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

# CONFIG
DATA_DIR = "/mnt/d/projects/recoscale/als/data/processed"
EXPORT_DIR = "/mnt/d/projects/recoscale/two_tower/data"
SPARK_TMP = "/mnt/d/spark_tmp"
CHECKPOINT_DIR = "/mnt/d/spark_checkpoints"
CATEGORIES = ["clothing_shoes_and_jewelry", "electronics"]
MIN_USER_INTERACTIONS = 4
MIN_ITEM_INTERACTIONS = 4
MAX_ITERS = 5


def create_spark():
    return (
        SparkSession.builder
        .appName("RecoScale-PrepareColabData")
        .master("local[3]")
        .config("spark.driver.memory", "8g")
        .config("spark.local.dir", SPARK_TMP)
        .config("spark.checkpoint.dir", CHECKPOINT_DIR)
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.driver.memoryOverhead", "2g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.memory.fraction", "0.7") \
        .config("spark.memory.storageFraction", "0.2") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m") \
        .config("spark.network.timeout", "300s") \
        .getOrCreate()
    )


# STEP 1: LOAD + UNION INTERACTIONS
def load_interactions(spark):
    dfs = []
    for cat in CATEGORIES:
        path = f"{DATA_DIR}/{cat}/reviews.parquet"
        df = spark.read.parquet(path).select(
            "user_id", "parent_asin", "rating", "timestamp", "verified_purchase"
        )
        dfs.append(df)
    return dfs[0].unionByName(dfs[1])

# STEP 2: FILTER LOW RATINGS
def filter_ratings(df):
    return df.filter(F.col("rating") > 2)

#STEP 3: APPLYING K-CORE FOR INTERACTION CUTOFF
def apply_k_core(df, min_user=4, min_item=4, max_iters=5):
    prev_n = None

    for i in range(max_iters):
        print(f"  Iteration {i+1}:")
        valid_items = (
            df.groupBy("parent_asin")
              .count()
              .filter(F.col("count") >= min_item)
              .select("parent_asin")
        )
        df = df.join(valid_items, on="parent_asin", how="inner")

        valid_users = (
            df.groupBy("user_id")
              .count()
              .filter(F.col("count") >= min_user)
              .select("user_id")
        )
        df = df.join(valid_users, on="user_id", how="inner")

        n = df.count()
        print(f"k-core iter {i+1}: rows={n:,}")
        if prev_n is not None and n == prev_n:
            print("k-core converged early")
            break
        prev_n = n

    return df

# STEP 4: ENCODE USER + ITEM INDICES
def encode_indices(df):
    # User index
    user_map = (
        df.select("user_id").distinct()
        .rdd.zipWithIndex()
        .map(lambda x: (x[0][0], x[1]))
        .toDF(["user_id", "user_idx"])
    )
    user_map = user_map.withColumn("user_idx", F.col("user_idx").cast(IntegerType()))

    # Item index
    item_map = (
        df.select("parent_asin").distinct()
        .rdd.zipWithIndex()
        .map(lambda x: (x[0][0], x[1]))
        .toDF(["parent_asin", "item_idx"])
    )
    item_map = item_map.withColumn("item_idx", F.col("item_idx").cast(IntegerType()))

    # Join back
    df = df.join(user_map, on="user_id", how="left")
    df = df.join(item_map, on="parent_asin", how="left")

    return df, user_map, item_map

# STEP 5: LEAVE-ONE-OUT SPLIT
def leave_one_out_split(df):
    w = Window.partitionBy("user_idx").orderBy(F.col("timestamp").desc())
    df = df.withColumn("rank", F.rank().over(w))
    train = df.filter(F.col("rank") > 1).drop("rank")
    test = df.filter(F.col("rank") == 1).drop("rank")
    train = train.dropDuplicates(['user_idx', 'item_idx', 'rating', 'timestamp', 'verified_purchase'])
    test = test.dropDuplicates(['user_idx', 'item_idx', 'rating', 'timestamp', 'verified_purchase'])
    test = test.dropDuplicates()
    return train, test

# STEP 6: LOAD + PROCESS ITEM METADATA
def load_item_features(spark, item_map):
    dfs = []
    for cat in CATEGORIES:
        path = f"{DATA_DIR}/{cat}/meta.parquet"
        df = spark.read.parquet(path)
        dfs.append(df)
    meta = dfs[0].unionByName(dfs[1], allowMissingColumns=True)

    # Parse JSON from value column
    from pyspark.sql.types import StructType, StructField, StringType, FloatType
    from pyspark.ml.feature import StringIndexer

    schema = StructType([
        StructField("parent_asin", StringType()),
        StructField("title", StringType()),
        StructField("price", FloatType()),
        StructField("average_rating", FloatType()),
        StructField("rating_number", IntegerType()),
        StructField("store", StringType()),
        StructField("main_category", StringType()),
    ])

    meta = meta.withColumn("parsed", F.from_json(F.col("value"), schema)).select("parsed.*")

    # main_category → binary (1=Electronics, 0=Fashion)
    meta = meta.withColumn(
        "main_category",
        F.when(F.lower(F.col("main_category")).contains("electronic"), 1).otherwise(0)
    )

    # Median impute price
    median_price = meta.approxQuantile("price", [0.5], 0.01)[0]
    meta = meta.withColumn(
        "price",
        F.when(F.col("price").isNull(), median_price).otherwise(F.col("price"))
    )

    meta = meta.withColumn(
        "store",
        F.coalesce(F.col("store"), F.lit("unknown"))
    )
    # Store label encoding via String Indexer
    indexer = StringIndexer(
        inputCol= 'store',
        outputCol= 'store_encoded',
        handleInvalid= 'keep'
    )

    meta = indexer.fit(meta).transform(meta)
    meta = meta.withColumn("store_encoded", F.col("store_encoded").cast("int"))
    meta = meta.drop("store")

    # Join item_idx
    meta = meta.join(item_map, on="parent_asin", how="inner")

    return meta

# STEP 7: EXPORT
def export(df, path, name):
    out = f"{path}/{name}"
    df.write.mode("overwrite").parquet(out)
    print(f"Exported: {out}")

# MAIN
if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)

    spark = create_spark()
    spark.sparkContext.setCheckpointDir(CHECKPOINT_DIR)

    print("Step 1: Loading interactions...")
    interactions = load_interactions(spark)

    print("Step 2: Filtering low ratings...")
    interactions = filter_ratings(interactions)

    print("Step 3: Applying k-core cutoff...")
    interactions = apply_k_core(interactions, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)

    print("Step 4: Encoding indices...")
    interactions, user_map, item_map = encode_indices(interactions)
    interactions.checkpoint()

    print("Step 5: Leave-one-out split...")
    train, test = leave_one_out_split(interactions)

    print("Step 6: Loading item features...")
    item_features = load_item_features(spark, item_map)

    print("Step 7: Exporting...")

    export(
        train.select("user_idx", "item_idx", "rating", "verified_purchase", "timestamp"),
        EXPORT_DIR, "interactions_train"
    )
    export(
        test.select("user_idx", "item_idx", "rating", "verified_purchase", "timestamp"),
        EXPORT_DIR, "interactions_test"
    )
    export(item_features, EXPORT_DIR, "item_features")
    export(user_map, EXPORT_DIR, "user_id_map")
    export(item_map, EXPORT_DIR, "item_id_map")

    print("\n All exports complete.")
    spark.stop()