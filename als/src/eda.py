from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
        .appName('recoscale-eda') \
        .config('spark.driver.memory', '8g') \
        .config('spark.sql.shuffle.partitions', '200') \
        .getOrCreate()

spark.sparkContext.setLogLevel('ERROR')

categories = {
    "Electronics": "data/raw/electronics/Electronics.jsonl.gz", #43886944
    "Books": "data/raw/books/Books.jsonl.gz", #29475453
    "Home_and_Kitchen": "data/raw/home_and_kitchen/Home_and_Kitchen.jsonl.gz", #67409944 
    "Clothing_Shoes_and_Jewelry": "data/raw/clothing_shoes_and_jewelry/Clothing_Shoes_and_Jewelry.jsonl.gz", #66033346
    "Health_and_Household": "data/raw/health_and_household/Health_and_Household.jsonl.gz", #25631345
    "Sports_and_Outdoors": "data/raw/sports_and_outdoors/Sports_and_Outdoors.jsonl.gz", #19595170
}

for name, path in categories.items():
    print(f"\n=== Loading {name} ===")
    df = spark.read.json(path)

    print("\n=== Schema ===")
    df.printSchema()

    print("\n=== Row Count ===")
    print(f"Total no.of rows = {df.count()}") 

    print("\n=== Rating Distribution ===")
    df.groupBy('rating').count().orderBy('rating').show()

    print("\n=== Reviews per User (distribution) ===")
    user_activity = df.groupBy("user_id").count().withColumnRenamed("count", "review_count")
    user_activity.select(
        F.min("review_count").alias("min"),
        F.percentile_approx("review_count", 0.25).alias("p25"),
        F.percentile_approx("review_count", 0.50).alias("median"),
        F.percentile_approx("review_count", 0.75).alias("p75"),
        F.max("review_count").alias("max"),
        F.avg("review_count").alias("mean")
    ).show()

    print("\n=== Reviews per Item (distribution) ===")
    item_activity = df.groupBy("parent_asin").count().withColumnRenamed("count", "review_count")
    item_activity.select(
        F.min("review_count").alias("min"),
        F.percentile_approx("review_count", 0.25).alias("p25"),
        F.percentile_approx("review_count", 0.50).alias("median"),
        F.percentile_approx("review_count", 0.75).alias("p75"),
        F.max("review_count").alias("max"),
        F.avg("review_count").alias("mean")
    ).show()

    print("\n=== Null Check ===")
    df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in ["user_id", "parent_asin", "rating", "timestamp"]
    ]).show()

spark.stop()