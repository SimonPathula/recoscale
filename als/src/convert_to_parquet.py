from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import from_json, col
import json


schema = StructType([
    StructField('asin', StringType(), True),
    StructField('helpful_vote', IntegerType(), True),
    StructField('images', StringType(), True),
    StructField('parent_asin', StringType(), True),
    StructField('rating', FloatType(), True),
    StructField('text', StringType(), True),
    StructField('timestamp', LongType(), True),
    StructField('title', StringType(), True),
    StructField('user_id', StringType(), True),
    StructField('verified_purchase', BooleanType(), True)
])

CATEGORIES = {
    "books": "Books",
    "clothing_shoes_and_jewelry": "Clothing_Shoes_and_Jewelry",
    "electronics": "Electronics",
    "health_and_household": "Health_and_Household",
    "home_and_kitchen": "Home_and_Kitchen",
    "sports_and_outdoors": "Sports_and_Outdoors"
}

BASE = "/mnt/d/Projects/recoscale/data"

def convert_category(category_key, category_name):

    spark = SparkSession.builder \
        .appName('cat_to_par') \
        .config('spark.driver.memory', '8g') \
        .config('spark.executor.heartbeatInterval', '60s') \
        .config('spark.network.timeout', '300s') \
        .getOrCreate()
    
    review_df = spark.read.json(f'{BASE}/raw/{category_key}/{category_name}.jsonl.gz', schema= schema)
    meta_df = spark.read.text(f'{BASE}/raw/{category_key}/meta_{category_name}.jsonl.gz')

    review_out = f'{BASE}/processed/{category_key}/reviews.parquet'
    meta_out = f'{BASE}/processed/{category_key}/meta.parquet'

    review_df.coalesce(8).write.mode("overwrite").parquet(review_out)
    meta_df.coalesce(8).write.mode("overwrite").parquet(meta_out)

    print(f"Reviews: {spark.read.parquet(review_out).count()} rows")
    print(f"Meta: {spark.read.parquet(meta_out).count()} rows")

    spark.stop()
    
if __name__ == "__main__":
    for key, name in CATEGORIES.items():
        print(f"\n=== Processing {name} ===")
        convert_category(key, name)