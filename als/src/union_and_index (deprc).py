from pyspark.sql import SparkSession
from pyspark.sql.types import *

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

BASE = "/mnt/d/Projects/recoscale/data/processed"

def union_and_indexed(spark):
    
    union_df = None

    for key, value in CATEGORIES.items():

        print(f"\n=== Adding{value} ===")
    
        df = spark.read.parquet(f"{BASE}/{key}/reviews.parquet", schema = schema).select("user_id", "parent_asin", "rating", "timestamp")

        union_df = df if union_df is None else union_df.union(df)

    user_mapping = union_df.select('user_id') \
        .rdd.map(lambda r: r[0]) \
        .zipWithIndex() \
        .toDF(['user_id', 'user_idx'])

    product_mapping = union_df.select('parent_asin') \
        .rdd.map(lambda r: r[0]) \
        .zipWithIndex() \
        .toDF(['parent_asin', 'item_idx'])

    df_indexed = union_df.join(user_mapping, on = 'user_id') \
                     .join(product_mapping, on = 'parent_asin')
    
    df_indexed = df_indexed.coalesce(8)
    df_indexed = df_indexed.write.parquet("/mnt/d/Projects/recoscale/data/df_indexed.parquet")
    df_indexed.printSchema()
    df_indexed.show(5)

    return df_indexed

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName('als_train') \
        .config('spark.driver.memory', '8g') \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.memory.storageFraction", "0.3") \
        .config('spark.executor.heartbeatInterval', '60s') \
        .config('spark.network.timeout', '300s') \
        .getOrCreate()

    df_indexed = union_and_indexed(spark)

    spark.stop()