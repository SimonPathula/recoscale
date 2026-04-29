from pyspark.sql import SparkSession

CATEGORIES = {
    # "books": "Books",
    "clothing_shoes_and_jewelry": "Clothing_Shoes_and_Jewelry",
    "electronics": "Electronics"
    # "health_and_household": "Health_and_Household",
    # "home_and_kitchen": "Home_and_Kitchen",
    # "sports_and_outdoors": "Sports_and_Outdoors"
}

BASE = "/mnt/d/Projects/recoscale/data/processed"

def union_dataset(spark):
    
    union_df = None

    for key, value in CATEGORIES.items():

        print(f"\n=== Adding{value} ===")
    
        df = spark.read.parquet(f"{BASE}/{key}/reviews.parquet").select("user_id", "parent_asin", "rating", "timestamp")

        union_df = df if union_df is None else union_df.union(df)

    union_df.write.mode('overwrite').parquet("/mnt/d/Projects/recoscale/data/union_df_110M.parquet")
    union_df.printSchema()
    union_df.show(5)

    return union_df

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName('als_train') \
        .config('spark.driver.memory', '5g') \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.memory.storageFraction", "0.3") \
        .config('spark.executor.heartbeatInterval', '60s') \
        .config('spark.network.timeout', '300s') \
        .getOrCreate()

    union_df = union_dataset(spark)

    spark.stop()