from pyspark.sql import SparkSession

BASE = "/mnt/d/Projects/recoscale/data/"

def indexing(spark):
   
    union_df = spark.read.parquet("/mnt/d/Projects/recoscale/data/union_df_110M.parquet")

    user_mapping = union_df.select('user_id').distinct() \
            .rdd.map(lambda r: r[0]) \
            .zipWithIndex() \
            .toDF(['user_id', 'user_idx'])

    product_mapping = union_df.select('parent_asin').distinct() \
            .rdd.map(lambda r: r[0]) \
            .zipWithIndex() \
            .toDF(['parent_asin', 'item_idx'])

    df_indexed = union_df.join(user_mapping, on = 'user_id') \
                        .join(product_mapping, on = 'parent_asin')

    df_indexed.coalesce(8).write.mode("overwrite") \
    .parquet("/mnt/d/Projects/recoscale/data/df_indexed_110M.parquet")

    print("Write complete")


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("als_indexing") \
        .config("spark.driver.memory", "5g") \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.local.dir", "/mnt/d/spark_tmp") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.network.timeout", "300s") \
        .getOrCreate()

    indexing(spark)
    spark.stop()

