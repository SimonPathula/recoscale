from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def split_train_test(df: DataFrame) -> tuple[DataFrame, DataFrame]:

    #calculate the interaction count
    user_counts = df.groupBy('user_idx').count().withColumnRenamed('count', 'interaction_count')
    df = df.join(user_counts, on= 'user_idx', how= 'left')

    #now give the order number in descending order
    w = Window.partitionBy('user_idx').orderBy(F.col('timestamp').desc())
    df = df.withColumn('rn', F.row_number().over(w))

    df.persist()

    #split the data into train and test
    test  = df.filter((F.col("rn") <= 2) & (F.col("interaction_count") >= 5))
    train = df.filter(~((F.col("rn") <= 2) & (F.col("interaction_count") >= 5)))

    df.unpersist()

    cols_to_drop = ["rn", "interaction_count"]

    train = train.drop(*cols_to_drop).select(
        F.col("user_idx").cast("int"),
        F.col("item_idx").cast("int"),
        F.col("rating").cast("float"),
        F.col("timestamp")
    )
    test = test.drop(*cols_to_drop).select(
        F.col("user_idx").cast("int"),
        F.col("item_idx").cast("int"),
        F.col("rating").cast("float"),
        F.col("timestamp")
    )
    train = train.dropDuplicates(['user_idx', 'item_idx', 'rating', 'timestamp'])
    test = test.dropDuplicates(['user_idx', 'item_idx', 'rating', 'timestamp'])
    
    return train, test

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("RecoScale-ALS-Split") \
        .master("local[3]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.memory.fraction", "0.7") \
        .config("spark.memory.storageFraction", "0.2") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.local.dir", "/mnt/d/spark_tmp") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.network.timeout", "300s") \
        .getOrCreate()
    
    df = spark.read.parquet("data/df_indexed_110M.parquet")
    train, test = split_train_test(df)
    train.write.mode('overwrite').parquet("/mnt/d/Projects/recoscale/data/train.parquet")
    test.write.mode('overwrite').parquet("/mnt/d/Projects/recoscale/data/test.parquet")