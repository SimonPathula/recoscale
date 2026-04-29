from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("RecoScale-row-inspection") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.driver.memoryOverhead", "2g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.memory.fraction", "0.7") \
    .config("spark.memory.storageFraction", "0.2") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.local.dir", "/mnt/d/spark_tmp") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.network.timeout", "300s") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

print('Spark Session Started...')

train_df = spark.read.parquet("/mnt/d/Projects/recoscale/data/train.parquet")
test_df = spark.read.parquet("/mnt/d/Projects/recoscale/data/test.parquet")

# print(f"Number of rows in training data before sampling: {train_df.count()}")
# print(f"Number of rows in testing data before sampling: {test_df.count()}")

# print(f"Number of distinct rows in training data before sampling: {train_df.select('user_idx').distinct().count()}")
# print(f"Number of distinct rows in testing data before sampling: {test_df.select('user_idx').distinct().count()}")

user_sampling = test_df.select('user_idx').distinct().sample(fraction= 0.25, seed= 42)
train_df = train_df.join(user_sampling, on= 'user_idx').repartition(200, 'user_idx')
test_df = test_df.join(user_sampling, on= 'user_idx').repartition(200, 'user_idx')

print(f"Number of rows in training data after sampling: {train_df.count()}")
print(f"Number of rows in testing data after sampling: {test_df.count()}")

# distinct_test = test_df.select('user_idx').distinct()
# distinct_train = train_df.select('user_idx').distinct()

# print(f"Number of distict rows in train data after sampling: {distinct_train.count()}")
# print(f"Number of distict rows in test data after sampling: {distinct_test.count()}")