import os
os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.io.tmpdir=/mnt/d/spark_tmp"

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS


def train_als(train: DataFrame, rank: int, reg: float, max_iter: int) -> ALS:
    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg,
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        numUserBlocks=50,
        numItemBlocks=50,
        checkpointInterval=5
    )

    return als.fit(train)

def compute_ndcg(model, test: DataFrame, k: int = 50):
    gt  = test.groupBy("user_idx").agg(F.collect_set("item_idx").alias("gt_items")).cache()

    gt_sampled = gt.sample(fraction=0.1, seed=42).cache()
    users = gt_sampled.select("user_idx")

    recs = model.recommendForUserSubset(users, k) \
        .withColumn("rec_items", F.col("recommendations.item_idx"))
    
    df = recs.join(gt_sampled, "user_idx")

    df = df.withColumn(
        "dcg",
        F.expr("""
        aggregate(
            sequence(1, CAST(least(size(rec_items), 10) AS INT)),
            CAST(0.0 AS DOUBLE),
            (acc, i) -> acc + 
                CASE 
                    WHEN array_contains(gt_items, rec_items[i-1])
                    THEN 1.0 / log2(CAST(i + 1 AS DOUBLE))
                    ELSE 0.0
                END
        )
        """)
    ).withColumn(
        "idcg",
        F.expr("""
        aggregate(
            sequence(1, CAST(least(size(gt_items), 10) AS INT)),
            CAST(0.0 AS DOUBLE),
            (acc, i) -> acc + 1.0 / log2(CAST(i + 1 AS DOUBLE))
        )
        """)
    ).withColumn(
        "ndcg",
        F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(F.lit(0.0))
    )

    result = df.select(F.mean("ndcg")).first()[0]

    gt.unpersist()
    gt_sampled.unpersist()

    return result

def run_model(train, test, output_dir):
    # for rank = 50, reg = 0.01, NDCG@10 = 0.0003
    # for rank = 100, reg = 0.01, NDCG@10 = 0.000475
    # for rank = 100, reg = 0.1, NDCG@10 =0.000485
    # k = 50, for rank = 100, reg = 0.1, NDCG@10 =0.000484
    # k = 50, for rank = 150, reg = 0.1, NDCG@10 =0.000596

    ranks = [200]
    regs = [0.1]
    max_iter = 10

    train.cache()
    test.cache()
    for rank in ranks:
        for reg in regs:
            print(f"Training rank={rank}, reg={reg}")

            model = train_als(train, rank, reg, max_iter)
            model.write().overwrite().save(f"{output_dir}/50_als_({rank},{reg})_model")

            ndcg = compute_ndcg(model, test)
            print(f"rank={rank}, reg={reg} → NDCG@10 = {ndcg:.6f}")

if __name__ == "__main__":
    spark = SparkSession.builder \
    .appName("RecoScale-ALS-Train_v2") \
    .master("local[3]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.driver.memoryOverhead", "2g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.memory.fraction", "0.7") \
    .config("spark.memory.storageFraction", "0.2") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.local.dir", "/mnt/d/spark_tmp") \
    .config("spark.hadoop.java.io.tmpdir", "/mnt/d/spark_tmp") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.network.timeout", "300s") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

    spark.sparkContext.setCheckpointDir("/mnt/d/spark_checkpoints")

    train = spark.read.parquet("/mnt/d/Projects/recoscale/data/train.parquet")
    test = spark.read.parquet("/mnt/d/Projects/recoscale/data/test.parquet")

    user_sample = test.select("user_idx").distinct().sample(fraction=0.25, seed=42)

    train = train.join(user_sample, on="user_idx").repartition(50, 'user_idx')
    test = test.join(user_sample, on="user_idx")
    run_model(train, test, output_dir="/mnt/d/Projects/recoscale/models")
    
    spark.stop()