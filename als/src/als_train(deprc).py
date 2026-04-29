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
        numUserBlocks=100,
        numItemBlocks=100,
        checkpointInterval=2
    )

    return als.fit(train)

def compute_ndcg(model, test: DataFrame, k: int = 10):
    gt  = test.groupBy("user_idx").agg(F.collect_set("item_idx").alias("gt_items")).cache()

    users = gt.sample(fraction=0.1, seed=42).select("user_idx").cache()
    gt_sampled = gt.join(users, "user_idx")  # filter gt to 750K before join

    recs = model.recommendForUserSubset(users, k) \
        .withColumn("rec_items", F.col("recommendations.item_idx"))
    
    df = recs.join(gt_sampled, "user_idx")   # not the full 3M gt
    # df = df.withColumn(
    #     "ndcg",
    #     F.expr("""
    #     aggregate(
    #         sequence(1, size(rec_items)),
    #         CAST(0.0 AS DOUBLE),
    #         (acc, i) -> acc + 
    #             CASE 
    #                 WHEN array_contains(gt_items, rec_items[i-1])
    #                 THEN 1.0 / log2(CAST(i + 1 AS DOUBLE))
    #                 ELSE 0.0
    #              END
    #     )
    #     """)
    # )
    df = df.withColumn(
        "dcg",
        F.expr("""
        aggregate(
            sequence(1, size(rec_items)),
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
    users.unpersist()

    return result

def run_model(train, test, output_dir):
    rank = 50
    reg = 0.01
    max_iter = 10

    print(f"Training rank={rank}, reg={reg}")

    model = train_als(train, rank, reg, max_iter)
    model.write().overwrite().save(f"{output_dir}/als_({rank},{reg})_model")

    ndcg = compute_ndcg(model, test)
    print(f"  NDCG@10 = {ndcg:.4f}")

    return model, ndcg

# def compute_ndcg(model, test: DataFrame, k: int = 10) -> float:
#     # get test users' ground truth
#     ground_truth = test.select("user_idx", "item_idx")
    
#     # get top-k recommendations for all test users
#     test_users = test.select("user_idx").distinct()
#     recs = model.recommendForUserSubset(test_users, k)
#     # recs schema: user_idx, recommendations (array of (item_idx, rating))
    
#     # explode recommendations to get ranked list
#     recs = recs.withColumn("rec_items", 
#     F.col("recommendations.item_idx"))  # extract just item ids as array

#     joined = recs.join(ground_truth, on="user_idx")
#     joined = joined.withColumn(
#         "position",
#         F.expr("array_position(rec_items, item_idx)")  # 1-indexed, 0 if not found
#     )
#     joined = joined.withColumn(
#         "ndcg",
#         F.when(F.col("position") > 0, 
#             F.lit(1.0) / F.log2(F.col("position") + 1))
#         .otherwise(F.lit(0.0))
#     )
#     return joined.agg(F.mean("ndcg")).collect()[0][0]

# def run_grid_search(train, test, output_dir):
#     ranks = [50, 100, 150]
#     regs = [0.01, 0.1]  
#     max_iter = 10
    
#     best_ndcg = -1
#     best_params = {}
#     best_model = None
    
#     for rank in ranks:
#         for reg in regs:
#             print(f"Training rank={rank}, reg={reg}")
#             model = train_als(train, rank, reg, max_iter)
#             ndcg = compute_ndcg(model, test)
#             print(f"  NDCG@10 = {ndcg:.4f}")
#             if ndcg > best_ndcg:
#                 best_ndcg = ndcg
#                 best_params = {"rank": rank, "reg": reg}
#                 best_model = model
    
#     print(f"\nBest: {best_params}, NDCG@10={best_ndcg:.4f}")
#     best_model.save(f"{output_dir}/als_best_model")
#     return best_model, best_ndcg, best_params

if __name__ == "__main__":
    spark = SparkSession.builder \
    .appName("RecoScale-ALS-Train_v2") \
    .master("local[3]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.driver.memoryOverhead", "2g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.memory.fraction", "0.7") \
    .config("spark.memory.storageFraction", "0.2") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.local.dir", "/mnt/d/spark_tmp") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.network.timeout", "300s") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

    spark.sparkContext.setCheckpointDir("/mnt/d/spark_checkpoints")

    train = spark.read.parquet("/mnt/d/Projects/recoscale/data/train.parquet")
    test = spark.read.parquet("/mnt/d/Projects/recoscale/data/test.parquet")

    user_sample = test.select("user_idx").distinct().sample(fraction=0.25, seed=42)

    train = train.join(user_sample, on="user_idx").repartition(100, 'user_idx')
    test = test.join(user_sample, on="user_idx").repartition(100, 'user_idx')
    run_model(train, test, output_dir="/mnt/d/Projects/recoscale/models")
    
    spark.stop()

    # watch -n 15 'du -sh /mnt/d/spark_tmp' 