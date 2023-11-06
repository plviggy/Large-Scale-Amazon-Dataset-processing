import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR
# -------- Import your own dependencies--------
from pyspark.sql.functions import desc, asc, count, mean, avg, variance, isnan, col, split, isnull
from pyspark.sql.functions import map_keys, map_values
from pyspark.sql.functions import when, countDistinct, size, explode, explode_outer
from pyspark.sql.functions import array_contains
from pyspark.sql.functions import lower, first, last
from pyspark.ml.feature import Word2Vec, OneHotEncoder, StringIndexer, PCA
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Summarizer
#-----------------------------


# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics


# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------




def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    # Group by on reviews table, and calculate product mean ratings and count ratings
    df1 = review_data.groupBy('asin').agg(mean("overall").alias("MeanRating"),
                                          count("overall").alias("CountRatings"))

    # Select only relevant columns from product_data
    product_data_filtered = product_data.select("asin")

    # Perform a left join of filtered product_data with df1
    t1 = product_data_filtered.join(df1, "asin", "left")

    agg_results = t1.agg(count("*").alias("count_total"),
                         mean("MeanRating").alias("mean_meanRating"),
                         variance("MeanRating").alias("variance_meanRating"),
                         count(when(isnull("MeanRating"), True)).alias("numNulls_meanRating"),
                         mean("CountRatings").alias("mean_countRating"),
                         variance("CountRatings").alias("variance_countRating"),
                         count(when(isnull("CountRatings"), True)).alias("numNulls_countRating")).first()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:

    res = {
        'count_total': agg_results["count_total"],
        'mean_meanRating': agg_results["mean_meanRating"],
        'variance_meanRating': agg_results["variance_meanRating"],
        'numNulls_meanRating': agg_results["numNulls_meanRating"],
        'mean_countRating': agg_results["mean_countRating"],
        'variance_countRating': agg_results["variance_countRating"],
        'numNulls_countRating': agg_results["numNulls_countRating"]
    }

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------




def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    product_data = product_data.withColumn('category', when(col('categories').isNotNull() & (col('categories')[0][0] != ''),
                                                            col('categories')[0][0]).otherwise(None))

    product_data = product_data.withColumn('bestSalesCategory', when(col('salesRank').isNotNull(), F.map_keys("salesRank")[0].alias("bestSalesCategory")))
    product_data = product_data.withColumn('bestSalesRank', when(col('salesRank').isNotNull(), F.map_values("salesRank")[0].alias("bestSalesRank")))


    t2 = product_data.select('asin','category','bestSalesCategory','bestSalesRank')

    agg_results = t2.agg(count("*").alias("count_total"),
                         mean("bestSalesRank").alias("mean_bestSalesRank"),
                         variance("bestSalesRank").alias("variance_bestSalesRank"),
                         count(when(isnull("category"), True)).alias("numNulls_category"),
                         countDistinct("category").alias("countDistinct_category"),
                         count(when(isnull("bestSalesCategory"), True)).alias("numNulls_bestSalesCategory"),
                         countDistinct("bestSalesCategory").alias("countDistinct_bestSalesCategory"),
                        ).first()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:

    res = {
        'count_total': agg_results["count_total"],
        'mean_bestSalesRank': agg_results["mean_bestSalesRank"],
        'variance_bestSalesRank': agg_results["variance_bestSalesRank"],

        'numNulls_category': agg_results["numNulls_category"],
        'countDistinct_category': agg_results["countDistinct_category"],

        'numNulls_bestSalesCategory': agg_results["numNulls_bestSalesCategory"],
        'countDistinct_bestSalesCategory': agg_results["countDistinct_bestSalesCategory"]
    }
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    # LEFT SIDE OF JOIN
    product_data = product_data.withColumn("also_viewed",product_data.related['also_viewed'])
    exploded_prod_table = product_data['asin',explode_outer('also_viewed').alias('asin_l_rel')]
    #exploded_prod_table.show(10)

    # RIGHT SIDE OF JOIN
    right = product_data['asin', 'price'].withColumnRenamed('asin','asin_r').withColumnRenamed('price','price_r')

    # JOIN
    joined_df = exploded_prod_table.join(right,[exploded_prod_table.asin_l_rel == right.asin_r],"left")
    #joined_df.show(10)

    # Get list of mean prices of related products
    mean_price_rel = joined_df.groupBy('asin').agg(mean("price_r").alias("meanPriceAlsoViewed"))

    # Get required output columns into t3
    t3 = product_data.join(mean_price_rel,[product_data.asin == mean_price_rel.asin],"left")
    t3 = t3.withColumn("count_also_viewed", when(col("related.also_viewed").isNull(), None).otherwise(size("related.also_viewed")))

    agg_results = t3.agg(count("*").alias("count_total"),
                         mean("meanPriceAlsoViewed").alias("mean_meanPriceAlsoViewed"),
                         variance("meanPriceAlsoViewed").alias("variance_meanPriceAlsoViewed"),
                         count(when(isnull("meanPriceAlsoViewed"), True)).alias("numNulls_meanPriceAlsoViewed"),
                         mean("count_also_viewed").alias("mean_countAlsoViewed"),
                         variance("count_also_viewed").alias("variance_countAlsoViewed"),
                         count(when(isnull("count_also_viewed"), True)).alias("numNulls_countAlsoViewed")
                        ).first()


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:

    res = {
        'count_total': agg_results["count_total"],

        'mean_meanPriceAlsoViewed': agg_results["mean_meanPriceAlsoViewed"],
        'variance_meanPriceAlsoViewed': agg_results["variance_meanPriceAlsoViewed"],
        'numNulls_meanPriceAlsoViewed': agg_results["numNulls_meanPriceAlsoViewed"],

        'mean_countAlsoViewed': agg_results["mean_countAlsoViewed"],
        'variance_countAlsoViewed': agg_results["variance_countAlsoViewed"],
        'numNulls_countAlsoViewed': agg_results["numNulls_countAlsoViewed"]
    }



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    # Compute the mean
    mean_price = product_data.select(mean("price").alias("mean_price")).first()["mean_price"]

    # Compute the approximate median
    med_price = product_data.approxQuantile("price", [0.5], 0.0001)[0]

    # Display the results
    price_stats = {"mean_price": mean_price,
                   "med_price": med_price
                  }

    #print(price_stats)

    product_data = product_data.withColumn("meanImputedPrice",when(col('price').isNull(), price_stats['mean_price']).otherwise(col('price')))
    product_data = product_data.withColumn("medianImputedPrice",when(col('price').isNull(),price_stats['med_price']).otherwise(col('price')))
    product_data = product_data.withColumn("unknownImputedTitle", when((col('title').isNull()) | (col('title') == ''), 'unknown').otherwise(col('title')))
    product_data = product_data.withColumn("unknown_titles", when((col('title').isNull()) | (col('title') == ''), 1).otherwise(None))

    t4 = product_data.select('asin','title','price','meanImputedPrice','medianImputedPrice','unknownImputedTitle','unknown_titles')

    #t4.show(5)

    agg_results = t4.agg(count("*").alias("count_total"),

                         mean("meanImputedPrice").alias("mean_meanImputedPrice"),
                         variance("meanImputedPrice").alias("variance_meanImputedPrice"),
                         count(when(isnull("meanImputedPrice"), True)).alias("numNulls_meanImputedPrice"),

                         mean("medianImputedPrice").alias("mean_medianImputedPrice"),
                         variance("medianImputedPrice").alias("variance_medianImputedPrice"),
                         count(when(isnull("medianImputedPrice"), True)).alias("numNulls_medianImputedPrice"),
                         count("unknown_titles").alias("numUnknowns_unknownImputedTitle")
                        ).first()



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:

    res = {
        'count_total': agg_results["count_total"],
        'mean_meanImputedPrice': agg_results["mean_meanImputedPrice"],
        'variance_meanImputedPrice': agg_results["variance_meanImputedPrice"],
        'numNulls_meanImputedPrice': agg_results["numNulls_meanImputedPrice"],
        'mean_medianImputedPrice': agg_results["mean_medianImputedPrice"],
        'variance_medianImputedPrice': agg_results["variance_medianImputedPrice"],
        'numNulls_medianImputedPrice': agg_results["numNulls_medianImputedPrice"],
        'numUnknowns_unknownImputedTitle': agg_results["numUnknowns_unknownImputedTitle"],
    }



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    product_processed_data = product_processed_data.select('asin','title')
    product_processed_data = product_processed_data.withColumn("titleArray",F.split(lower(col('title')),"\s+"))

    w2v  = Word2Vec(vectorSize = 16, minCount = 100, numPartitions = 4, seed=102,inputCol="titleArray", outputCol="word2vec_features")
    model = w2v.fit(product_processed_data)



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:
    res['count_total'] = product_processed_data.count()
    res['size_vocabulary'] = model.getVectors().count()

    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------

    #String Indexer
    stringIndexer = StringIndexer(inputCol="category", outputCol="category_indexed")
    si_model = stringIndexer.fit(product_processed_data)
    product_processed_data = si_model.transform(product_processed_data)

    #One Hot Encoder
    ohe = OneHotEncoder(inputCol="category_indexed", outputCol="categoryOneHot", dropLast = False)
    single_col_model = ohe.fit(product_processed_data)
    product_processed_data = single_col_model.transform(product_processed_data)

    #PCA
    pca = PCA(k=15, inputCol="categoryOneHot", outputCol="categoryPCA")
    pca_model = pca.fit(product_processed_data)
    product_processed_data = pca_model.transform(product_processed_data)

    #Mean Vector Summarizer
    summarizer = Summarizer.metrics("mean")
    mean_ohe = product_processed_data.select(Summarizer.mean(product_processed_data["categoryOneHot"])).collect()
    mean_pca = product_processed_data.select(Summarizer.mean(product_processed_data["categoryPCA"])).collect()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:

    res = {
        'count_total': product_processed_data.count(),
        'meanVector_categoryOneHot': mean_ohe[0][0],
        'meanVector_categoryPCA': mean_pca[0][0]
    }

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    #Decision Tree Regression
    dt_regressor = DecisionTreeRegressor(featuresCol = 'features',labelCol = 'overall', maxDepth = 5)

    #fit to train data
    model = dt_regressor.fit(train_data)

    #predict with test data
    predictions = model.transform(test_data)

    #RMSE Calculation
    r_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="overall", metricName="rmse")
    rmse = r_evaluator.evaluate(predictions)

    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:

    res = {
        'test_rmse': rmse
    }

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------

    #Splitting the train_data
    train_ratio = 0.75
    train2, test2 = train_data.randomSplit([train_ratio, 1 - train_ratio], seed=42)

    min_rmse_model=[]
    rmses=[]
    min_rmse = float('inf')
    min_rmse_depth = None

    rmse_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="overall", metricName="rmse")

    #Decision Tree Regression
    for depth in [12,9,7,5]:

        dt_regressor = DecisionTreeRegressor(featuresCol = 'features',labelCol = 'overall', maxDepth = depth)
        model = dt_regressor.fit(train2)
        predictions2 = model.transform(test2)
        rmse = rmse_evaluator.evaluate(predictions2)

        if rmse < min_rmse:
            min_rmse = rmse
            min_rmse_model = [model]
            min_rmse_depth = depth

        rmses.append(rmse)

    test_data_predictions = min_rmse_model[0].transform(test_data)
    test_rmse = rmse_evaluator.evaluate(test_data_predictions)
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:

    res = {
        'test_rmse': test_rmse,
        'valid_rmse_depth_5': rmses[3],
        'valid_rmse_depth_7': rmses[2],
        'valid_rmse_depth_9': rmses[1],
        'valid_rmse_depth_12': rmses[0]
    }

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------

