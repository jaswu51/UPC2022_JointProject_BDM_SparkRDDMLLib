from sre_constants import RANGE
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, lit
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark import SparkConf
from pyspark.context import SparkContext


import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download(
    [
        "names",
        "stopwords",
        "state_union",
        "twitter_samples",
        "movie_reviews",
        "averaged_perceptron_tagger",
        "vader_lexicon",
        "punkt",
        "maxent_ne_chunker",
        "words",
    ]
)
sia = SentimentIntensityAnalyzer()

import re

"""Read csv and save to Mongo"""
# in bpm_p2, we were unable to use the origianl kaggle dataset due to limitations, thus need to import new files to Mongo for P2 usage

if __name__ == "__main__":
    spark = (
        SparkSession.builder.master(f"local[*]")
        .appName("myApp")
        .config(
            "spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1"
        )
        .getOrCreate()
    )


schema1 = StructType(
    [
        StructField("sequence", StringType(), True),
        StructField("name", StringType(), False),
        StructField("text", StringType(), False),
        StructField("publishAt", StringType(), False),
        StructField("publishedAtDate", StringType(), False),
        StructField("likesCount", StringType(), False),
        StructField("reviewId", StringType(), False),
        StructField("reviewUrl", StringType(), False),
        StructField("reviewerId", StringType(), False),
        StructField("reviewerUrl", StringType(), False),
        StructField("reviewerNumberOfReviews", StringType(), False),
        StructField("isLocalGuide", StringType(), False),
        StructField("stars", StringType(), False),
        StructField("rating", StringType(), False),
        StructField("responseFromOwnerDate", StringType(), False),
        StructField("responseFromOwnerText", StringType(), False),
        StructField("biz_name", StringType(), False),
        StructField("biz_id", StringType(), False),
    ]
)

# we choose a new dataset, and in P2 we need to load it again into MongoDB
df = spark.read.csv(
    "/Users/xiaokeai/Downloads/SparkMongoMWE/Google_Apify_Reviews.csv",
    sep=";",
    header=False,
    schema=schema1,
)
print(df.head(1))

df.write.format("mongo").mode("overwrite").option(
    "spark.mongodb.output.uri",
    "mongodb://127.0.0.1/bdm_project2022.Google_Apify_Reviews",
).save()


"""Read Spark Dataframe from Mongo"""

GM_DF = (
    spark.read.format("mongo")
    .option("uri", f"mongodb://127.0.0.1/bdm_project2022.Google_Apify_Reviews")
    .load()
)
print(GM_DF.first())
GM_RDD = GM_DF.rdd

"""Formatting Zone"""

# Eliminate Null/Meaningless Columns and Null Rows
nonNullColumn_list = []
ll = int(len(GM_DF.columns))
for i in range(0, ll):
    if GM_RDD.filter(lambda x: x[i] == None).count() < 0.3 * (GM_RDD.count()):
        nonNullColumn_list.append(i)
print(nonNullColumn_list)
GM_RDD_notNull = GM_RDD.map(lambda x: [(x[i]) for i in nonNullColumn_list]).filter(
    lambda row: all(x != None for x in row)
)

print(GM_RDD_notNull.first())
# One-hot encoding on column 'isLocalGuide' True/False values
def trueFalseEncoder(x):
    if x == "False":
        return 0
    elif x == "True":
        return 1


GM_RDD__trueFalseEncoder = GM_RDD_notNull.map(
    lambda x: ((x, x[-1]), trueFalseEncoder(x[3]),)
)
print(GM_RDD__trueFalseEncoder.first())

# Create Sentiment Score on column 'text'
def sentenceSentimentScore(x):
    return (
        sia.polarity_scores(x)["compound"],
        sia.polarity_scores(x)["neg"],
        sia.polarity_scores(x)["neu"],
        sia.polarity_scores(x)["pos"],
    )


def regexRemoveTranslation(x):
    a = re.sub("(\\(Original).*$", "", x)
    b = re.sub("(\\(Translated by Google)\\)", "", a)
    return b


GM_RDD__sentenceSentimentScore = (
    GM_RDD__trueFalseEncoder.map(
        lambda x: ((x[0][0], regexRemoveTranslation(x[0][1]), x[1]))
    )
    .map(
        lambda x: (
            x[0][1],
            x[0][2],
            x[0][3],
            x[0][4],
            x[0][9],
            x[0][10],
            x[0][11],
            x[1],
            sentenceSentimentScore(x[1]),
            x[2],
        )
    )
    .map(
        lambda x: (
            x[0],
            x[1],
            int(x[3]),
            int(float(x[4])),
            int(float(x[6])),
            x[7],
            x[8][0],
            x[8][1],
            x[8][2],
            x[8][3],
            x[9],
        )
    )
)
print(GM_RDD__sentenceSentimentScore.first())


# Create GM_RDD_isResponse
GM_RDD_isResponse = GM_RDD.map(lambda x: (x[1], x[8]))
print(GM_RDD_isResponse.first())

# Create schema
schema__sentenceSentimentScore = StructType(
    [
        StructField("restaurantID", StringType(), True),
        StructField("restaurantName", StringType(), True),
        StructField("likesCount", StringType(), True),
        StructField("reviewerNumberOfReviews", StringType(), True),
        StructField("stars", StringType(), True),
        StructField("textCleaned", StringType(), True),
        StructField("sentimentCompound", FloatType(), True),
        StructField("sentimentNeg", FloatType(), True),
        StructField("sentimentNeutrarl", FloatType(), True),
        StructField("sentimentPositive", FloatType(), True),
        StructField("isLocalGuideTrueFalse", StringType(), True),
    ]
)
schema_isResponse = StructType(
    [
        StructField("restaurantID", StringType(), True),
        StructField("isResponde", StringType(), True),
    ]
)


GM_DF__sentenceSentimentScore = GM_RDD__sentenceSentimentScore.toDF(
    schema__sentenceSentimentScore
)

GM_DF_isResponse = GM_RDD_isResponse.toDF(schema_isResponse)
print(GM_DF__sentenceSentimentScore.head(1))
print(GM_DF_isResponse.head(1))
"""Save Spark Dataframe to Mongo"""
GM_DF_isResponse.write.format("com.mongodb.spark.sql.DefaultSource").mode(
    "overwrite"
).option(
    "spark.mongodb.output.uri", "mongodb://127.0.0.1/bdm_project2022.GM_DF_isResponse"
).save()
GM_DF__sentenceSentimentScore.write.format("com.mongodb.spark.sql.DefaultSource").mode(
    "overwrite"
).option(
    "spark.mongodb.output.uri",
    "mongodb://127.0.0.1/bdm_project2022.GM_DF__sentenceSentimentScore",
).save()

