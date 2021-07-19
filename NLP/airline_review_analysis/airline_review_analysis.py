%pyspark

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.sql.functions import monotonically_increasing_id
import re

rawdata = spark.read.load("/tmp/demo_data/airlines.csv", format="csv", header=True)
rawdata = rawdata.fillna({'review': ''})                               # Replace nulls with blank string

# Add Unique ID
rawdata = rawdata.withColumn("uid", monotonically_increasing_id())     # Create Unique ID

# Generate YYYY-MM variable
rawdata = rawdata.withColumn("year_month", rawdata.date.substr(-6,6))

# Generate day
rawdata=rawdata.withColumn("day_tmp",rawdata.date.substr(1,2)).withColumn("day", regexp_replace('day_tmp','-','').cast(IntegerType())).drop('day_tmp')

# Show rawdata (as DataFrame)
rawdata.show(10)

# Print data types
for type in rawdata.dtypes:
    print(type)

target = rawdata.select(rawdata['rating'].cast(IntegerType()))
target.dtypes

################################################################################################
#
# Step1:  Text Pre-processing (consider using one or all of the following):
#       - Remove common words (with stoplist)
#       - Handle punctuation
#       - lowcase/upcase
#       - Stemming: remove common suffix (e.g. studies -> studi, studying -> study). There are different algorithms that can be used in 
#                   the stemming process, but the most common in English is Porter stemmer. The rules contained in this algorithm are 
#                   divided in five different phases numbered from 1 to 5. The purpose of these rules is to reduce the words to the root.
#       - lemmatization: Get the origin of the word (e.g. am, are, was -> be, lives -> live). This is much harder to implement, because
#                    deep linguistics knowledge is required to create the dictionaries that allow the algorithm to look for the proper 
#                    form of the word. But the result will be more accurate.
#       - Part-of-Speech Tagging (nouns, verbs, adj, etc.)
#
# For more info about stemming vs lemmatization:
# https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/
################################################################################################

def cleanup_text(record):
    text  = record[8]
    uid   = record[9]
    words = text.split()
    
    # Default list of Stopwords
    stopwords_core = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', 
    u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', 
    u'can', 'cant', 'come', u'could', 'couldnt', 
    u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', 
    u'each', 
    u'few', 'finally', u'for', u'from', u'further', 
    u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', 
    u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', 
    u'just', 
    u'll', 
    u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', 
    u'no', u'nor', u'not', u'now', 
    u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over', u'own', 
    u'r', u're', 
    u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such', 
    u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', 
    u'under', u'until', u'up', 
    u'very', 
    u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', 
    u'y', u'you', u'your', u'yours', u'yourself', u'yourselves']


# Custom List of Stopwords - Add your own here
    stopwords_custom = ['']
    stopwords = stopwords_core + stopwords_custom
    stopwords = [word.lower() for word in stopwords]    
    
    text_out = [re.sub('[^a-zA-Z0-9]','',word) for word in words]                                       # Remove special characters
    text_out = [word.lower() for word in text_out if len(word)>2 and word.lower() not in stopwords]     # Remove stopwords and words under X length
    return text_out

udf_cleantext = udf(cleanup_text , ArrayType(StringType()))
clean_text = rawdata.withColumn("words", udf_cleantext(struct([rawdata[x] for x in rawdata.columns])))

#tokenizer = Tokenizer(inputCol="description", outputCol="words")
#wordsData = tokenizer.transform(text)

################################################################################################
#
# Step2:  Extract keyword by using TFIDF
#
################################################################################################

# Term Frequency Vectorization  - Option 1 (Using hashingTF): 
#hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
#featurizedData = hashingTF.transform(clean_text)

# Term Frequency Vectorization  - Option 2 (CountVectorizer)    : 
# build the vector of top 1000 words ordered by term frequency 
cv = CountVectorizer(inputCol="words", outputCol="rawFeatures", vocabSize = 2000)
cvmodel = cv.fit(clean_text)
featurizedData = cvmodel.transform(clean_text)

vocab = cvmodel.vocabulary
vocab_broadcast = sc.broadcast(vocab)

# use IDF to reduce weight
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.show(5)

################################################################################################
#
# Step3: LDA Clustering - generate Topics
#
################################################################################################


# Generate 25 Topics:
lda = LDA(k=25, seed=123, optimizer="em", featuresCol="features")

ldamodel = lda.fit(rescaledData)

#model.isDistributed()
#model.vocabSize()

ldatopics = ldamodel.describeTopics()
#ldatopics.show(25)

def map_termID_to_Word(termIndices):
    words = []
    for termID in termIndices:
        words.append(vocab_broadcast.value[termID])
    
    return words

udf_map_termID_to_Word = udf(map_termID_to_Word , ArrayType(StringType()))
ldatopics_mapped = ldatopics.withColumn("topic_desc", udf_map_termID_to_Word(ldatopics.termIndices))
ldatopics_mapped.select(ldatopics_mapped.topic, ldatopics_mapped.topic_desc).show(50,False)




################################################################################################
#
# Step4:  Combine the generated Topics with the Original Airlines Dataset
#
################################################################################################

ldaResults = ldamodel.transform(rescaledData)

ldaResults.select('id','airline','date','cabin','rating','words','features','topicDistribution').show(5)

################################################################################################
#
# Step5:  Breakout LDA Topics for Modeling and Reporting
#
################################################################################################



def breakout_array(index_number, record):
    vectorlist = record.tolist()
    return vectorlist[index_number]

udf_breakout_array = udf(breakout_array, FloatType())

# Extract document weights for Topics 12(Bad_feeling) and 20(Good_feeling)
enrichedData = ldaResults                                                                   \
        .withColumn("Bad_feeling", udf_breakout_array(lit(12), ldaResults.topicDistribution))  \
        .withColumn("Good_feeling", udf_breakout_array(lit(20), ldaResults.topicDistribution))            

enrichedData.select('id','airline','date','year_month','day','cabin','rating','words','features','topicDistribution','Bad_feeling','Good_feeling').show(5)

#enrichedData.agg(max("Bad_feeling")).show()


################################################################################################
#
# Step6: Register Table for SparkSQL
#
################################################################################################

enrichedData.createOrReplaceTempView("enrichedData")

################################################################################################
#
# Step7:  Visualize Airline Volume and Average Rating Trends (by Date)
#
################################################################################################
%sql
SELECT id, airline, date, day, year_month, rating, Bad_feeling, Good_feeling FROM enrichedData where airline = "${item=Delta Air Lines,Delta Air Lines|US Airways|Southwest Airlines|American Airlines|United Airlines}" and year_month == "Jun-14" order by day

################################################################################################
#
# Step8:  Visualize Airline Ratings over time (for all 5 Airlines)
#
################################################################################################

SELECT id, airline, date, year_month, rating, Bad_feeling, Good_feeling FROM enrichedData where date >= "2015-01-01" order by date

################################################################################################
#
# Step9:  Visualize relation between topic and rating 
#
################################################################################################

SELECT id, airline, date, year_month, day ,rating, Bad_feeling, Good_feeling FROM enrichedData where year_month == "Jun-14" order by day 

SELECT id, airline, date, year_month, day ,rating, Bad_feeling, Good_feeling FROM enrichedData where year_month == "Jun-14" order by day 

################################################################################################
#
# Step10:  Show all reviews related to Topic 12(bad feeling)
#
################################################################################################

SELECT ID, DATE, AIRLINE, REVIEW, Bad_feeling FROM ENRICHEDDATA WHERE Bad_feeling >= 0.25 ORDER BY Bad_feeling DESC
