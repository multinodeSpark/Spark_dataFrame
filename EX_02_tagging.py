
# coding: utf-8

# In[ ]:


#tagging session 2.1
from pyspark.sql import SparkSession
from pyspark.sql.functions import lag, when, isnull, sum, count, lit, avg, stddev
from pyspark.sql.window import Window

session = SparkSession.builder.appName("Manipulating Recommender Dataset with Apache Spark").getOrCreate()
sc = session.sparkContext
sc.setLogLevel("ERROR")

df = sc.textFile("/home/javeed/Desktop/ml-10m/tags.dat").map(lambda x: x.split('::')).map(lambda x: [int(x[0]), int(x[1]), x[2], int(x[3])]).toDF()
df.show()

df = df.select(df._1.alias("UserId"), df._2.alias("MovieId"),
               df._3.alias("Tag"), df._4.alias("TimeStamp"))
df.show()


# In[ ]:


#Tag session 2.2
tagging_session = Window.partitionBy('UserId').orderBy(['UserId', 'TimeStamp'])

df = df.withColumn("lagged", lag(df.TimeStamp).over(tagging_session))
df.show()
df = df.withColumn("SessionTime", when(isnull(df.TimeStamp - df.lagged), 0).otherwise(df.TimeStamp - df.lagged))


df = df.withColumn("sessionTimeOut", when(df.SessionTime > (30 * 60), 1).otherwise(0))
df.show()

tagging_session = Window.partitionBy("UserId").orderBy('TimeStamp')
df = df.withColumn("SessionId", sum(df.sessionTimeOut).over(tagging_session))
df.orderBy('MovieId').show()

df = df.withColumn("SessionId", df.SessionId + lit(1))
df.show()


# In[ ]:


# session ststistics 2.2.1 (frequencies)
session_stats = df.groupBy(['UserId', 'SessionId']).agg(count('SessionId').alias('Frequency'))
session_stats.show()


# In[ ]:


# average and standard deviation of each user 2.2.2
avg_stddev = session_stats.groupBy(['UserId']).agg(avg('Frequency').alias('Average'), stddev('Frequency').alias('Standard_dev'))
avg_stddev.show()


# In[ ]:


# Frequency across the users 2.2.3
f_across_user = session_stats.groupBy().agg(
      avg('Frequency').alias('Average'),
       stddev('Frequency').alias('StdDev')).collect()
print(f_across_user)


# In[ ]:


# mean more than two standard deviation 2.2.4
statistics = session_stats.groupBy().agg(avg('Frequency').alias('Average'), stddev('Frequency').alias('StdDev')).collect()
print(statistics)
avg_stddev.filter(avg_stddev['Average'] > 2 * statistics[0][0] + statistics[0][1]).show()

