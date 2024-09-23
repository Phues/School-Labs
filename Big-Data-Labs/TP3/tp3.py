import pyspark

from pyspark.sql import DataFrame, SparkSession
from pyspark import SparkConf, SparkContext
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F

spark= SparkSession \
       .builder \
       .appName("TP3") \
       .getOrCreate()

file = "hdfs:///user/root/ngram.csv"
df = spark.read.csv(file, header=True, inferSchema=True, sep='\t')

df.printSchema()
print(df.columns)


df = df.withColumnRenamed('! $17.95', 'Ngram')
df = df.withColumnRenamed('1985', 'Year')
df = df.withColumnRenamed('12', 'Count')
df = df.withColumnRenamed('13', 'Pages')
df = df.withColumnRenamed('14', 'Books')

# Print the updated schema and columns
df.printSchema()
print(df.columns)

print("##############################################################################################")
print()
print("Printing the table:")

print("##############################################################################################")
print()
df.createOrReplaceTempView("ngram")
spark.sql("SELECT * FROM ngram").show()

print("##############################################################################################")
print("3.1) Retourner tous les bi-grammes dont le nombre Count est superieur a cinq.")

# Spark API
df.where(df['Count'] > 5).show()

print("##############################################################################################")
print("3.2) Retourner le nombre total de bi-grammes dans chaque annee.")


# Spark API
df.groupBy("Year").agg({"Count": "sum"}).show()

print("##############################################################################################")
print("3.3) Retourner les bi-grammes qui ont le plus grand nombre de count dans chaque annee.")

# Spark API
df.groupBy("Year", "ngram").agg({"Count": "max"}).show()

print("##############################################################################################")
print("3.4) Retourner tous les bi-grammes qui sont apparus dans 20 annees differentes.")

df.groupBy("Ngram").count().filter("count >= 20").show()

print("##############################################################################################")
print("3.5) Retourner tous les bi-grammes qui contiennent le caractere '!' dans la premiere partie et le caractere '9' dans la deuxieme partie (les deux parties sont separees parun espace).")

# Spark API
df.filter(df['Ngram'].like('!% 09%')).show()


print("##############################################################################################")
print("3.6) Retourner les bi-grammes qui sont apparus dans toutes les annees presentes dans les donnees.")

distinct_year_count = df.select("Year").distinct().count()

df.groupBy("Ngram") \
                    .agg(F.countDistinct("Year").alias("DistinctYearCount")) \
                    .filter(F.col("DistinctYearCount") == distinct_year_count) \
                    .select("Ngram").show()

print("##############################################################################################")
print("3.7) Retourner le nombre total de pages et de livres dans lesquels chaque bi-gramme apparait pour chaque annee disponible, trie par ordre alphabetique.")

df.groupBy("Year", "Ngram") \
  .agg(F.sum("Pages").alias("TotalPages"), F.sum("Books").alias("TotalBooks")) \
  .orderBy("Ngram") \
  .show()

print("##############################################################################################")
print("3.8) Retourner le nombre total de bi-grammes differents dans chaque annee, tries par ordre decroissant de l'annee.")

df.groupBy("Year", "Ngram") \
  .agg(F.countDistinct("Ngram").alias("DistinctNgramCount")) \
  .groupBy("Year") \
  .agg(F.sum("DistinctNgramCount").alias("TotalDistinctNgrams")) \
  .sort("Year", ascending=False) \
  .show()

