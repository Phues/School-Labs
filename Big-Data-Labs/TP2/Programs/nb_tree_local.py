from pyspark import SparkConf, SparkContext

appName = "TP2_4"
conf = SparkConf().setAppName(appName)
sc = SparkContext(conf=conf)

#print current working directory
import os
print(os.getcwd())

brut = sc.textFile("file:///root/arbres.csv")

def parse_genre(line):
    parts = line.split(';')
    try:
        genre = parts[2].strip()
        return genre
    except IndexError:
        return None

genres = brut.map(parse_genre).filter(lambda x: x is not None)
genres_nombres = genres.map(lambda genre: (genre, 1))
genres_total = genres_nombres.reduceByKey(lambda a, b: a + b)

print(genres_total.collect())

