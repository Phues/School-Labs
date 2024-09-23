from pyspark import SparkConf, SparkContext

appName = "count_lines"
conf = SparkConf().setAppName(appName)
sc = SparkContext(conf=conf)



file = sc.textFile("hdfs:///user/root/arbres.csv")

nombre = file.count()
print("number of line is:", nombre)
