from pyspark import SparkConf, SparkContext

def parse_line(line):
    parts = line.split(';')
    try:
        height = float(parts[6])
        return height
    except ValueError:
        return None

app = "tp2_2"
conf = SparkConf().setAppName(app)
sc = SparkContext(conf=conf)

brut = sc.textFile("hdfs:///user/root/arbres.csv")

heights = brut.map(parse_line)
print(heights)
heights_ok = heights.filter(lambda x: x)
total = heights_ok.sum()
count = heights_ok.count()
average = total / count if count != 0 else None
print("Average height of tree is:", average)

