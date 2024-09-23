from pyspark import SparkConf, SparkContext

app = "TP2_3"
conf = SparkConf().setAppName(app)
sc = SparkContext(conf=conf)

def parse_line(line):
    parts = line.split(';')
    try:
        height = float(parts[6])
        genre = parts[2].strip()
        return (height, genre)
    except (ValueError, IndexError):
        return (None, None)

brut = sc.textFile("hdfs:///user/root/arbres.csv")
hauteurs_genres = brut.map(parse_line)

hauteurs_genres_ok = hauteurs_genres.filter(lambda x: x[0] is not None and x[1] is not None)

classement = hauteurs_genres_ok.sortByKey(ascending=False)

grand = classement.first()
print("Plus grand arbre:", grand)

