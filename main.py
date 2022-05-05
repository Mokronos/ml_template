import pandas

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

data = pandas.read_csv(url, names=["test","test2","easd","easd1","easd2"])

print(data.head())
