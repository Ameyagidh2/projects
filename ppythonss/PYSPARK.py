from pyspark import SparkConf, SparkContext
import pandas as pd
sc = SparkContext(master="local",appName="Spark Demo")
print(sc.textFile("A:\\Big Data\\deckofcards.txt").first())