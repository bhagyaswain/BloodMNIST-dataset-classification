import sys
outputFilePath = "Q1_output.txt"
sys.stdout = open(outputFilePath, "w")

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, desc
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Q1") \
    .master("local[4]") \
    .config("spark.local.dir", os.environ.get('TMPDIR', '/tmp')) \
    .getOrCreate()
    
# Set log level
spark.sparkContext.setLogLevel("WARN")

# Load data
data_path = "/users/acp22abj/com6012/acp22abj-COM6012/Data/NASA_access_log_Jul95.gz"
logDataFrame = spark.read.text(data_path)

# Define the regular expression pattern to extract information
logRegex = '^(\S+) - - \[(\d{2})/(\w{3})/(\d{4}):(\d{2}):\d{2}:\d{2}'

# Extract information from the log data
logDataFrame = logDataFrame.withColumn('host', regexp_extract('value', logRegex, 1))\
                            .withColumn('day', regexp_extract('value', logRegex, 2).cast('int'))\
                            .withColumn('month', regexp_extract('value', logRegex, 3))\
                            .withColumn('year', regexp_extract('value', logRegex, 4).cast('int'))\
                            .withColumn('hour', regexp_extract('value', logRegex, 5).cast('int'))

# Count requests from each country
requestsDE = logDataFrame.filter(logDataFrame['host'].endswith(".de")).count()
requestsCA = logDataFrame.filter(logDataFrame['host'].endswith(".ca")).count()
requestsSG = logDataFrame.filter(logDataFrame['host'].endswith(".sg")).count()
print(f"Number of requests from Germany: {requestsDE}")
print(f"Number of requests from Canada: {requestsCA}")
print(f"Number of requests from Singapore: {requestsSG}")

#Plot the number of requests from each country
countries = ['Germany', 'Canada', 'Singapore']
requests = [requestsDE, requestsCA, requestsSG]
plt.bar(countries, requests, color='cyan')
plt.xlabel('Country', fontweight='bold')
plt.ylabel('Number of Requests', fontweight='bold')
plt.title('Number of Requests from Each Country')
plt.tight_layout()
plt.show()
#add value on the top of each bar
for i in range(len(countries)):
    plt.text(i, requests[i], requests[i], ha = 'center')
plt.savefig('Q1_figA.jpg')

# Get the top 9 most frequent hosts for each country
top_hosts_germany = logDataFrame.filter(logDataFrame['host'].endswith(".de")) \
                               .groupBy("host").count().orderBy(desc("count")).limit(9).collect()
top_hosts_canada = logDataFrame.filter(logDataFrame['host'].endswith(".ca")) \
                              .groupBy("host").count().orderBy(desc("count")).limit(9).collect()
top_hosts_singapore = logDataFrame.filter(logDataFrame['host'].endswith(".sg")) \
                                 .groupBy("host").count().orderBy(desc("count")).limit(9).collect()

print("Top 9 most frequent hosts in Germany:")
for host in top_hosts_germany:
    print(host['host'], host['count'])

print("Top 9 most frequent hosts in Canada:")
for host in top_hosts_canada:
    print(host['host'], host['count'])

print("Top 9 most frequent hosts in Singapore:")
for host in top_hosts_singapore:
    print(host['host'], host['count'])

# Get the number of requests for each of the top 9 most frequent hosts for each country
requests_top_hosts_germany = [host['count'] for host in top_hosts_germany]
requests_top_hosts_canada = [host['count'] for host in top_hosts_canada]
requests_top_hosts_singapore = [host['count'] for host in top_hosts_singapore]
print("Number of requests for each of the top 9 most frequent hosts in Germany:")
print(requests_top_hosts_germany)
print("Number of requests for each of the top 9 most frequent hosts in Canada:")
print(requests_top_hosts_canada)
print("Number of requests for each of the top 9 most frequent hosts in Singapore:")
print(requests_top_hosts_singapore)

# Get the top 9 most frequent hosts for each country
top_hosts_germany = [host['host'] for host in top_hosts_germany]
top_hosts_canada = [host['host'] for host in top_hosts_canada]
top_hosts_singapore = [host['host'] for host in top_hosts_singapore]
print ("Top 9 most frequent hosts in Germany:")
print(top_hosts_germany)
print ("Top 9 most frequent hosts in Canada:")
print(top_hosts_canada)
print ("Top 9 most frequent hosts in Singapore:")
print(top_hosts_singapore)

# Create a bar chart for Germany
fig, ax = plt.subplots()
barWidth = 0.3
r1 = np.arange(9)
plt.bar(r1, requests_top_hosts_germany, color='r', width=barWidth, edgecolor='grey', label='Top 9 hosts')
plt.xlabel('Hosts', fontweight='bold')
plt.xticks([r for r in range(9)], top_hosts_germany, rotation=45, ha='right')
plt.ylabel('Number of requests', fontweight='bold')
plt.title('Number of requests by each host in Germany')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Q1_figC1.jpg')

# Create a bar chart for Canada
fig, ax = plt.subplots()
plt.bar(r1, requests_top_hosts_canada, color='r', width=barWidth, edgecolor='grey', label='Top 9 hosts')
plt.xlabel('Hosts', fontweight='bold')
plt.xticks([r for r in range(9)], top_hosts_canada, rotation=45, ha='right')
plt.ylabel('Number of requests', fontweight='bold')
plt.title('Number of requests by each host in Canada')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Q1_figC2.jpg')

# Create a bar chart for Singapore
fig, ax = plt.subplots()
plt.bar(r1, requests_top_hosts_singapore, color='r', width=barWidth, edgecolor='grey', label='Top 9 hosts')
plt.xlabel('Hosts', fontweight='bold')
plt.xticks([r for r in range(9)], top_hosts_singapore, rotation=45, ha='right')
plt.ylabel('Number of requests', fontweight='bold')
plt.title('Number of requests by each host in Singapore')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Q1_figC3.jpg')

# Get the most frequent host for each country
most_frequent_host_germany = logDataFrame.filter(logDataFrame['host'].endswith(".de")) \
                                        .groupBy("host").count().orderBy(desc("count")).limit(1).collect()[0]['host']
most_frequent_host_canada = logDataFrame.filter(logDataFrame['host'].endswith(".ca")) \
                                        .groupBy("host").count().orderBy(desc("count")).limit(1).collect()[0]['host']
most_frequent_host_singapore = logDataFrame.filter(logDataFrame['host'].endswith(".sg")) \
                                        .groupBy("host").count().orderBy(desc("count")).limit(1).collect()[0]['host']

print("Most frequent host in Germany:", most_frequent_host_germany)
print("Most frequent host in Canada:", most_frequent_host_canada)
print("Most frequent host in Singapore:", most_frequent_host_singapore)

# Get the number of visits for the most frequent host for each country
visits_most_frequent_host_germany = logDataFrame.filter(logDataFrame['host'] == most_frequent_host_germany) \
                                                .groupBy("day", "hour").count().collect()
visits_most_frequent_host_canada = logDataFrame.filter(logDataFrame['host'] == most_frequent_host_canada) \
                                                .groupBy("day", "hour").count().collect()
visits_most_frequent_host_singapore = logDataFrame.filter(logDataFrame['host'] == most_frequent_host_singapore) \
                                                .groupBy("day", "hour").count().collect()

print("Number of visits for the most frequent host in Germany:")
print(visits_most_frequent_host_germany)
print("Number of visits for the most frequent host in Canada:")
print(visits_most_frequent_host_canada)
print("Number of visits for the most frequent host in Singapore:")
print(visits_most_frequent_host_singapore)

# Get the days and hours for the heatmap plot
days = list(range(1, 32))  # Assuming days range from 1 to 31
hours = list(range(24))  # 24 hours in a day

# Get the number of visits for the heatmap plot
visits_germany = np.zeros((len(days), len(hours)))
visits_canada = np.zeros((len(days), len(hours)))
visits_singapore = np.zeros((len(days), len(hours)))

for visit in visits_most_frequent_host_germany:
    visits_germany[visit['day'] - 1, visit['hour']] = visit['count']

for visit in visits_most_frequent_host_canada:
    visits_canada[visit['day'] - 1, visit['hour']] = visit['count']

for visit in visits_most_frequent_host_singapore:
    visits_singapore[visit['day'] - 1, visit['hour']] = visit['count']

# Create heatmap plots
plt.figure(figsize=(10, 6))

# Germany heatmap
plt.imshow(visits_germany, cmap='viridis', interpolation='nearest', aspect='auto', extent=[0, 24, 1, 32])
plt.colorbar(label='Number of Visits')
plt.xlabel('Hour')
plt.ylabel('Day')
plt.title('Germany')
plt.xticks(np.arange(0, 25, 1), rotation=45, ha='right')
plt.yticks(np.arange(1, 33, 1))
plt.savefig('Q1_figD1.jpg')
plt.show()

# Canada heatmap
plt.figure(figsize=(10, 6))
plt.imshow(visits_canada, cmap='viridis', interpolation='nearest', aspect='auto', extent=[0, 24, 1, 32])
plt.colorbar(label='Number of Visits')
plt.xlabel('Hour')
plt.ylabel('Day')
plt.title('Canada')
plt.xticks(np.arange(0, 25, 1), rotation=45, ha='right')
plt.yticks(np.arange(1, 33, 1))
plt.gca().invert_yaxis()
plt.savefig('Q1_figD2.jpg')
plt.show()

# Singapore heatmap
plt.figure(figsize=(10, 6))
plt.imshow(visits_singapore, cmap='viridis', interpolation='nearest', aspect='auto', extent=[0, 24, 1, 32])
plt.colorbar(label='Number of Visits')
plt.xlabel('Hour')
plt.ylabel('Day')
plt.title('Singapore')
plt.xticks(np.arange(0, 25, 1), rotation=45, ha='right')
plt.yticks(np.arange(1, 33, 1))
plt.gca().invert_yaxis()
plt.savefig('Q1_figD3.jpg)
plt.show()

# Stop Spark session
spark.stop()

sys.stdout.close()

