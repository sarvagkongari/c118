import pandas as pd
import plotly.express as px
df=pd.read_csv("stars.csv")
print(df.head())

fig=px.scatter(df,x="petal_size",y="sepal_size")
fig.show()

from sklearn.cluster import KMeans
X=df.iloc[:,[0,1]].values
print(X)

wcss=[]
for i in range (1,11):
  kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),wcss,marker="o",color="red")
plt.title("The Elbow Method")
plt.xlabel("no of clusters")
plt.ylabel("Wcss")
plt.show()

plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans==0,0],X[y_kmeans==0,1],color="yellow",label="cluster 1")
sns.scatterplot(X[y_kmeans==1,0],X[y_kmeans==1,1],color="cyan",label="cluster 2")
sns.scatterplot(X[y_kmeans==2,0],X[y_kmeans==2,1],color="red",label="cluster 3")
sns.scatterplot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="black",label="centroids",s=100,marker=",")
plt.grid(False)
plt.title('clusters of intesstellar')
plt.xlabel(" size")
plt.ylabel("light")
plt.legend()
plt.show()