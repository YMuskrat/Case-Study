from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

class KNNCLUSTERING:
    def __init__(self,data,col) -> None:
        self.data=data
        self.col=col

    def KNN_clustering(self):
    
        # Prepare the data for clustering
        X_0 = self.data[self.col].values.reshape(-1, 1)
        #X_1 = self.data['in_1'].values.reshape(-1, 1)

        # Perform k-means clustering on in_0
        k = 2  # Number of clusters for in_0
        self.kmeans_0 = KMeans(n_clusters=k, random_state=0)
        self.clusters_0 = self.kmeans_0.fit_predict(X_0)


    
    def get_cluster_profiles(self):
        # Step 2: Calculate cluster profiles
        self.data['Cluster'] = self.clusters_0
        cluster_profiles = self.data.groupby('Cluster').agg({'in_0': ['mean', 'median']})

        # Step 3: Print cluster profiles
        print(cluster_profiles)

        # Step 4: Visualize cluster profiles
        cluster_profiles['in_0'].plot(kind='bar', figsize=(10, 6))
        plt.xlabel('Cluster')
        plt.ylabel('Value')
        plt.title('Cluster Profiles for in_0')
        plt.xticks(rotation=0)
        plt.legend(['Mean', 'Median'])
        plt.show()

    def hypothesis_testing(self):
        # Perform a hypothesis test (e.g., t-test or Mann-Whitney U test) for each pair of clusters
        for i in range(max(self.clusters_0) + 1):
            for j in range(i + 1, max(self.clusters_0) + 1):
                cluster_i_data = self.data[self.clusters_0 == i]['in_0']
                cluster_j_data = self.data[self.clusters_0 == j]['in_0']

                # Perform the hypothesis test
                self.t_statistic, self.p_value = stats.ttest_ind(cluster_i_data, cluster_j_data)

                # Plot the data for each cluster as a scatter plot
                plt.figure(figsize=(8, 6))
                plt.scatter(cluster_i_data.index, cluster_i_data, label='Cluster {}'.format(i))
                plt.scatter(cluster_j_data.index, cluster_j_data, label='Cluster {}'.format(j))
                plt.xlabel('Index')
                plt.ylabel('in_0')
                plt.title('Cluster Comparison: Cluster {} vs Cluster {}'.format(i, j))
                plt.legend()

                # Add p-value and t-statistic to the plot
                plt.text(0.5, 0.9, 'T-Statistic: {:.6f}'.format(self.t_statistic), transform=plt.gca().transAxes)
                plt.text(0.5, 0.85, 'P-Value: {:.6f}'.format(self.p_value), transform=plt.gca().transAxes)

                plt.show()

                # Print the test statistic and p-value for each pair of clusters
                print("Cluster", i, "vs Cluster", j)
                print("T-Statistic:", self.t_statistic)
                print("P-Value:", self.p_value)
                print()
    
    def get_subprocesses(self):
        if self.p_value < 0.0000006:
            # Get the unique cluster labels
            unique_clusters = np.unique(self.clusters_0)

            # Create a dictionary to store the subpopulations
            subpopulations = {}

            # Split the data based on cluster assignments
            for cluster_label in unique_clusters:
                subpopulations[cluster_label] = self.data[self.clusters_0 == cluster_label]  # Replace 'data' with your original dataframe

        return subpopulations
        