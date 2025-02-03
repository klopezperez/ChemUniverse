from iSIM.iSIM import calculate_isim, calculate_comp_sim
from iSIM.utils import binary_fps
from iSIM.bitbirch import BitBirch
import numpy as np
import pandas as pd

# Universe class composed of galaxies
class Universe:
    def __init__(self, data, fingerprint_type='RDKIT', n_bits=2048, n_ary='JT'):
        # Read the smiles release csv, one column should be 'smiles' and another 'release', the release should be an integer sort the df by release
        self.data = pd.read_csv(data)
        self.data_check()
        self.name = data.split('.')[0].split('/')[-1]
        self.fingerprints = None
        self.n_ary = n_ary
        self.fingerprint_type = fingerprint_type
        self.n_bits = n_bits
        self.calc_release_sizes()
        self.releases = []

    def data_check(self):
        # Check if the data has the columns smiles and release
        if 'smiles' not in self.data.columns:
            raise ValueError('The data does not have the column "smiles"')
        if 'release' not in self.data.columns:
            raise ValueError('The data does not have the column "release"')
        if self.data['release'].dtype != 'int64':
            raise ValueError('The data has to have the column "release" as an integer')
        if 'NaN' in self.data['smiles']:
            raise ValueError('The data has NaN values in the column "smiles"')

    def calc_release_sizes(self):
        # Get the sizes of the releases accumulated
        additions = self.data['release'].value_counts().sort_index()
        sizes = [additions[:i].sum() for i in range(1, len(additions))]
        sizes.append(len(self.data))

        self.release_sizes = sizes

    def get_release_sizes(self):
        if self.release_sizes is None: self.calc_release_sizes()
        return self.release_sizes
    
    def calc_universe_fingerprints(self):
        # Get the fingerprints of all the molecules in the universe
        self.fingerprints = binary_fps(self.data['smiles'], self.fingerprint_type, n_bits=self.n_bits)

    def save_universe_fingerprints(self):
        if self.fingerprints is None: self.calc_universe_fingerprints()
        return np.save(self.name + '.npy', self.fingerprints)
    
    def set_universe_fingerprints(self, fingerprints):
        self.fingerprints = fingerprints
    
    def universe_analysis(self):
        if self.fingerprints is None: self.calc_universe_fingerprints()
        
        output_data = []
        for i, release_size in enumerate(self.get_release_sizes()):
            release = Release(i + 1)
            release.release_analysis(self.fingerprints[:release_size], n_ary=self.n_ary)
            self.releases.append(release)   

            # Add the release data to the output dataframe
            output_data.append([i + 1, release.size, release.iSIM, release.medoids, release.outliers, release.iSIM_outliers, release.iSIM_medoids, release.c_sum_outliers, release.c_sum_medoids])
            
        output_data = pd.DataFrame(output_data, columns=['release', 'size', 'iSIM', 'medoids', 'outliers', 'iSIM_outliers', 'iSIM_medoids', 'c_sum_outliers', 'c_sum_medoids'])
        
        output_data.to_csv(self.name + '_analysis.csv', index=False)

    def universe_clustering(self, n=10, threshold=0.65):
        if self.fingerprints is None: self.calc_universe_fingerprints()
        
        output_data = []
        for i, release_size in enumerate(self.get_release_sizes()):
            release = Release(i + 1)
            release.release_clustering(self.fingerprints[:release_size], n=n, threshold=threshold)
            self.releases.append(release)   

            # Add the release data to the output dataframe
            output_data.append([i + 1, release.dense_clusters, release.outlier_clusters, release.avg_pop, release.avg_isim])
            
        output_data = pd.DataFrame(output_data, columns=['release', 'dense_clusters', 'outlier_clusters', 'avg_pop', 'avg_isim'])
        
        output_data.to_csv(self.name + '_clustering.csv', index=False)

        return print('Clustering of universe', self.name, 'completed')


class Release(Universe):
    def __init__(self, name):
        self.name = name
        self.analysis = False

    def release_analysis(self, fingerprints, n_ary='JT'):
        self.size = len(fingerprints)
        self.c_sum = np.sum(fingerprints, axis = 0)
        self.iSIM = calculate_isim(self.c_sum, n_objects=len(fingerprints), n_ary=n_ary)
        self.comp_isim = calculate_comp_sim(fingerprints, n_ary=n_ary)
        self.medoids = self.get_medoids(percentage = 5)
        self.outliers = self.get_outliers(percentage = 5)
        self.iSIM_outliers = calculate_isim(fingerprints[self.outliers], n_objects=len(self.outliers), n_ary=n_ary)
        self.iSIM_medoids = calculate_isim(fingerprints[self.medoids], n_objects=len(self.medoids), n_ary=n_ary)
        self.c_sum_outliers = np.sum(fingerprints[self.outliers], axis = 0)
        self.c_sum_medoids = np.sum(fingerprints[self.medoids], axis = 0)

        print('Analysis of release', self.name, 'completed')

    def release_clustering(self, fingerprints, n=10, threshold=0.65):
        # Perform clustering of the release
        brc = BitBirch(threshold=threshold, branching_factor=50)
        brc.fit(fingerprints)

        clusters = brc.get_cluster_mol_ids()

        # Order the clusters by size
        clusters = sorted(clusters, key=lambda x: len(x), reverse=True)

        # Get the number of clusters with more than n molecules
        clusters_dense = len([cluster for cluster in clusters if len(cluster) > n])
        clusters_outs = len([cluster for cluster in clusters if len(cluster) <= n])

        top_clusters = clusters[:n]

        avg_pop = np.mean([len(cluster) for cluster in top_clusters])

        avg_isim = np.mean([calculate_isim(fingerprints[cluster], n_objects=len(cluster)) for cluster in top_clusters])

        self.dense_clusters = clusters_dense
        self.outlier_clusters = clusters_outs
        self.avg_pop = avg_pop
        self.avg_isim = avg_isim

        print("Clustering of release", self.name, "completed")
        
        
    def get_outliers(self, percentage = 5):
        # Get the indexes of the outliers (mols with highest comp_isim values)
        if self.comp_isim is None: self.calc_comp_isim()
        num_outliers = int(self.size * percentage/100)
        args = np.argsort(self.comp_isim)
        return args[-num_outliers:]
    
    def get_medoids(self, percentage = 5):
        # Get the indexes of the medoids (mols with lowest comp_isim values)
        if self.comp_isim is None: self.calc_comp_isim()
        num_medoids = int(self.size * percentage/100)
        args = np.argsort(self.comp_isim)
        return args[:num_medoids]
    