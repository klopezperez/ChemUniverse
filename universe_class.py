from iSIM.iSIM import calculate_isim, calculate_comp_sim
from iSIM.utils import binary_fps
from iSIM.bitbirch import BitBirch, set_isim, mol_set_sim
import numpy as np
import pandas as pd

# Universe class composed of galaxies
class Universe:
    def __init__(self, data, fingerprint_type='RDKIT', n_bits=2048, n_ary='JT'):
        """
        data: str  - path to the csv file containing the smiles and release columns
        fingerprint_type: str - type of fingerprint to use, default is 'RDKIT' others are 'MACCS', 'ECFP4' and 'ECFP6'
        n_bits: int - number of bits for the fingerprint, default is 2048
        n_ary: str - type of n-ary to use, default is 'JT'"""

        # Read the dataframe, check if it has the right columns and types. Columns should contain smiles and release
        self.data = pd.read_csv(data)
        self.data_check()
        self.name = data.split('.')[0].split('/')[-1]

        # Initialize other attributes
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
    
    def calc_release_names(self):
        names = self.data['release'].unique()
        names = np.sort(names)
        self.names = [int(name) for name in names]

    def get_release_names(self):
        if self.names is None: self.calc_release_names()
        return self.names
    
    def calc_universe_fingerprints(self):
        # Get the fingerprints of all the molecules in the universe
        self.fingerprints = binary_fps(self.data['smiles'], self.fingerprint_type, n_bits=self.n_bits)

    def save_universe_fingerprints(self):
        # Save the fingerprints of the universe as a numpy array
        if self.fingerprints is None: self.calc_universe_fingerprints()
        return np.save(self.name + '.npy', self.fingerprints)
    
    def set_universe_fingerprints(self, fingerprints):
        # Set the fingerprints of the universe as a numpy array, load from previously computed fingerprints
        fingerprints = np.load(fingerprints, mmap_mode='r')
        self.fingerprints = fingerprints

    def get_universe_fingerprints(self):
        if self.fingerprints is None: self.calc_universe_fingerprints()
        return self.fingerprints
    
    def universe_analysis(self, save=True):
        # Check if fingerprints are already computed
        if self.fingerprints is None: self.calc_universe_fingerprints()

        # Calculate the names of the releases
        self.calc_release_names()

        # Calculate the sizes of the releases
        release_sizes = self.get_release_sizes()
        
        # Perform the analysis of the universe for each of the releases
        output_data = []
        for i, release_name in enumerate(self.names):
            release = Release(release_name)
            release.release_analysis(self.fingerprints[:release_sizes[i]], n_ary=self.n_ary)
            self.releases.append(release)   

            # Add the release data to the output dataframe
            output_data.append([i + 1, release.size, release.iSIM, release.medoids, release.outliers, release.iSIM_outliers, release.iSIM_medoids, release.c_sum_outliers, release.c_sum_medoids])
            
        output_data = pd.DataFrame(output_data, columns=['release', 'size', 'iSIM', 'medoids', 'outliers', 'iSIM_outliers', 'iSIM_medoids', 'c_sum_outliers', 'c_sum_medoids'])
        
        print('Analysis of universe', self.name, 'completed')
        if save:
            output_data.to_csv(self.name + '_analysis.csv', index=False)
            print('Output saved as', self.name + '_analysis.csv')

        return output_data

    def universe_clustering(self, n=10, threshold=0.65, save_csv=True, return_clusters=False):
        if self.fingerprints is None: self.calc_universe_fingerprints()
        
        output_data = []
        clusters = {}
        for i, release_size in enumerate(self.get_release_sizes()):
            release = Release(i + 1)

            if return_clusters:
                clusters[i + 1] = release.release_clustering(self.fingerprints[:release_size], n=n, threshold=threshold, n_ary=self.n_ary, return_clusters=return_clusters)
                self.releases.append(release) 
            else:
                release.release_clustering(self.fingerprints[:release_size], n=n, threshold=threshold, n_ary=self.n_ary)
                self.releases.append(release)  

            # Add the release data to the output dataframe
            output_data.append([i + 1, release.dense_clusters, release.outlier_clusters, release.avg_pop, release.avg_isim])
            
        output_data = pd.DataFrame(output_data, columns=['release', 'dense_clusters', 'outlier_clusters', 'avg_pop', 'avg_isim'])
        
        print('Clustering of universe', self.name, 'completed')
        if save_csv:
            output_data.to_csv(self.name + '_clustering.csv', index=False)
            print('Output saved as', self.name + '_clustering.csv')

        if return_clusters:
            return clusters
        else:
            return output_data


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
        self.c_sum_outliers = list(np.sum(fingerprints[self.outliers], axis = 0))
        self.c_sum_medoids = list(np.sum(fingerprints[self.medoids], axis = 0))

        print('Analysis of release', self.name, 'completed')

    def release_clustering(self, fingerprints, n=10, threshold=0.65, n_ary='JT', return_clusters=False):
        # Perform clustering of the release
        set_isim(n_ary)
        mol_set_sim(n_ary)
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
        
        if return_clusters:
            return clusters
        
    def get_outliers(self, percentage = 5):
        # Get the indexes of the outliers (mols with highest comp_isim values)
        if self.comp_isim is None: self.calc_comp_isim()
        num_outliers = int(self.size * percentage/100)
        args = np.argsort(self.comp_isim)
        return list(args[-num_outliers:])
    
    def get_medoids(self, percentage = 5):
        # Get the indexes of the medoids (mols with lowest comp_isim values)
        if self.comp_isim is None: self.calc_comp_isim()
        num_medoids = int(self.size * percentage/100)
        args = np.argsort(self.comp_isim)
        return list(args[:num_medoids])
    