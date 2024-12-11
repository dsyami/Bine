import numpy as np
from draw.adjMatrix import plot_heatmaps

class SubjectController:
    
    def draw_adj_by_subjects(self, all_set, all_label, index, methods='Pearson'):
        index = self.extract_numbers_to_int_list(index) - 1
        set, label = all_set[index], all_label[index]
        adj_list = self.adjacency_from_dataset(set, method='Pearson')
        plt = plot_heatmaps(adj_list, label, index)
        return plt
            
    def adjacency_from_dataset(self, dataset, method='Pearson'):
        # dataset [trials, electrodes, sfreq*time]
        trials, channels, features = dataset.shape
        Adjacency_Matrix_list = np.zeros((trials, channels, channels))
        if (method == 'Pearson'):
            for i, trial in enumerate(dataset):
                Adjacency_Matrix = np.empty([channels, channels])
                Adjacency_Matrix = np.abs(np.corrcoef(trial)) - np.eye(channels)
                Adjacency_Matrix_list[i, :, :] = Adjacency_Matrix
        return Adjacency_Matrix_list

    def extract_numbers_to_int_list(self, strings):
        import re
        return int(re.search(r'\d+$', strings).group())
            