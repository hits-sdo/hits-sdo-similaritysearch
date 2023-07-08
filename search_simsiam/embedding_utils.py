import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
import torch



class EmbeddingUtils:
    def __init__(self, dataset=None, model=None, batch_size=None, num_workers=None):
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers

    def embedder(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        embeddings = []
        filenames = []

        if all(x is not None for x in [self.dataset, self.model, self.batch_size, self.num_workers]):
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            )

            self.model.eval()
            with torch.no_grad():
                # passes batches and filenames to model to find embeddings
                # embedding -> vectorize image, simpler representation of image
                for x, _, fnames in dataloader:
                    # move the images to the gpu
                    # embed the images with the pre-trained backbone
                    y = self.model.backbone(x.to(device)).flatten(start_dim=1)
                    # store the embeddings and filenames in lists
                    embeddings.append(y)
                    filenames = filenames + list(fnames)

            # concatenate the embeddings and convert to numpy
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.cpu().numpy()

        return filenames, embeddings

    def diverse_sampler(self, filenames, features, n):
        """
        Parameters:
            filenames(list): filename
            features (list): embedded data
            n (int): number of points to sample from the embedding space

        Returns:

            result (list): list of n points sampled from the embedding space

        Ref:
            https://arxiv.org/pdf/2107.03227.pdf

        """
        filenames_ = filenames.copy()
        features_ = features.copy()
        result = [random.choice(features_)]
        filenames_results = [random.choice(filenames_)]
        distances = [1000000] * len(features_)
        
        for _ in range(n):
            for i in range(features_.shape[0]):
                dist = np.linalg.norm(features_[i] - result[-1])
                if distances[i] > dist:
                    distances[i] = dist
            idx = distances.index(max(distances))
            result.append(features_[idx])
            filenames_results.append(filenames_[idx])
            
            features_ = np.delete(features_, idx, axis=0)
            del filenames_[idx]
            del distances[idx]

        return filenames_results[1:], np.array(result[1:])



    def circles_test_dataset(self):
        '''
        Returns:
            x, y coordinates of 4500 points creating 10 concentric circles
            thanks to Max Rehm
        '''
        colors = []
        numb_points_per_circle = []
        for i in range(10):
            if i % 2 != 1:
                numb_points_per_circle.append(100)
                colors += ['r']*100
            else:
                numb_points_per_circle.append(800)
                colors += ['g']*800
            
        points = []
        for i in range(10):
            radius = 0.05 * (i + 1)

            theta = np.random.uniform(0, 2 * np.pi, numb_points_per_circle[i])

            x = 0.5 + radius * np.cos(theta)
            y = 0.5 + radius * np.sin(theta)

            circle_points = np.column_stack((x, y))
            points.append(circle_points)
        
        points_array = np.vstack(points)
        return points_array, colors


def main():
    
    testEmbeddingUtils = EmbeddingUtils()
    points_array, colors = testEmbeddingUtils.circles_test_dataset()


    file_name_list = [f'file{i}' for i in range(len(points_array))]

    points_array, colors = shuffle(points_array, colors)
    random_subset_points_array = points_array[:500]
    colors_random = colors[:500]
    # diverse_subset_points_array = diverse_sampler(embedder, points_array, 500)
    diverse_filename, diverse_subset_points_array = testEmbeddingUtils.diverse_sampler(file_name_list, points_array, 500)

    print(len(diverse_filename))
    print(len(diverse_subset_points_array))

    idx = file_name_list.index(diverse_filename[0])
    print(points_array[idx], diverse_subset_points_array[0])


    colors_diverse = [colors[np.where(np.all(points_array == i,
                                             axis=1))[0][0]]
                      for i in list(diverse_subset_points_array)]

    title = ['Circles Dataset', 'Random Subset', 'Diverse Subset']
    ar = [points_array,
          random_subset_points_array,
          diverse_subset_points_array]
    c_ = [colors, colors_random, colors_diverse]

    _, axs = plt.subplots(1, 3, figsize=(20, 6))
    for t, a, c, ax in zip(title, ar, c_, axs):
        ax.scatter(a[:, 0], a[:, 1], s=10, c=c)
        red = c.count('r')
        green = c.count('g')
        ax.set_title(f'{t}\n red:{red} green:{green}')
        ax.axis('off')
    # plt.savefig('/home/schatterjee/Desktop/diverse_sampling.png')
    plt.show()


if __name__ == '__main__':
    main()
