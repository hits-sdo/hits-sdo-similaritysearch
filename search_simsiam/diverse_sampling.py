import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle

def diverse_sampler(embedder, data, n):
    """
    Parameters:
        embedder (function): embeds a high dimensional vector to a lower dim
        data (list): data to embed
        n (int): number of points to sample from the embedding space

    Returns:
        result (list): list of n points sampled from the embedding space

    Ref:
        https://arxiv.org/pdf/2107.03227.pdf

    """
    features = embedder(data)
    result = [random.choice(data)]
    distances = [1000000] * data.shape[0]
    for _ in range(n):
        for i in range(features.shape[0]):
            dist = np.linalg.norm(features[i] - result[-1])
            if distances[i] > dist:
                distances[i] = dist
        idx = distances.index(max(distances))
        result.append(features[idx])
        # del data[idx]
        # result.append(features[idx])
        features = np.delete(features, idx, axis=0)
        del distances[idx]
        # distances = [1000000] * features.shape[0]

    return np.array(result[1:])


def circles_dataset():
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
    def embedder(x): return x
    points_array, colors = circles_dataset()
    points_array, colors = shuffle(points_array, colors)
    random_subset_points_array = points_array[:500]
    colors_random = colors[:500]
    diverse_subset_points_array = diverse_sampler(embedder, points_array, 500)
    colors_diverse = [colors[np.where(np.all(points_array == i,
                                             axis=1))[0][0]]
                      for i in list(diverse_subset_points_array)]

    title = ['Circles Dataset', 'Random Subset', 'Diverse Subset']
    ar = [points_array,
          random_subset_points_array,
          diverse_subset_points_array]
    c_ = [colors, colors_random, colors_diverse]

    _, axs = plt.subplots(1, 3, figsize=(9, 3))
    for t, a, c, ax in zip(title, ar, c_, axs):
        ax.scatter(a[:, 0], a[:, 1], s=10, c=c)
        ax.set_title(t)
        ax.axis('off')
    plt.savefig('/home/schatterjee/Desktop/diverse_sampling.png')
    plt.show()


if __name__ == '__main__':
    main()
