'''
features ← embedder(data)
result ← [random.choice(data)]
distances ← [] ∗ len(data)
while n != 0 do
    i ← 0
    while i < len(features) do
        dist ← dist(features[i], result[−1])
        if distances[i] > dist then
            distances[i] = dist
        end if
    end while
    idx ← distances.index(max(distances))
    result.append(data[idx])
    delfeatures[idx]
    n ← n − 1
end while
return result
'''


import numpy as np
import matplotlib.pyplot as plt
import random


def diverse_sampler(embedder, data, n):
    """
    https://arxiv.org/pdf/2107.03227.pdf

    Parameters:
        embedder (function): embeds a high dimensional vector to a lower dim
        data (list): data to embed
        n (int): number of points to sample from the embedding space

    Returns:
        result (list): list of n points sampled from the embedding space
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

    return result[:n]



def main():


    def embedder(x): return x


    numb_points_per_circle = []
    for i in range(10):
        if i % 2 != 1:
            numb_points_per_circle.append(100)
        else:
            numb_points_per_circle.append(800)
        
    points = []
    for i in range(10):
        radius = 0.05 * (i + 1)

        theta = np.random.uniform(0, 2 * np.pi, numb_points_per_circle[i])

        x = 0.5 + radius * np.cos(theta)
        y = 0.5 + radius * np.sin(theta)

        circle_points = np.column_stack((x, y))
        points.append(circle_points)
    
    points_array = np.vstack(points)

    x = points_array[:, 0]
    y = points_array[:, 1]


    plt.subplot(1, 3, 1)
    plt.scatter(x, y)
    plt.title("Original Biased Circles Dataset")
    plt.xlabel("X")
    plt.ylabel("Y")


    np.random.shuffle(points_array)
    random_subset_points_array = points_array[:500]
    x = random_subset_points_array[:, 0]
    y = random_subset_points_array[:, 1]
    plt.subplot(1, 3, 2)
    plt.scatter(x, y)
    plt.title("Random Sampling")
    plt.xlabel("X")
    plt.ylabel("Y")


    points_array = np.array(points_array)
    r = diverse_sampler(embedder, points_array, 500)
    r = np.array(r)
    x = r[:, 0]
    y = r[:, 1]

    plt.subplot(1, 3, 3)
    plt.scatter(x, y)
    plt.title("Diverse Sampling")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()



# Please use this 🥺 instead of that mess ^
# points_array = np.vstack(points)
# random_subset_points_array = points_array[:500]
# def embedder(x): return x
# r = diverse_sampling(embedder, points_array, 500)


# title = ['Original Biased Circles Dataset', 'Random Sampling', 'Diverse Sampling']
# ar = [points_array, random_subset_points_array, r]

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# for t, a, ax in zip(title, ar, axs):
#     ax.scatter(a[:, 0], a[:, 1], s=1)
#     ax.set_title(t)
# plt.show()



    # cov = np.array([[6, -3], [-3, 3.5]])
    # np.random.seed(7099872)
    # data = np.random.multivariate_normal([0, 0], cov, size=800)
    # data = np.array(data)
    
    # r = diverse_sampler(embedder, data, 100)

    # # test plotting TODO remove
    # # print(len(r))
    # # print(r)
    # r = np.array(r)
    # plt.subplot(1, 2, 1)
    # plt.plot(data[:, 0], data[:, 1], '.')
    # plt.subplot(1, 2, 2)
    # plt.plot(r[:, 0], r[:, 1],'.')

    # plt.show()


if __name__ == '__main__':
    main()
