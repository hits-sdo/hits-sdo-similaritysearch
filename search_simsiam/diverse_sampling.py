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
    result = [[0,0]]
    distances = [1000000] * data.shape[0]
    for _ in range(n):
        for i in range(features.shape[0]):
            dist = np.linalg.norm(features[i] - result[-1])
            if distances[i] > dist:
                distances[i] = dist
        idx = distances.index(max(distances))
        result.append(features[idx])
        features = np.delete(features, idx, axis=0)
        distances = [1000000] * features.shape[0]

    return result``



def main():
    cov = np.array([[6, -3], [-3, 3.5]])
    # np.random.seed(7099872)
    data = np.random.multivariate_normal([0, 0], cov, size=800)
    data = np.array(data)
    def embedder(x): return x
    r = diverse_sampler(embedder, data, 600)

    # test plotting TODO remove
    # print(len(r))
    # print(r)
    r = np.array(r)
    plt.subplot(1, 2, 1)
    plt.plot(data[:, 0], data[:, 1], '.')
    plt.subplot(1, 2, 2)
    plt.plot(r[:, 0], r[:, 1],'.')

    # plt.plot(r[:, 0], r[:, 1], '.')
    plt.show()


if __name__ == '__main__':
    main()
