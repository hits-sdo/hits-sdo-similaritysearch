import pickle
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp
import torchvision.transforms.functional as functional
from PIL import Image
import os
import torch
import torchvision
from model import load_model
from gui_utils import fetch_n_neighbor_filenames
import yaml


def get_scatter_plot_with_thumbnails(filenames, embeddings_2d, q, title, config_data):
    """Creates a scatter plot with image overlays."""
    
   
    # initialize empty figure and add subplot
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(1, 1, 1)
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1.0, 1.0]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 5e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    Drawing_uncolored_circle = plt.Circle( (q[0], q[1] ),
                                      0.05 ,
                                      fill = False, color='r', zorder=100 )
    # plot image overlays
    for idx in shown_images_idx:
        thumbnail_size = int(rcp["figure.figsize"][0] * 2.0)
        path = os.path.join(config_data['data_dir'][config_data['instr']], filenames[idx])
        img = Image.open(path)
        img = functional.resize(img, thumbnail_size)
        img = np.array(img)
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),
            embeddings_2d[idx],
            pad=0.2,
        )
        ax.add_artist(img_box)
        
    ax.add_artist( Drawing_uncolored_circle )
    ax.set_xticks([])
    ax.set_yticks([])
    # set aspect ratio
    ratio = 1.0 / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable="box")
    return fig

def main():
    with open('config.yml','r') as f:
        config_data = yaml.safe_load(f)
    instr = config_data['instr']
    model_path = config_data['model_dir'][instr] + config_data['model_name'][instr]
    pl_module = load_model(model_path)
    pl_module.eval()
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    query_image = Image.open(config_data['query_img'][instr])
    ys, xs = config_data['query_roi'][instr]
    
    images = [np.array(query_image), np.array(query_image)[ys:(ys + 256), xs:(xs + 256),:]]
    titles = ['Full-disk','Query RoI']
    
    tensor = transforms(query_image)
    tensor = tensor[None, :3, ys:(ys + 256), xs:(xs + 256)]
    with torch.no_grad():
        embedding = pl_module.backbone(tensor.to('cuda:1')).flatten(start_dim=1)
    query_embedding = embedding[0].cpu().data.numpy()
    print(query_embedding.shape)
    stride = 100
    p = pickle.load(open(f"{config_data['model_dir'][instr]}embeddings_dict_{instr}.p",'rb'))
    nn_filenames = fetch_n_neighbor_filenames(query_embedding,
                                        p,
                                        "EUCLIDEAN",
                                        num_images=4)
    
    
    
    for i,f in enumerate(nn_filenames):
        images.append(np.array(Image.open(config_data['data_dir'][instr] + f))) # 
        n = i+1
        titles.append(f"NN {n}")

 
    fig, axes = plt.subplots(1,6, figsize=(24,4), constrained_layout=True)
    ax = axes.ravel()

    for i, t in enumerate(titles):
        ax[i].imshow(images[i])
        if i==0:
            xs = [xs, xs + 256, xs + 256, xs, xs]
            ys = [ys, ys, ys + 256, ys + 256, ys]
            ax[i].plot(xs, ys, color="red", linewidth=2.0)
        ax[i].set_title(t)
        
    plt.savefig((f"{instr}_nearest_neighbors.png"))
        
    filenames = p['filenames'][::stride]
    embeddings = p['embeddings'][::stride]
    
    embeddings_ = np.zeros((len(filenames) + 1,512))
    embeddings_[:len(filenames),:] = embeddings
    embeddings_[len(filenames), :] = query_embedding

    
    embeddings = embeddings_
    print(embeddings.shape)
    tsne_2d = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne_2d.fit_transform(embeddings)
    M = np.max(embeddings_2d, axis=0)
    m = np.min(embeddings_2d, axis=0)
    embeddings_2d = (embeddings_2d - m) / (M - m)
    query_2d = embeddings_2d[-1]
    embeddings_2d = embeddings_2d[:-1]

    

    get_scatter_plot_with_thumbnails(filenames, embeddings_2d, query_2d,
                                     f"{instr}_patch_Embedding_Space",
                                     config_data)
    plt.savefig(f"{instr}_embedding_space.png")
    
if __name__ == '__main__':
    main()