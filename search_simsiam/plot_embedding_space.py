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


def get_scatter_plot_with_thumbnails(filenames, embeddings_2d, q, 
                                     nn_filenames, nn_embeddings_2d, title):
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
        #path = os.path.join('/d0/euv/aia/preprocessed/HMI/HMI_256x256/', filenames[idx])
        path = os.path.join('/d0/euv/aia/preprocessed/AIA_211_193_171/AIA_211_193_171_256x256/', filenames[idx])
        img = Image.open(path)
        img = functional.resize(img, thumbnail_size)
        img = np.array(img)
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),
            embeddings_2d[idx],
            pad=0.2,
        )
        ax.add_artist(img_box)
        
       # plot image overlays
    
    # for i, ff in enumerate(nn_filenames):
    #     thumbnail_size = int(rcp["figure.figsize"][0] * 5.0)
    #     # path = os.path.join('/d0/euv/aia/preprocessed/HMI/HMI_256x256/', ff)
    #     path = os.path.join('/d0/euv/aia/preprocessed/AIA_211_193_171/AIA_211_193_171_256x256/', ff)
    #     img = Image.open(path)
    #     img = functional.resize(img, thumbnail_size)
    #     img = np.array(img)
    #     img_box = osb.AnnotationBbox(
    #         osb.OffsetImage(img, cmap=plt.cm.gray_r),
    #         [nn_embeddings_2d[i][0], nn_embeddings_2d[i][1]],
    #         pad=0.2,
    #     )
    #     ax.add_artist(img_box)
        
    # ax.set_xlim([0.9, 1])
    # ax.set_ylim([0.7, 0.8])
    ax.add_artist( Drawing_uncolored_circle )
    ax.set_xticks([])
    ax.set_yticks([])
    # set aspect ratio
    ratio = 1.0 / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable="box")
    return fig

def main():
    # model_path = '/d0/subhamoy/models/search/magnetograms/Subh_SIMSIAM_Magnetic_patch_stride_1_batch_64_lr_0.0125.ckpt'
    model_path = '/d0/subhamoy/models/search/AIA_211_193_171/Subh_SIMSIAM_ftrs_512_pretrained_False_projdim_512_preddim_128_odim_512_contrastive_False_AIA_211_193_171_patch_stride_1_batch_64_optim_sgd_lr_0.0125_schedule_coeff_1.0_offlimb_frac_1.ckpt'
    pl_module = load_model(model_path)
    pl_module.eval()
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # query_image = Image.open('/home/subhamoy/search/latest_4096_HMIBC.jpg')
    query_image = Image.open('/d0/euv/aia/preprocessed/AIA_211_193_171/raw/20230714_074800_aia_211_193_171.jpg')
    tensor = transforms(query_image)
    # tensor = tensor[None, :3, 1500:1756,2400:2656]
    tensor = tensor[None, :3, 1500:1756,1100:1356]
    # plt.imshow(tensor[0,:,:,:].permute(1, 2, 0))
    # plt.savefig('img_crop.png')
    with torch.no_grad():
        embedding = pl_module.backbone(tensor.to('cuda:1')).flatten(start_dim=1)
    query_embedding = embedding[0].cpu().data.numpy()
    print(query_embedding.shape)
    stride = 10
    # p = pickle.load(open('/d0/subhamoy/models/search/magnetograms/embeddings_dict_mag.p','rb'))
    p = pickle.load(open('/d0/subhamoy/models/search/magnetograms/embeddings_dict_AIA_211_193_171.p','rb'))
    nn_filenames = fetch_n_neighbor_filenames(query_embedding,
                                        p,
                                        "EUCLIDEAN",
                                        num_images=4)
    filenames = p['filenames'][::stride]
    embeddings = p['embeddings'][::stride]
    
    nn_embeddings = np.zeros((4, 512))
    
    for i in range(4):
        #ff_image = Image.open('/d0/euv/aia/preprocessed/HMI/HMI_256x256/'+ nn_filenames[i])
        ff_image = Image.open(nn_filenames[i])
        tensor = transforms(ff_image)
        tensor = tensor[None, :3, :, :]
        with torch.no_grad():
            embedding = pl_module.backbone(tensor.to('cuda:1')).flatten(start_dim=1)
        nn_embeddings[i,:] = embedding[0].cpu().data.numpy()
    
    
    
    embeddings_ = np.zeros((len(filenames) + 5,512))
    embeddings_[:len(filenames),:] = embeddings
    embeddings_[len(filenames), :] = query_embedding
    embeddings_[-4:,:] = nn_embeddings
    
    embeddings = embeddings_
    print(embeddings.shape)
    tsne_2d = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne_2d.fit_transform(embeddings)
    M = np.max(embeddings_2d, axis=0)
    m = np.min(embeddings_2d, axis=0)
    embeddings_2d = (embeddings_2d - m) / (M - m)
    query_2d = embeddings_2d[-5]
    nn_embeddings_2d = embeddings_2d[-4:]
    
    print(query_2d)
    print(nn_embeddings_2d)
    get_scatter_plot_with_thumbnails(filenames, embeddings_2d, query_2d,
                                     nn_filenames, nn_embeddings_2d,
                                     'EUV_patch_Emebedding_Space')
    plt.savefig('AIA_211_193_171_embedding_space.png')
    
if __name__ == '__main__':
    main()