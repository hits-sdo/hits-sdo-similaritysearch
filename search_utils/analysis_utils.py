import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams as rcp
import matplotlib.offsetbox as osb   
import numpy as np
import os
import torch
import torchvision.transforms.functional as functional


def get_scatter_plot_with_thumbnails(embeddings_2d,filenames,root='../'):
    """
    Creates a scatter plot with image overlays and corresponding 2D histogram.
    
    Parameters:
        embeddings_2d (np array):   nx2 array with 2D embeddings for n samples
        filenames (list):           corresponding image file paths 
        root (str):                 root directory for image files

    Returns:
        fig                         figure handle
    """
    
    # initialize empty figure and add subplot
    fig = plt.figure(figsize=[6,3], layout='constrained', dpi=300)
    ax = fig.add_subplot(121)
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1.0, 1.0]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 2e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    # plot image overlays
    for idx in shown_images_idx:
        thumbnail_size = int(rcp["figure.figsize"][0] * 2.0)
        img = np.load(root+filenames[idx])
        img = np.expand_dims(img,0)
        img = functional.resize(torch.Tensor(img), thumbnail_size,antialias=True)
        img = np.array(img)[0,:,:]
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),
            embeddings_2d[idx],
            pad=0.2,
        )
        ax.add_artist(img_box)

    # set aspect ratio
    ratio = 1.0 / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable="box")

    # title
    ax.set_title('Embedding space in 2D')

    # 2D histogram
    ax2 = fig.add_subplot(122)
    histout = np.histogram2d(embeddings_2d[:,0], embeddings_2d[:,1], bins=50)
    ax2.hist2d(embeddings_2d[:,0], embeddings_2d[:,1], bins=100, cmap=plt.cm.magma_r, vmax=np.percentile(histout[0], 99))
    ax2.set_title('Density distribution in 2D')

    return fig

def save_predictions(preds,dir,appendstr:str=''):
    """
    Save predicted files and embeddings
    
    Parameters:
        preds:  output of model predict step (as list of batch predictions)
        dir:    directory for saving
        appendstr: string to save at end of filename
    Returns:
        embeddings (np array):      output of model embed step 
        embeddings_proj (np array): output of model projection head
    """
    file = []
    embeddings = []
    embeddings_proj = []
    for predbatch in preds:
        file.extend(predbatch[0])
        embeddings.extend(np.array(predbatch[1]))
        embeddings_proj.extend(np.array(predbatch[2]))
    embeddings = np.array(embeddings)
    embeddings_proj = np.array(embeddings_proj)

    np.save(dir+os.sep+'embeddings'+appendstr+'.npy',embeddings)
    np.save(dir+os.sep+'embeddings_proj'+appendstr+'.npy',embeddings_proj)
    df = pd.DataFrame({'filename':file})
    df.to_csv(dir+os.sep+'filenames'+appendstr+'.csv',index=False)

    return file, embeddings, embeddings_proj


def load_model(ckpt_path,modelclass,api):
    """
    Load model into wandb run by downloading and initializing weights

    Parameters:
        ckpt_path:  wandb path to download model checkpoint from
        model:      model class
        api:        instance of wandb Api
    Returns:
        model:      Instantiated model class object with loaded weights
    """
    print('Loading model checkpoint from ', ckpt_path)
    artifact = api.artifact(ckpt_path,type='model')
    artifact_dir = artifact.download()
    model = modelclass.load_from_checkpoint(artifact_dir+'/model.ckpt',map_location='cpu')
    return model
