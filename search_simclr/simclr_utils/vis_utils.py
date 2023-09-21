import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import pyprojroot
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
# import matplotlib.offsetbox as offsetbox
from matplotlib import offsetbox
from matplotlib import cm


root = pyprojroot.here()
utils_dir = root/'search_utils'

import sys
sys.path.append(str(root))


def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, _, fnames in tqdm(dataloader):
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)


def plot_knn_examples(embeddings, filenames, path_to_data="root/data", n_neighbors=3, num_examples=6, vis_output_dir=None):
    """Plots multiple rows of random images with their nearest neighbors"""
    print(f'Embeddings.shape: {len(embeddings)}')
    # print("Inside plot_knn_examples() " + path_to_data)
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)
    print(f'len(samples_idx): {len(samples_idx)}')

    # loop through our randomly picked samples
    i = 0
    for idx in samples_idx:
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # print("Inside plot_knn_examples() loop")
            
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # get the correponding filename for the current index
            fname = os.path.join(path_to_data, filenames[neighbor_idx]) # tailor to our path directory structure
            print(f'file name: {fname}')
            # plot the image
            plt.imshow(get_image_as_np_array(fname)[:,:,0], cmap='hot')
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            # let's disable the axis
            plt.axis("off")
        #plt.show()
        i +=1
        plt.savefig(os.path.join(vis_output_dir, f'{i}_knn_plot.png'))
            # Save to file
        # if vis_output_dir is not None:
        #     now = datetime.now()
        #     now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            
        #     plt.savefig(os.path.join(vis_output_dir, f'{now_str}_knn_plot.png'))

def plot_scatter(components_table: pd.DataFrame,
                 vis_output_dir=None,
                 data_dir: str = None,
                 type: str = "TSNE"):
    #create figure
    
    
    columns = (components_table.shape[1] - 1)
    
    if columns == 2:
        # Create scatterplot In 2D
        fig, ax = plt.subplots()

        # Add scatter points
        ax.scatter(components_table['CP_0'], components_table['CP_1'])

        for i in range(len(components_table)):
            path = os.path.join(data_dir, components_table['filename'][i])
            img = Image.open(path)
            # print("Plotting img: "+str(path))
            img.thumbnail((16, 16), Image.ANTIALIAS)  # resizes image in-place
            img = np.array(img)
            print(f'img shape: {img.shape}')
            # img = cm.get_cmap('hot')(img)
            im = OffsetImage(img, zoom=1)
            ab = AnnotationBbox(im, (components_table['CP_0'][i], components_table['CP_1'][i]), frameon=False)

            ax.add_artist(ab)

        ax.set_xlabel('CP_0')
        ax.set_ylabel('CP_1')
        ax.set_title(label=f'{columns} component {type}')
        
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.join(vis_output_dir, f'{now_str}_{type}_plot.png'))
        plt.show()

    else: #3D
        # create scatterplot in 3D
        # fig = plt.figure()
        # ax=fig.add_subplot(111,projection='3d')

        # scatter = ax.scatter(components_table['CP_0'], components_table['CP_1'], components_table['CP_2'], cmap='hot', sizes=[10])
        # ax.set_xlabel('CP_0')
        # ax.set_ylabel('CP_1')
        # ax.set_zlabel('CP_2')
        # ax.set_title(f'{columns} component {type}')
        
        plot_3D(components_table, vis_output_dir, data_dir, type)
    
    





def plot_3D(components_table: pd.DataFrame,
            vis_output_dir = None,
            data_dir: str = None,
            type: str = "TSNE"):
    # xs = [1,1.5,2,2]
    # ys = [1,2,3,1]
    # zs = [0,1,2,0]
    # cmap = ["b","r","g","gold"]
    


    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)
    
    xs = components_table['CP_0'].values.tolist()
    ys = components_table['CP_1'].values.tolist()
    zs = components_table['CP_2'].values.tolist()
    print("3D Component Lengths: "+str(len(xs))+" "+str(len(ys))+" "+str(len(zs)))
    print("Components: "+str(len(components_table))+" "+str(components_table.shape))

    ax.scatter(xs, ys, zs, marker="o", alpha=0)

    # Create a dummy axes to place annotations to
    ax2 = fig.add_subplot(111,frame_on=False) 
    ax2.axis("off")
    ax2.axis([0,1,0,1])

    class ImageAnnotations3D():
        def __init__(self, xyz, imgs, ax3d,ax2d):
            self.xyz = xyz
            self.imgs = imgs
            print("Images Count: "+str(len(imgs)))
            self.ax3d = ax3d
            self.ax2d = ax2d
            self.annot = []
            for s,im in zip(self.xyz, self.imgs):
                x,y = self.proj(s)
                self.annot.append(self.image(im,[x,y]))
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event",self.update)

            self.funcmap = {"button_press_event" : self.ax3d._button_press,
                            "motion_notify_event" : self.ax3d._on_move,
                            "button_release_event" : self.ax3d._button_release}

            self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) \
                            for kind in self.funcmap.keys()]

        def cb(self, event):
            event.inaxes = self.ax3d
            self.funcmap[event.name](event)

        def proj(self, X):
            """ From a 3D point in axes ax1, 
                calculate position in 2D in ax2 """
            x,y,z = X
            x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax3d.get_proj())
            tr = self.ax3d.transData.transform((x2, y2))
            return self.ax2d.transData.inverted().transform(tr)

        def image(self,arr,xy):
            """ Place an image (arr) as annotation at position xy """
            distance = np.sqrt((self.ax3d.get_w_lims()[1] - self.ax3d.get_w_lims()[0]) ** 2 + (self.ax3d.get_proj()[1] - self.ax3d.get_proj()[0]) ** 2)
            new_zoom_level = 150 / distance[0]
            im = offsetbox.OffsetImage(arr, zoom=new_zoom_level)
            im.image.axes = ax
            # ab = offsetbox.AnnotationBbox(im, xy, xybox=(-30., 30.),
            #         xycoords='data', boxcoords="offset points",
            #         pad=0.3, arrowprops=dict(arrowstyle="->"))
            ab = offsetbox.AnnotationBbox(im, xy, xybox=(0,0),
                                xycoords='data', boxcoords="offset points", frameon=False)
            self.ax2d.add_artist(ab)
            return ab

        def update(self, event):
            lims = self.ax3d.get_w_lims()
            proj = self.ax3d.get_proj()
            if np.any(lims != self.lim) or np.any(proj != self.rot):
                self.lim = lims
                self.rot = proj
                distance = np.sqrt((lims[1] - lims[0]) ** 2 + (proj[1] - proj[0]) ** 2)
                zoom = 150 / distance[0]
                for s,ab in zip(self.xyz, self.annot):
                    ab.xy = self.proj(s)
                    im = ab.get_children()[0]  # Get the first child, which is the OffsetImage
                    im.set_zoom(zoom)  # Set the new zoom level

    imgs = []

    # Run a loop to open, resize and convert images into numpy arrays
    for i in range(len(components_table)):
        path = os.path.join(data_dir, components_table['filename'][i])
        img = Image.open(path)
        img.thumbnail((8, 8), Image.ANTIALIAS)  # resizes image in-place
        img = np.array(img)
        imgs.append(img)  # Add to images list
   
    ia = ImageAnnotations3D(np.c_[xs,ys,zs], imgs, ax, ax2 )

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()   
           
# BELOW THIS LINE IS SIMSIAM CODE

# display a scatter plot of the dataset
# clustering similar images together


# FOR TSNE/PCA/UMAP/ETC:
# def get_scatter_plot_with_thumbnails(filenames, embeddings_2d, title):
#     """Creates a scatter plot with image overlays."""
#     # initialize empty figure and add subplot
#     fig = plt.figure()
#     fig.suptitle(title)
#     ax = fig.add_subplot(1, 1, 1)
#     # shuffle images and find out which images to show
#     shown_images_idx = []
#     shown_images = np.array([[1.0, 1.0]])
#     iterator = [i for i in range(embeddings_2d.shape[0])]
#     np.random.shuffle(iterator)
#     for i in iterator:
#         # only show image if it is sufficiently far away from the others
#         dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
#         if np.min(dist) < 2e-4:
#             continue
#         shown_images = np.r_[shown_images, [embeddings_2d[i]]]
#         shown_images_idx.append(i)

#     # plot image overlays
#     for idx in shown_images_idx:
#         thumbnail_size = int(rcp["figure.figsize"][0] * 2.0)
#         path = os.path.join(path_to_data, filenames[idx])
#         img = Image.open(path)
#         img = functional.resize(img, thumbnail_size)
#         img = np.array(img)
#         img_box = osb.AnnotationBbox(
#             osb.OffsetImage(img, cmap=plt.cm.gray_r),
#             embeddings_2d[idx],
#             pad=0.2,
#         )
#         ax.add_artist(img_box)

#     # set aspect ratio
#     ratio = 1.0 / ax.get_data_ratio()
#     ax.set_aspect(ratio, adjustable="box")
#     return fig


def get_image_as_np_array_with_frame(filename: str, w: int = 5):
    """Returns an image as a numpy array with a black frame of width w."""
    img = get_image_as_np_array(filename)
    ny, nx, _ = img.shape
    # create an empty image with padding for the frame
    framed_img = np.zeros((w + ny + w, w + nx + w, 3))
    framed_img = framed_img.astype(np.uint8)
    # put the original image in the middle of the new one
    framed_img[w:-w, w:-w] = img
    return framed_img
   
def plot_nearest_neighbors_3x3(example_image: str, i: int, embeddings, filenames, path_to_data):
    """Plots the example image and its eight nearest neighbors."""
    n_subplots = 9
    # initialize empty figure
    fig = plt.figure()
    fig.suptitle(f"Nearest Neighbor Plot {i + 1}")
    #
    example_idx = filenames.index(example_image)
    # get distances to the cluster center
    distances = embeddings - embeddings[example_idx]
    distances = np.power(distances, 2).sum(-1).squeeze()
    # sort indices by distance to the center
    nearest_neighbors = np.argsort(distances)[:n_subplots]
    # show images
    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        ax = fig.add_subplot(3, 3, plot_offset + 1)
        # get the corresponding filename
        fname = os.path.join(path_to_data, filenames[plot_idx])
        if plot_offset == 0:
            ax.set_title(f"Query Image")
            plt.imshow(get_image_as_np_array_with_frame(fname))
        else:
            plt.imshow(get_image_as_np_array(fname))
        # let's disable the axis
        plt.axis("off")





# ===========================


# def display_search_result(session_state, col2, embeddings_dict, data_path):
#     if 'All' in session_state["indices"]:
#         session_state["indices"] = [x for x in range(session_state['neighbors'])]

#     col2.write('Retrieved Images')
#     idx = embeddings_dict['filenames'].index(session_state['fnames'][0])
#     print('N:', embeddings_dict['embeddings'][idx, :5])

#     dim = math.ceil(math.sqrt(session_state['neighbors']))
#     fig, ax = plt.subplots(dim, dim, figsize=(10, 10))
#     ax = ax.ravel()
#     plt.subplots_adjust(wspace=0.1, hspace=0.1)

#     for a in ax:
#         a.axis('off')

#     for i, f in enumerate(session_state['fnames']):
#         img = cv2.imread(data_path + f)
#         ax[i].imshow(img[:, :, ::-1])

#         ax[i].text(10, 30, i, color='black', fontsize=(10/dim)*10)

#         if i in session_state['indices']:
#             overlay = cv2.rectangle(img, (0, 0), (127, 127), (0, 0, 255), 10)
#             ax[i].imshow(overlay[:, :, ::-1])

#     col2.pyplot(fig)

#     if st.button('Download Selected Images'):
#         with zipfile.ZipFile("selected_images.zip", "w") as zipf:
#             for n in session_state['indices']:
#                 # Add each file to the ZIP archive
#                 zipf.write(data_path+session_state['fnames'][n])

def get_image_as_np_array_with_frame(filename: str, w: int = 5):
    """Returns an image as a numpy array with a black frame of width w."""
    img = get_image_as_np_array(filename)
    ny, nx, _ = img.shape
    # create an empty image with padding for the frame
    framed_img = np.zeros((w + ny + w, w + nx + w, 3))
    framed_img = framed_img.astype(np.uint8)
    # put the original image in the middle of the new one
    framed_img[w:-w, w:-w] = img
    return framed_img