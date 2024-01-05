import json
import pickle
import torchvision
from lightly.data import LightlyDataset
from dataset import HMItilesDataset

from embedding_utils import EmbeddingUtils
from model import load_model

# path = '/home/schatterjee/Documents/hits/aia_171_color_1perMonth'
# model_path = '/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/saved_model/epoch=8-step=14769.ckpt'
# path = '/d0/euv/aia/preprocessed/HMI/HMI_256x256/'
path = '/d0/euv/aia/preprocessed/AIA_211_193_171/AIA_211_193_171_256x256/'
model_path = '/d0/subhamoy/models/search/AIA_211_193_171/Subh_SIMSIAM_ftrs_512_pretrained_False_projdim_512_preddim_128_odim_512_contrastive_False_AIA_211_193_171_patch_stride_1_batch_64_optim_sgd_lr_0.0125_schedule_coeff_1.0_offlimb_frac_1.ckpt'
#'/d0/subhamoy/models/search/magnetograms/Subh_SIMSIAM_Magnetic_patch_stride_1_batch_64_lr_0.0125.ckpt'
#input_size = 128
# transforms = torchvision.transforms.Compose([
#         #torchvision.transforms.Resize((input_size, input_size)),
#         torchvision.transforms.ToTensor()]
# )

# dataset_val_simsiam = LightlyDataset(input_dir=path, transform=transforms)
dataset_val_simsiam = HMItilesDataset(data_path=path, batch_size=256,
                                      augmentation=None, instr='euv',
                                      data_stride=1)
E = EmbeddingUtils(dataset=dataset_val_simsiam,
                   model=load_model(model_path),
                   batch_size=256,
                   num_workers=8,
                   projection=False,
                   prediction=False)

filenames, embeddings = E.embedder()

dct = {'filenames':filenames, 'embeddings': embeddings}
print(len(filenames))
print(embeddings.shape)

embeddings = {fname: embed for fname, embed in zip(filenames, embeddings.tolist())}

# with open("embeddings_dict.json", "w") as out_file:
#     json.dump(embeddings, out_file)

# with open("/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/embeddings_dict.p", "wb") as out_file:
#     pickle.dump(embeddings, out_file)

# with open("/d0/subhamoy/models/search/magnetograms/embeddings_dict_mag.p", "wb") as out_file:
#     pickle.dump(dct, out_file)
with open("/d0/subhamoy/models/search/magnetograms/embeddings_dict_AIA_211_193_171.p", "wb") as out_file:
    pickle.dump(dct, out_file)