import json
import pickle
import torchvision
from lightly.data import LightlyDataset

from embedding_utils import EmbeddingUtils
from model import load_model

path = '/home/schatterjee/Documents/hits/AIA211_193_171_Miniset'
model_path = '/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/saved_model/epoch=4-step=435.ckpt'

input_size = 128
transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor()]
)

dataset_val_simsiam = LightlyDataset(input_dir=path, transform=transforms)
E = EmbeddingUtils(dataset=dataset_val_simsiam,
                   model=load_model(model_path),
                   batch_size=64,
                   num_workers=8)

filenames, embeddings = E.embedder()

dct = {'filenames':filenames, 'embeddings': embeddings}
print(len(filenames))
print(embeddings.shape)

embeddings = {fname: embed for fname, embed in zip(filenames, embeddings.tolist())}

# with open("embeddings_dict.json", "w") as out_file:
#     json.dump(embeddings, out_file)

# with open("/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/embeddings_dict.p", "wb") as out_file:
#     pickle.dump(embeddings, out_file)

with open("/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/embeddings_dict_new.p", "wb") as out_file:
    pickle.dump(dct, out_file)