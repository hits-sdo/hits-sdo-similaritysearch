import json
import pickle
import torchvision
from lightly.data import LightlyDataset
from dataset import AIAHMItilesDataset
import yaml
from embedding_utils import EmbeddingUtils
from model import load_model

def main():
    with open('config.yml','r') as f:
        config_data = yaml.safe_load(f)
    instr = config_data['instr']
    path = config_data['data_dir'][instr]
    model_path = config_data['model_dir'][instr] + config_data['model_name'][instr]

    dataset_val_simsiam = AIAHMItilesDataset(data_path=path, batch_size=256,
                                        augmentation=None, instr=instr,
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

    with open(f"{config_data['model_dir'][instr]}embeddings_dict_{instr}.p", "wb") as out_file:
        pickle.dump(dct, out_file)
        
if __name__=='__main__':
    main()