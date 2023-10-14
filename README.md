# hits-sdo-similaritysearch

To run:
1. Create conda environment from environment yml
2. Start environment
3. Change path in filepath.yml to the directory this repository is located (without repository name)
4. Dowload the following datasets to root_path location:
    - https://drive.google.com/uc?id=15C5spf1la7L09kvWXll2qt67Ec0rwLsY
    - https://drive.google.com/uc?id=1DMIatOmA4XcoWeW0oAUkZujx8YrhLkpY 
    - rename folder to match pattern AIA'wavelength'_Miniset
5. Download the following pickle and checkpoint files:
    - https://drive.google.com/drive/u/2/folders/1BZu8EMoV0svUUYLFc23ISmto1YogRUSC
6. Move the pickle files to hits-sdo-similaritysearch/search_simsiam/embeddings_dicts
7. Move the checkpoint files to hits-sdo-similaritysearch/search_simsiam/saved_model
8. run with streamlit run hits-sdo-similaritysearch/search_simsiam/simsearch_gui.py