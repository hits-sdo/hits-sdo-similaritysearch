import glob
import cv2
import numpy as np
import pandas as pd

def main():
    df = {'filename':[],
        'intensity_min':[],
        'intensity_mode':[],
        'intensity_max':[],
        'offlimb_frac_area':[]}

    files = glob.glob('/d0/euv/aia/preprocessed/HMI/HMI_256x256/**/*.jpg',  
                    recursive = True) 

    for f in files:
        img_ = cv2.imread(f)
        h, b = np.histogram(img_[:,:,2].flatten(), bins = 256, range=[-0.5,255.5])
        idx = np.argmax(h)
        mode = (b[idx] + b[idx + 1]) / 2
        frac = np.sum(img_[:, :, 2] == 154)/(256.*256.)
        print(f,"--",'IMODE:',mode,'IMIN:',
                img_[:, :, 2].min(),'IMAX:',
                img_[:, :, 2].max(),'OFF_frac:',
                frac)
        df['filename'].append(f)
        df['intensity_mode'].append(mode)
        df['intensity_min'].append(img_[:, :, 2].min())
        df['intensity_max'].append(img_[:, :, 2].max())
        df['offlimb_frac_area'].append(frac)
        
    df = pd.DataFrame(df)
    df.to_csv('/d0/subhamoy/models/search/magnetograms/indexer.csv')
    
if __name__ == '__main__':
    main()