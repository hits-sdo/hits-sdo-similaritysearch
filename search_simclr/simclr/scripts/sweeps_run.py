import wandb
import os.path
import pyprojroot
root = pyprojroot.here()
import sys
sys.path.append(str(root))
import yaml
from search_simclr.simclr.scripts import model_run


def main():
       # main()
    with open("sweeps.yaml") as f:
        sweep_config = yaml.safe_load(f)
        print("Sweep Config: "+ str(sweep_config))
    sweep_id = wandb.sweep(sweep_config, project="search_simclr") 
    wandb.agent(sweep_id, function=model_run.train)
    # wandb.agent(sweep_id, function=main) 

if __name__ == "__main__":
    main()