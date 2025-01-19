from data import get_train_loaders, download_and_preprocess
from model import get_model
from pathlib import Path
import typer
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sweep_config = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'lr': {
            'values': [0.01, 0.001, 0.0001]
        },
        'batch_size': {
            'values': [32, 64]
        }
    }
}

def train(model, train_loader, val_loader, loss_fn, optimizer):
    """
    Train the model on Fashion MNIST.
    """
    model.train()
    train_loss = 0
    for i, (img, label) in enumerate(train_loader):
        img, label = img.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(img)
        loss = loss_fn(y_pred, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
        if i % 100 == 0:
            wandb.log({'iter': i, 'loss': loss.item()})

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for img, label in val_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            y_pred = model(img)
            val_loss += loss_fn(y_pred, label).item()
    val_loss /= len(val_loader)
    train_loss /= len(train_loader)
    return train_loss, val_loss

def train_sweep(config: DictConfig):
    config_dict = OmegaConf.to_container(config, resolve=True)

    with wandb.init(
        project="sweep_project",
        config=config_dict 
    ) as run:  
        
        wb_config = wandb.config  

        run.name = f"sweep_lr_{wb_config.lr}_batch_{wb_config.batch_size}"

        model = get_model(config)
        model.to(DEVICE)
        train_loader, val_loader = get_train_loaders(config.paths.processed_dir, wb_config.batch_size)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=wb_config.lr)

        for epoch in range(config.experiment.epochs):
            wandb.log({'epoch': epoch})  
            train_loss, val_loss = train(model, train_loader, val_loader, loss_fn, optimizer)
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss})


@hydra.main(config_path="../../configs", config_name="config", version_base="1.1")
def hydra_train(config: DictConfig):
    download_and_preprocess(config)
    sweep_id = wandb.sweep(sweep_config, project='sweep_project')
    wandb.agent(sweep_id, function=lambda: train_sweep(config))

def main():
    hydra_train()

if __name__ == "__main__":
    typer.run(main)
