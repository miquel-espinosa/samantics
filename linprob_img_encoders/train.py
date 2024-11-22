import os
import sys
from typing import Optional, Tuple, Dict, Any
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import timm
from segment_anything import sam_model_registry
from torchmetrics import Accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train or evaluate linear classifiers on frozen vision encoders')
    
    # Add evaluation mode argument
    parser.add_argument('--eval_only', action='store_true',
                        help='Run only evaluation on a checkpoint')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to checkpoint file for evaluation')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to ImageNet dataset')
    parser.add_argument('--sam_checkpoint', type=str, default=None,
                        help='Path to SAM/SAM2 checkpoint (required for SAM encoder)')
    parser.add_argument('--sam2_config', type=str, default=None,
                        help='Path to SAM2 config file (required for SAM2)')
    
    # Training arguments
    parser.add_argument('--encoder', type=str, choices=['clip', 'dinov2', 'sam', 'sam2'], 
                        required=True, help='Encoder type to evaluate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size per GPU')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers per GPU')
    
    # Hardware arguments
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['32', '16-mixed', 'bf16-mixed'],
                        help='Precision for training')
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['cpu', 'gpu', 'auto'],
                        help='Accelerator type')
    parser.add_argument('--devices', type=str, default='auto',
                        help='Number of devices to use (int or "auto")')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='Number of nodes for distributed training')
    parser.add_argument('--strategy', type=str, default='ddp',
                        choices=['ddp', 'ddp_find_unused_parameters_true'],
                        help='Distributed training strategy')
    
    # Optimization arguments
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--grad_clip_val', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Number of batches for gradient accumulation')
    
    # Logging arguments
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment (default: encoder_timestamp)')
    
    # Add checkpoint argument
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    
    return parser.parse_args()

class EncoderLoader:
    @staticmethod
    def load_clip_model() -> Tuple[Optional[nn.Module], int]:
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            return model.visual, 512
        except ImportError:
            print("Error: open_clip not installed. Please install with: pip install open-clip-torch")
            return None, 0

    @staticmethod
    def load_dinov2_model() -> Tuple[Optional[nn.Module], int]:
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            return model, 384
        except Exception as e:
            print(f"Error loading DINOv2: {e}")
            return None, 0

    @staticmethod
    def load_sam_model(checkpoint_path: str) -> Tuple[Optional[nn.Module], int]:
        try:
            model = sam_model_registry['vit_b'](checkpoint=checkpoint_path)
            return model.image_encoder, 64
        except Exception as e:
            print(f"Error loading SAM: {e}")
            return None, 0

    @staticmethod
    def load_sam2_model(checkpoint_path: str, config_path: str) -> Tuple[Optional[nn.Module], int]:
        try:
            from sam2.build_sam import build_sam2
            model = build_sam2(config_path, checkpoint_path)
            return model.image_encoder, 256  # Assuming SAM2 uses same feature dim as SAM
        except Exception as e:
            print(f"Error loading SAM2: {e}")
            return None, 0

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        encoder_name: str,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.encoder_name = encoder_name
        
        if encoder_name in ['sam', 'sam2']:
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage == 'validate' or stage is None:
            self.val_dataset = datasets.ImageNet(
                root=self.data_dir,
                split='val',
                transform=self.transform
            )
            
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.ImageNet(
                root=self.data_dir,
                split='train',
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

class LinearClassifierModule(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str,
        num_classes: int = 1000,
        learning_rate: float = 1e-3,
        sam_checkpoint: Optional[str] = None,
        sam2_config: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Add accuracy metrics
        self.top1_accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.top5_accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        
        # Load encoder
        if encoder_name == 'clip':
            self.encoder, self.feature_dim = EncoderLoader.load_clip_model()
        elif encoder_name == 'dinov2':
            self.encoder, self.feature_dim = EncoderLoader.load_dinov2_model()
        elif encoder_name == 'sam':
            if sam_checkpoint is None:
                raise ValueError("SAM checkpoint path must be provided")
            self.encoder, _ = EncoderLoader.load_sam_model(sam_checkpoint)
            self.feature_dim = 256
        elif encoder_name == 'sam2':
            if sam_checkpoint is None or sam2_config is None:
                raise ValueError("SAM2 checkpoint and config paths must be provided")
            self.encoder, _ = EncoderLoader.load_sam2_model(sam_checkpoint, sam2_config)
            self.feature_dim = 256
        
        if self.encoder is None:
            raise RuntimeError(f"Failed to load {encoder_name} encoder")
        
        self.encoder = self.encoder.cuda().eval()
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.encoder_name = encoder_name

    def forward(self, x):
        with torch.no_grad():
            if self.encoder_name == 'clip':
                features = self.encoder(x.to(self.device))
            elif self.encoder_name == 'dinov2':
                features = self.encoder(x.to(self.device))
            elif self.encoder_name == 'sam':
                features = self.encoder(x.to(self.device))
                features = features.mean(dim=(-2, -1))
            elif self.encoder_name == 'sam2':
                features = self.encoder(x.to(self.device))
                features = features['vision_features']
                features = features.mean(dim=(-2, -1))
        
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate top-1 and top-5 accuracy
        top1_acc = self.top1_accuracy(logits, y)
        top5_acc = self.top5_accuracy(logits, y)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_top1_acc', top1_acc, prog_bar=True, sync_dist=True)
        self.log('val_top5_acc', top5_acc, prog_bar=True, sync_dist=True)
        
        return {'val_loss': loss, 'val_top1_acc': top1_acc, 'val_top5_acc': top5_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

def train_eval_encoder(
    encoder_name: str,
    data_dir: str,
    num_classes: int = 1000,
    batch_size: int = 256,
    max_epochs: int = 10,
    sam_checkpoint: Optional[str] = None,
    sam2_config: Optional[str] = None,
    num_workers: int = 2,
    precision: str = '16-mixed',
    accelerator: str = 'auto',
    devices: str = 'auto',
    num_nodes: int = 1,
    strategy: str = 'ddp',
    learning_rate: float = 1e-3,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, float]:
    # Initialize data module with encoder_name
    dm = ImageNetDataModule(
        data_dir=data_dir,
        encoder_name=encoder_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Initialize model with learning_rate
    model = LinearClassifierModule(
        encoder_name=encoder_name,
        num_classes=num_classes,
        sam_checkpoint=sam_checkpoint,
        sam2_config=sam2_config,
        learning_rate=learning_rate
    )
    
    # Configure callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            filename=f'{encoder_name}-{{epoch:02d}}-{{val_acc:.2f}}',
            save_top_k=1,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            mode='min'
        )
    ]
    
    # Configure logger
    logger = TensorBoardLogger("lightning_logs", name=encoder_name)
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=DDPStrategy(process_group_backend="gloo") if strategy == "ddp" else strategy,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        deterministic=False  # Set to True if you need reproducibility
    )
    
    # Train and evaluate
    trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)
    
    # Return best validation metrics
    return {
        'best_val_acc': trainer.callback_metrics['val_acc'].item(),
        'best_val_loss': trainer.callback_metrics['val_loss'].item()
    }

def evaluate_checkpoint(
    checkpoint_path: str,
    encoder_name: str,
    data_dir: str,
    sam_checkpoint: Optional[str] = None,
    sam2_config: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 2,
    precision: str = '32',
    accelerator: str = 'auto',
    devices: str = 'auto',
) -> Dict[str, float]:
    """
    Evaluate a trained checkpoint on the ImageNet validation set.
    """
    # Initialize data module
    dm = ImageNetDataModule(
        data_dir=data_dir,
        encoder_name=encoder_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Explicitly setup the validation dataset
    dm.setup(stage='validate')
    
    # Load model from checkpoint
    model = LinearClassifierModule.load_from_checkpoint(
        checkpoint_path,
        encoder_name=encoder_name,
        sam_checkpoint=sam_checkpoint,
        sam2_config=sam2_config,
    )
    
    # Initialize trainer for evaluation
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,  # Disable logging for evaluation
    )
    
    # Run evaluation
    results = trainer.validate(model, datamodule=dm)[0]
    
    return {
        'top1_accuracy': results['val_top1_acc'],
        'top5_accuracy': results['val_top5_acc'],
        'val_loss': results['val_loss']
    }

if __name__ == "__main__":
    args = parse_args()
    
    # Validate SAM2 arguments
    if args.encoder == 'sam2' and (args.sam_checkpoint is None or args.sam2_config is None):
        raise ValueError("SAM2 requires both --sam_checkpoint and --sam2_config arguments")
    
    # Set experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.encoder}_{timestamp}"
    
    # Convert devices to int if specified as number
    if args.devices != 'auto':
        try:
            args.devices = int(args.devices)
        except ValueError:
            pass
    
    if args.eval_only:
        # Run evaluation only
        metrics = evaluate_checkpoint(
            checkpoint_path=args.checkpoint_path,
            encoder_name=args.encoder,
            data_dir=args.data_dir,
            sam_checkpoint=args.sam_checkpoint,
            sam2_config=args.sam2_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            precision=args.precision,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        
        print("\nEvaluation Results:")
        print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2%}")
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2%}")
        print(f"Validation Loss: {metrics['val_loss']:.4f}")
    
    else:
        # Initialize trainer with CLI arguments
        metrics = train_eval_encoder(
            encoder_name=args.encoder,
            data_dir=args.data_dir,
            sam_checkpoint=args.sam_checkpoint,
            sam2_config=args.sam2_config,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            precision=args.precision,
            accelerator=args.accelerator,
            devices=args.devices,
            num_nodes=args.num_nodes,
            strategy=args.strategy,
            learning_rate=args.learning_rate,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        
        # Print results
        print("\nFinal Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")