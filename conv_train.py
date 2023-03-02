import os
import os.path as osp
import yaml
import shutil
import torch
import click
import warnings

from loss import ESTyleLoss
warnings.simplefilter('ignore')

from munch import Munch
from tqdm import tqdm
from conv_dataset import build_dataloader
from torch.utils.tensorboard import SummaryWriter
from conv_model import DurationChangeNet
from Utils.token import pad_token
import logging
from logging import StreamHandler
from Modules.Discriminator import Discriminator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
torch.autograd.set_detect_anomaly(True)


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
@click.option("-w","--num_worker",default=1, type=int)
def main(config_path, num_worker):
    
    # Configs reader
    config = yaml.safe_load(open(config_path))
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # Define logger
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)
    
    ### Get configuration
    batch_size = config.get('batch_size', 2)
    device = config.get('device', 'cuda:1')
    epochs = config.get('epochs', 1000)
    save_freq = config.get('save_freq', 20)
    pretrained_model = Munch(config.get("pretrained_model"))
    dataset_configuration = config.get('dataset_configuration', None)
    training_set_path = dataset_configuration["training_set_path"]
    validation_set_path = dataset_configuration["validation_set_path"]
    model_architecture = Munch(config.get("model_architecture"))
    training_parameter = Munch(config.get("training_parameter"))
    ###
    
    is_postnet_active = model_architecture.post_net["activate"]
    # load dataloader 
    train_dataloader = build_dataloader(training_set_path,dataset_configuration,
                                        batch_size=batch_size,
                                        num_workers=num_worker,
                                        device=device)
    
    val_dataloader = build_dataloader(validation_set_path,dataset_configuration,
                                        batch_size=batch_size,
                                        num_workers=num_worker,
                                        device=device,
                                        validation=True)
    
    # Module Definition
    generator = DurationChangeNet()
    generator.to(device)
    
    discriminator = Discriminator()
    discriminator.to(device)
    
    lr = training_parameter.learning_rate
    
    gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=1e-4)
    dis_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.00001, weight_decay=1e-4)
    best_val_loss = float('+inf')
    epoch = 0
    
    # Load checkpoint, if exists
    if os.path.exists(osp.join(log_dir, 'estyle_backup.pth')):
        checkpoint = torch.load(osp.join(log_dir, 'estyle_backup.pth'), map_location=device) # Fix from https://github.com/pytorch/pytorch/issues/2830#issuecomment-718816292
        if generator.encoder_prenet is not None:
            generator.encoder_prenet.load_state_dict(checkpoint["encoder_prenet"])
        if generator.decoder_prenet is not None:
            generator.decoder_prenet.load_state_dict(checkpoint["decoder_prenet"])
        if generator.encoder is not None:
            generator.encoder.load_state_dict(checkpoint["encoder"])
        if generator.encoder_norm is not None:
            generator.encoder_norm.load_state_dict(checkpoint["encoder_norm"])
        if generator.decoder is not None:
            generator.decoder.load_state_dict(checkpoint["decoder"])
        if generator.decoder_norm is not None:
            generator.decoder_norm.load_state_dict(checkpoint["decoder_norm"])
        if generator.decpost_interface is not None:
            generator.decpost_interface.load_state_dict(checkpoint["decpost_interface"])
        if generator.decpost_interface_norm is not None:
            generator.decpost_interface_norm.load_state_dict(checkpoint["decpost_interface_norm"])
        if generator.postnet is not None:
            generator.postnet.load_state_dict(checkpoint["postnet"])
        if generator.postnet_norm is not None:
            generator.postnet_norm.load_state_dict(checkpoint["postnet_norm"])
        best_val_loss = checkpoint["loss"]
        epoch = checkpoint["reached_epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
    

    for epoch in range(epoch, epochs+1):
        adv_train_loss, adv_validation_loss = 0,0
        features_matching_training_loss, features_matching_validation_loss =  0,0
        mse_training_loss, mse_validation_loss = 0,0
        
        # Train
        for _, batch in enumerate(tqdm(train_dataloader, desc="[train]"), 1):
            batch = [b.to(device) for b in batch]
            losses = train_epoch(batch, generator, discriminator, gen_optimizer, dis_optimizer, device)
            adv_train_loss += losses["adv_loss"]
            features_matching_training_loss += losses["features_matching_loss"]
            mse_training_loss += losses["mse_loss"]
            
        # Validation    
        for _, batch in enumerate(tqdm(val_dataloader, desc="[validation]"), 1):
            batch = [b.to(device) for b in batch]
            losses = eval_epoch(batch, generator, discriminator, gen_optimizer, dis_optimizer, device)
            adv_validation_loss += losses["adv_loss"]
            features_matching_validation_loss += losses["features_matching_loss"]
            mse_validation_loss += losses["mse_loss"]
            
        # Training epoch loss
        epoch_adv_training_loss = adv_train_loss/len(train_dataloader)
        epoch_features_matching_training_loss = features_matching_training_loss/len(train_dataloader)
        epoch_mse_training_loss = mse_training_loss/len(train_dataloader)
        
        
        # Validation epoch loss
        epoch_adv_validation_loss = adv_validation_loss/len(val_dataloader)
        epoch_features_matching_validation_loss = features_matching_validation_loss/len(val_dataloader)
        epoch_mse_validation_loss = mse_validation_loss/len(val_dataloader)
        

        # Log writer
        logger.info('--- epoch %d ---' % epoch)
        logger.info("%-15s: %.4f" % ('epoch_adv_training_loss', epoch_adv_training_loss))
        logger.info("%-15s: %.4f" % ('epoch_features_matching_training_loss', epoch_features_matching_training_loss))
        logger.info("%-15s: %.4f" % ('epoch_mse_training_loss', epoch_mse_training_loss))
        logger.info("")
        logger.info("%-15s: %.4f" % ('epoch_adv_validation_loss', epoch_adv_validation_loss))
        logger.info("%-15s: %.4f" % ('epoch_features_matching_validation_loss', epoch_features_matching_validation_loss))
        logger.info("%-15s: %.4f" % ('epoch_mse_validation_loss', epoch_mse_validation_loss))
        
        # Generate tensorboard logs for the epoch 
        writer.add_scalar('epoch_adv_training_loss', epoch_adv_training_loss, epoch) 
        writer.add_scalar('epoch_features_matching_training_loss', epoch_features_matching_training_loss, epoch) 
        writer.add_scalar('epoch_mse_training_loss', epoch_mse_training_loss, epoch) 
        
        writer.add_scalar('epoch_adv_validation_loss', epoch_adv_validation_loss, epoch) 
        writer.add_scalar('epoch_features_matching_validation_loss', epoch_features_matching_validation_loss, epoch) 
        writer.add_scalar('epoch_mse_validation_loss', epoch_mse_validation_loss, epoch) 
        
        epoch_cumulative_validation_loss = epoch_adv_validation_loss + epoch_features_matching_validation_loss + epoch_mse_validation_loss

        # Saving point
        if epoch_cumulative_validation_loss < best_val_loss:
           save_checkpoint(osp.join(log_dir, 'estyle_best.pth'), generator, discriminator, gen_optimizer, dis_optimizer, epoch, epoch_cumulative_validation_loss) 
           best_val_loss = epoch_cumulative_validation_loss
           print("Best found! Save!")
        if (epoch % save_freq) == 0:
            save_checkpoint(osp.join(log_dir, 'estyle_backup.pth'), generator, discriminator, gen_optimizer, dis_optimizer, epoch, epoch_cumulative_validation_loss)
        
        
def train_epoch(batch, generator: torch.nn.Module, discriminator, gen_optimizer: torch.optim.Optimizer, dis_optimizer: torch.optim.Optimizer, device) -> float:
    """Function that perform a training step on the given batch

    Args:
        batch (List[__get_item__]): Sample batch.
        model (torch.nn.Module): Model.
        loss (torch.nn.SmoothL1Loss): Loss associated.
        optimizer (torch.optim.Optimizer): Optimizer associated.

    Returns:
        float: Loss value
    """    
    adv_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss(reduction="none")
    generator.train()
    discriminator.train()
    
    mel_lengths, ref_mel_lengths, padded_mel_tensor, padded_ref_mel_input_tensor, padded_ref_mel_target_tensor, mel_mask, ref_mel_input_mask, mel_padding_mask, ref_mel_input_padding_mask = batch
    trues = torch.ones((5,1)).to(device)
    falses = torch.zeros((5,1)).to(device)
    
    # Output
    fake_gen_out = generator(padded_mel_tensor, mel_lengths)
    
    # Discriminator training
    dis_optimizer.zero_grad()
    
    real_dis_out = discriminator(padded_ref_mel_target_tensor)
    real_dis_loss = adv_loss(real_dis_out, trues)
    
    fake_gen_dis_out = discriminator(fake_gen_out.detach())
    fake_gen_dis_out = adv_loss(fake_gen_dis_out, falses)
    
    dis_loss = real_dis_loss + fake_gen_dis_out
    dis_loss.backward()
    dis_optimizer.step()
    
    
    mask = (padded_ref_mel_target_tensor != pad_token.to(device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)).to(device) # obtain a mask from the target reference
    # Generator training
    gen_optimizer.zero_grad()
    
    # ADV Loss
    fake_gen_dis_out = discriminator(fake_gen_out)
    fake_gen_adv_loss = adv_loss(fake_gen_dis_out, trues)
    
    # MSE Loss        
        # Postnet-Target Loss
    fake_gen_mse_loss_wout_mask = mse_loss(fake_gen_out, padded_ref_mel_target_tensor)
    fake_gen_mse_loss_w_mask = fake_gen_mse_loss_wout_mask.where(mask, torch.tensor(0.0).to(device))
    fake_gen_mse_loss = fake_gen_mse_loss_w_mask.sum() / mask.sum()
    
    # # Features Matching Loss
    # fake_gen_features_matching_loss = torch.functional.F.mse_loss(discriminator.get_features(fake_gen_out.unsqueeze(1).transpose(3,2)), discriminator.get_features(padded_ref_mel_target_tensor.unsqueeze(1).transpose(3,2)))
    
    cumulative_loss = fake_gen_adv_loss + fake_gen_mse_loss 
    cumulative_loss.backward()
    gen_optimizer.step()
    
    return {
        "adv_loss": fake_gen_adv_loss.detach().item(),
        "mse_loss": fake_gen_mse_loss.detach().item(),
        "features_matching_loss": 0
    }

def eval_epoch(batch, generator: torch.nn.Module, discriminator, gen_optimizer: torch.optim.Optimizer, dis_optimizer: torch.optim.Optimizer, device) -> float:
    """Function that perform an evaluation step on the given batch 

    Args:
        batch (List[__get_item__]): Sample batch
        model (torch.nn.Module): Model
        loss (torch.nn): Loss

    Returns:
        float: Loss value
    """    
    generator.eval()
    discriminator.eval()
    adv_loss = torch.nn.BCELoss()
    trues = torch.ones((5,1)).to(device)
    falses = torch.zeros((5,1)).to(device)
    mel_lengths, ref_mel_lengths, padded_mel_tensor, padded_ref_mel_input_tensor, padded_ref_mel_target_tensor, mel_mask, ref_mel_input_mask, mel_padding_mask, ref_mel_input_padding_mask = batch
    
    with torch.no_grad():
        fake_gen_out = generator(padded_mel_tensor, mel_lengths)
        
        mask = (padded_ref_mel_target_tensor != pad_token.to(device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)).to(device) # obtain a mask from the target reference
        
        fake_dis_out = discriminator(fake_gen_out)
        
        # ADV Loss
        fake_adv_loss = adv_loss(fake_dis_out, trues)
        
        # MSE Loss        
            # Transformer-Target Loss
        fake_gen_mse_loss_wout_mask = torch.functional.F.mse_loss(fake_gen_out, padded_ref_mel_target_tensor, reduction="none")
        fake_gen_mse_loss_w_mask = fake_gen_mse_loss_wout_mask.where(mask, torch.tensor(0.0).to(device))
        fake_gen_mse_loss = fake_gen_mse_loss_w_mask.sum() / mask.sum()
        
        # Features Matching Loss
        # fake_transformer_features_matching_loss = torch.functional.F.mse_loss(discriminator.get_features(fake_transformer_gen_out.unsqueeze(1).transpose(3,2)), discriminator.get_features(padded_ref_mel_target_tensor.unsqueeze(1).transpose(3,2)))
        # fake_postnet_features_matching_loss = torch.functional.F.mse_loss(discriminator.get_features(fake_postnet_gen_out.unsqueeze(1).transpose(3,2)), discriminator.get_features(padded_ref_mel_target_tensor.unsqueeze(1).transpose(3,2)))
    
    return {
        "adv_loss": fake_adv_loss.item(),
        "mse_loss": fake_gen_mse_loss.item(),
        "features_matching_loss": 0
    }

def save_checkpoint(checkpoint_path: str, generator, discriminator, gen_optimizer: torch.optim.Optimizer, dis_optimizer: torch.optim.Optimizer, epoch: int, actual_loss: float):
    """
        Save checkpoint.
        
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
            model (nn.Module): Model to be saved.
            optimizer (torch.optim.Optimizer): Optimizer to be saved.
            epoch (int): Number of training epoch reached.
            actual_loss (float): Actual loss.
            global_step (int): Step counter 
    """
    state_dict = {
        "gen_optimizer": gen_optimizer.state_dict(),
        "dis_optimizer": dis_optimizer.state_dict(),
        "reached_epoch": epoch,
        "loss": actual_loss,
        "generator": generator.state_dict(),
        # "encoder_prenet": generator.encoder_prenet.state_dict() if generator.encoder_prenet is not None else None,
        # "decoder_prenet": generator.decoder_prenet.state_dict() if generator.decoder_prenet is not None else None,
        # "encoder": generator.encoder.state_dict() if generator.encoder is not None else None,
        # "encoder_norm": generator.encoder_norm.state_dict() if generator.encoder_norm is not None else None,
        # "decoder": generator.decoder.state_dict() if generator.decoder is not None else None,
        # "decoder_norm": generator.decoder_norm.state_dict() if generator.decoder_norm is not None else None,
        # "decpost_interface": generator.decpost_interface.state_dict() if generator.decpost_interface is not None else None,
        # "decpost_interface_norm": generator.decpost_interface_norm.state_dict() if generator.decpost_interface_norm is not None else None,
        # "postnet": generator.postnet.state_dict() if generator.postnet is not None else None,
        # "postnet_norm": generator.postnet_norm.state_dict() if generator.postnet_norm is not None else None,
        "discriminator": discriminator.state_dict()
    }
    
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    torch.save(state_dict, checkpoint_path)

if __name__=="__main__":
    main()
