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
from dataset import build_dataloader
from torch.utils.tensorboard import SummaryWriter
from model import ESTyle
from Utils.token import pad_token
import logging
from logging import StreamHandler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True #

def freeze_layer(model, configuration, mode = False):
    # What i need to freeze? Let's see
    # Encoder Prenet
    for param in model.encoder_prenet.parameters():
        param.requires_grad = mode
        
    # Encoder
    for index, layer in enumerate(model.encoder):
        if index > configuration.encoder:
            break
        for param in layer.parameters():
            param.requires_grad = mode
    
    # Decoder Prenet
    if configuration.decoder_prenet > 0:     
        for param in model.decoder_prenet.parameters():
            param.requires_grad = mode
            
    # Decoder
    for index, layer in enumerate(model.decoder):
        if index > configuration.decoder:
            break
        for param in layer.parameters():
            param.requires_grad = mode
            
    if configuration.decpost_interface > 0:
        for param in model.decpost_interface.parameters():
            param.requires_grad = mode

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
    device = config.get('device', 'cuda:0')
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
                                        device=device)
    
    # Module Definition
    model = ESTyle(model_architecture)
    model.to(device)
    
    # Load an eventual pretrained encoder
    if pretrained_model.path != "":
        if not os.path.exists(pretrained_model.path):
            raise FileNotFoundError(f"Pretrained Encoder Path not found! got: {pretrained_model.path}")
        checkpoint = torch.load(pretrained_model.path, map_location=device)
        if model.encoder_prenet is not None:
            model.encoder_prenet.load_state_dict(checkpoint["encoder_prenet"])
        if model.decoder_prenet is not None:
            model.decoder_prenet.load_state_dict(checkpoint["decoder_prenet"])
        if model.encoder is not None:
            model.encoder.load_state_dict(checkpoint["encoder"])
        if model.encoder_norm is not None:
            model.encoder_norm.load_state_dict(checkpoint["encoder_norm"])
        if model.decoder is not None:
            model.decoder.load_state_dict(checkpoint["decoder"])
        if model.decoder_norm is not None:
            model.decoder_norm.load_state_dict(checkpoint["decoder_norm"])
        if model.decpost_interface is not None:
            model.decpost_interface.load_state_dict(checkpoint["decpost_interface"])
        # if model.decpost_interface_norm is not None:
        #     model.decpost_interface_norm.load_state_dict(checkpoint["decpost_interface_norm"])
       
        

    lr = training_parameter.learning_rate
    noam_factor = training_parameter.warmp_up["noam_factor"]
    warm_up_step = training_parameter.warmp_up["warm_up_step"]
    
    
    # def noam_schedule(step):
    #     step+=1
    #     return noam_factor**(-0.5) * min(step**(-0.5), step * warm_up_step**(-1.5))
    
    losses = Munch(smoothl1=torch.nn.MSELoss(reduction="none"))
    loss = ESTyleLoss(losses, pad_token, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    best_val_loss = float('+inf')
    epoch = 0
    
    # Load checkpoint, if exists
    if os.path.exists(osp.join(log_dir, 'estyle_backup.pth')):
        checkpoint = torch.load(osp.join(log_dir, 'estyle_backup.pth'), map_location=device) # Fix from https://github.com/pytorch/pytorch/issues/2830#issuecomment-718816292
        if model.encoder_prenet is not None:
            model.encoder_prenet.load_state_dict(checkpoint["encoder_prenet"])
        if model.decoder_prenet is not None:
            model.decoder_prenet.load_state_dict(checkpoint["decoder_prenet"])
        if model.encoder is not None:
            model.encoder.load_state_dict(checkpoint["encoder"])
        if model.encoder_norm is not None:
            model.encoder_norm.load_state_dict(checkpoint["encoder_norm"])
        if model.decoder is not None:
            model.decoder.load_state_dict(checkpoint["decoder"])
        if model.decoder_norm is not None:
            model.decoder_norm.load_state_dict(checkpoint["decoder_norm"])
        if model.decpost_interface is not None:
            model.decpost_interface.load_state_dict(checkpoint["decpost_interface"])
        if model.decpost_interface_norm is not None:
            model.decpost_interface_norm.load_state_dict(checkpoint["decpost_interface_norm"])
        if model.postnet is not None:
            model.postnet.load_state_dict(checkpoint["postnet"])
        if model.postnet_norm is not None:
            model.postnet_norm.load_state_dict(checkpoint["postnet_norm"])
        best_val_loss = checkpoint["loss"]
        epoch = checkpoint["reached_epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
    
    
    if pretrained_model.path != "":
        stage_counter = 1
        freeze_layer(model, Munch(pretrained_model.layer_freeze_mode[f"stage_{stage_counter}"]))
        overfitting_alarm = 5
        overfitting_counter = 0
        flag_change_stage = False
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    for epoch in range(epoch, epochs+1):
        cumulative_train_loss, transformer_train_loss, postnet_train_loss = 0,0,0
        cumulative_validation_loss, transformer_validation_loss, postnet_validation_loss = 0,0,0
        if pretrained_model.path != "":
            logger.info("%-15s" % ("Check stage update.."))
            if overfitting_counter > overfitting_alarm:
                logger.info("%-15s" % ("Alarm raised, start stage change"))
                flag_change_stage = ~flag_change_stage
            logger.info(f"Ok.")
        
        if pretrained_model.path != "":
            # Check if we need stage change
            if flag_change_stage: 
                logger.info(f"De-freeze from stage {stage_counter}")
                # De-Freeze all for security
                freeze_layer(model, Munch(pretrained_model.layer_freeze_mode[f"stage_{stage_counter}"]), mode=True)
                stage_counter += 1
                
                logger.info(f"Freeze in stage {stage_counter}")
                # Freeze in new stage
                freeze_layer(model, Munch(pretrained_model.layer_freeze_mode[f"stage_{stage_counter}"]))
                flag_change_stage = ~flag_change_stage
                overfitting_counter = 0
                logger.info(f"Done.")
        
        # Train
        for _, batch in enumerate(tqdm(train_dataloader, desc="[train]"), 1):
            batch = [b.to(device) for b in batch]
            _cumulative_train_loss, _transformer_train_loss, _postnet_train_loss = train_epoch(batch, model, loss, optimizer)
            cumulative_train_loss += _cumulative_train_loss
            transformer_train_loss += _transformer_train_loss
            if is_postnet_active:
                postnet_train_loss += _postnet_train_loss
            
        # Validation    
        for _, batch in enumerate(tqdm(val_dataloader, desc="[validation]"), 1):
            batch = [b.to(device) for b in batch]
            _cumulative_validation_loss, _transformer_validation_loss, _postnet_validation_loss = eval_epoch(batch, model, loss)
            cumulative_validation_loss += _cumulative_validation_loss
            transformer_validation_loss += _transformer_validation_loss
            if is_postnet_active:
                postnet_validation_loss += _postnet_validation_loss
            
        epoch_cumulative_train_loss = cumulative_train_loss/len(train_dataloader)
        epoch_transformer_train_loss = transformer_train_loss/len(train_dataloader)
        epoch_postnet_train_loss = postnet_train_loss/len(train_dataloader)
            
        epoch_cumulative_validation_loss = cumulative_validation_loss/len(val_dataloader)
        epoch_transformer_validation_loss = transformer_validation_loss/len(val_dataloader)
        epoch_postnet_validation_loss = postnet_validation_loss/len(val_dataloader)
        
        scheduler.step(epoch_cumulative_validation_loss) # Decide if it is necessary to update the learning rate
        
        # Log writer
        logger.info('--- epoch %d ---' % epoch)
        logger.info("%-15s: %.4f" % ('cumulative_train_loss', epoch_cumulative_train_loss))
        logger.info("%-15s: %.4f" % ('transformer_train_loss', epoch_transformer_train_loss))
        logger.info("%-15s: %.4f" % ('postnet_train_loss', epoch_postnet_train_loss))
        logger.info("")
        logger.info("%-15s: %.4f" % ('cumulative_validation_loss', epoch_cumulative_validation_loss))
        logger.info("%-15s: %.4f" % ('transformer_validation_loss', epoch_transformer_validation_loss))
        logger.info("%-15s: %.4f" % ('postnet_validation_loss', epoch_postnet_validation_loss))
            
        
        if pretrained_model.path != "":
            if epoch_cumulative_validation_loss > best_val_loss:
                overfitting_counter += 1
        
        # Saving point
        if epoch_cumulative_validation_loss < best_val_loss:
           overfitting_counter = 0
           save_checkpoint(osp.join(log_dir, 'estyle_best.pth'), model, optimizer, epoch, epoch_cumulative_validation_loss) 
           best_val_loss = epoch_cumulative_validation_loss
           print("Best found! Save!")
        if (epoch % save_freq) == 0:
            save_checkpoint(osp.join(log_dir, 'estyle_backup.pth'), model, optimizer, epoch, epoch_cumulative_validation_loss)
        
        
def train_epoch(batch, model: torch.nn.Module, loss: torch.nn.SmoothL1Loss, optimizer: torch.optim.Optimizer) -> float:
    """Function that perform a training step on the given batch

    Args:
        batch (List[__get_item__]): Sample batch.
        model (torch.nn.Module): Model.
        loss (torch.nn.SmoothL1Loss): Loss associated.
        optimizer (torch.optim.Optimizer): Optimizer associated.

    Returns:
        float: Loss value
    """    
    model.train()
    optimizer.zero_grad()
    mel_lengths, ref_mel_lengths, padded_mel_tensor, padded_ref_mel_input_tensor, padded_ref_mel_target_tensor, mel_mask, ref_mel_input_mask, mel_padding_mask, ref_mel_input_padding_mask = batch
    transformer_output, postnet_output = model(padded_mel_tensor, padded_ref_mel_input_tensor, mel_padding_mask, ref_mel_input_padding_mask, mel_mask, ref_mel_input_mask)
    cumulative_loss, transformer_loss, postnet_loss = loss.evaluate(transformer_output, postnet_output, padded_ref_mel_target_tensor, mel_lengths, ref_mel_lengths)
    cumulative_loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm(model.parameters(),1.)
    optimizer.step()
    # scheduler.step()
    return cumulative_loss.detach().item(), transformer_loss.detach().item(), postnet_loss.detach().item()

def eval_epoch(batch, model: torch.nn.Module, loss: torch.nn) -> float:
    """Function that perform an evaluation step on the given batch 

    Args:
        batch (List[__get_item__]): Sample batch
        model (torch.nn.Module): Model
        loss (torch.nn): Loss

    Returns:
        float: Loss value
    """    
    model.eval()
    with torch.no_grad():
        mel_lengths, ref_mel_lengths, padded_mel_tensor, padded_ref_mel_input_tensor, padded_ref_mel_target_tensor, mel_mask, ref_mel_input_mask, mel_padding_mask, ref_mel_input_padding_mask = batch
        transformer_output, postnet_output  = model(padded_mel_tensor, padded_ref_mel_input_tensor, mel_padding_mask, ref_mel_input_padding_mask, mel_mask, ref_mel_input_mask)
        cumulative_loss, transformer_loss, postnet_loss = loss.evaluate(transformer_output, postnet_output, padded_ref_mel_target_tensor, mel_lengths, ref_mel_lengths)
    return cumulative_loss.detach().item(), transformer_loss.detach().item(), postnet_loss.detach().item()

def save_checkpoint(checkpoint_path: str, model: ESTyle, optimizer: torch.optim.Optimizer, epoch: int, actual_loss: float):
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
        "optimizer": optimizer.state_dict(),
        "reached_epoch": epoch,
        "loss": actual_loss,
        "encoder_prenet": model.encoder_prenet.state_dict() if model.encoder_prenet is not None else None,
        "decoder_prenet": model.decoder_prenet.state_dict() if model.decoder_prenet is not None else None,
        "encoder": model.encoder.state_dict() if model.encoder is not None else None,
        "encoder_norm": model.encoder_norm.state_dict() if model.encoder_norm is not None else None,
        "decoder": model.decoder.state_dict() if model.decoder is not None else None,
        "decoder_norm": model.decoder_norm.state_dict() if model.decoder_norm is not None else None,
        "decpost_interface": model.decpost_interface.state_dict() if model.decpost_interface is not None else None,
        "decpost_interface_norm": model.decpost_interface_norm.state_dict() if model.decpost_interface_norm is not None else None,
        "postnet": model.postnet.state_dict() if model.postnet is not None else None,
        "postnet_norm": model.postnet_norm.state_dict() if model.postnet_norm is not None else None
    }
    
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    torch.save(state_dict, checkpoint_path)

if __name__=="__main__":
    main()
