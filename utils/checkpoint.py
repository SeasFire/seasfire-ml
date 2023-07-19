import torch
import os
import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, epoch, best_so_far, optimizer, scheduler, model_name, out_dir):
    filename = "{}/{}.checkpoint.pt".format(out_dir, model_name)
    logger.info("Saving checkpoint as {}".format(filename))

    torch.save(model.state_dict(), "{}/{}.checkpoint.model.pt".format(out_dir, model_name))
    torch.save(optimizer.state_dict(), "{}/{}.checkpoint.optimizer.pt".format(out_dir, model_name))
    torch.save(scheduler.state_dict(), "{}/{}.checkpoint.scheduler.pt".format(out_dir, model_name))
    torch.save(epoch, "{}/{}.checkpoint.epoch.pt".format(out_dir, model_name))
    torch.save(best_so_far, "{}/{}.checkpoint.best_so_far.pt".format(out_dir, model_name))


def load_checkpoint(model_name, out_dir):
    epoch = 0
    if os.path.exists("{}/{}.checkpoint.epoch.pt".format(out_dir, model_name)): 
        epoch = torch.load("{}/{}.checkpoint.epoch.pt".format(out_dir, model_name))

    model_state_dict = None
    if os.path.exists("{}/{}.checkpoint.model.pt".format(out_dir, model_name)): 
        model_state_dict = torch.load("{}/{}.checkpoint.model.pt".format(out_dir, model_name), map_location=device)

    optimizer_state_dict = None
    if os.path.exists("{}/{}.checkpoint.optimizer.pt".format(out_dir, model_name)): 
        optimizer_state_dict = torch.load("{}/{}.checkpoint.optimizer.pt".format(out_dir, model_name), map_location=device)

    scheduler_state_dict = None
    if os.path.exists("{}/{}.checkpoint.scheduler.pt".format(out_dir, model_name)): 
        scheduler_state_dict = torch.load("{}/{}.checkpoint.scheduler.pt".format(out_dir, model_name), map_location=device)

    best_so_far = None
    if os.path.exists("{}/{}.checkpoint.best_so_far.pt".format(out_dir, model_name)): 
        best_so_far = torch.load("{}/{}.checkpoint.best_so_far.pt".format(out_dir, model_name))    

    return (model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, best_so_far)