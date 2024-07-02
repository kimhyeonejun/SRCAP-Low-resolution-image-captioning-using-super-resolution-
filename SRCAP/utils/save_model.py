import os
import utils
import torch
def make_dir():
    model_name = f'LR={utils.generator_learning_rate} Upscaling_factor= {utils.upscaling_factor} d_out_mean={utils.d_out_mean} batch_sizet={utils.batch_size} N_ROI={utils.max_length} Fine_tuning={utils.fine_tuning}'
    folder_path = os.path.join('models',
                            'testing',
                            model_name)

    folder_ckptpath = os.path.join(folder_path,'checkpoints')
    # Check if the folder exists
    if not os.path.exists(folder_ckptpath):
        # Create the folder if it doesn't exist
        os.makedirs(folder_ckptpath)
        print(f"Folder '{folder_ckptpath}' created successfully.")
    else:
        print(f"Folder '{folder_ckptpath}' already exists.")
        if os.path.exists(os.path.join(folder_path, f"{model_name}.pt")):
            return
    return model_name, folder_path

def save_model(file_path, model_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    #torch.save(state_dict, os.path.join(file_path, model_name))

    #report.to_csv(os.path.join(report_path,f"{model_name}_report.csv")) : save a dictionary that includes a variety of model information
    torch.save(model.state_dict(), os.path.join(file_path,f"{model_name}.pt"))