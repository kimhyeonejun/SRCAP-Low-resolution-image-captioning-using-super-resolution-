import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access dataset parameters
root                = config['dataset']['root']
validation_dataset_json   = config['dataset']['validation_dataset_json']
training_dataset_json = config['dataset']['training_dataset_json']
crop_size           = config['dataset']['crop_size']
upscaling_factor    = config['dataset']['upscaling_factor']

optimizer_name      = config['optimizer']['name']
weight_decay        = config['optimizer']['weight_decay']
opt_betas           = config['optimizer']['betas']

inference_mode      = config['inference_mode']
generator_learning_rate       = config['learning_rate']['generator']
discriminator_learning_rate   = config['learning_rate']['discriminator']
num_epochs          = config['num_epochs']
num_proposals       = config['num_proposals']

scheduler_name          = config['scheduler']['name']
scheduler_activate      = config['scheduler']['scheduler_activate']
gamma                   = config['scheduler']['gamma']
total_iters             = config['scheduler']['total_iters']
d_out_mean             = config['d_out_mean']
batch_size              = config['batch_size']
max_length              = config['max_length']
fine_tuning             = config['fine_tuning']