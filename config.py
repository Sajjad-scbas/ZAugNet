import argparse

def config():
    """
    Define and parse command-line arguments for the ZAugNet experiment.
    Returns:
        argparse.Namespace: Configuration parameters.
    """
    parser = argparse.ArgumentParser(description="ZAugNet Experiment Configuration")
    
    # General experiment settings
    parser.add_argument('--name', default='ZAugNet', help='experiment name')
    

    # Training settings 
    parser.add_argument('--n_epochs', default=100, type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, 
                        help='Mini-batch size for training')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, 
                        help='Initial learning rate')
    parser.add_argument('--beta1', default=0.5, type=float,
                        help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='Beta2 for Adam optimizer')
    parser.add_argument('--augmentations', default=True, type=bool, 
                        help='Whether to apply data augmentations')
    parser.add_argument('--normalization', default='min_max', choices=['min_max', 'zscore'], 
                        help='Normalization method for input data')
    parser.add_argument('--resize_par', default='resize', choices=['crop', 'resize'],
                        help='Resizing strategy for input images')
    parser.add_argument('--p_val', default=0.05, type=float, 
                        help='Validation split proportion')
    parser.add_argument('--n_critic', default=5, type=int,
                        help='Number of discriminator updates per generator update')   


    # Model-specific settings
    parser.add_argument('--model_name', default='zaugnet+', choices=['zaugnet', 'zaugnet+'],
                        help='Name of the model to use')
    parser.add_argument('--lambda_adv', default=0.001, type=float,
                        help='Weight for adversarial loss')
    parser.add_argument('--lambda_gp', default=10, type=float,
                        help='Weight for gradient penalty')
    parser.add_argument('--last_kernel_size', default=32, type=int,
                        help='Kernel size for the last convolutional layer')
    parser.add_argument('--patch_size', default=256, type=int,
                        help='Size of input images')
    parser.add_argument('--min_max', default=(0, (2**8-1)*1.0), type=tuple,
                        help='Range for min-max normalization')
    

    # Dataset and hardware settings
    parser.add_argument('--dataset_size', default=142, type=int,
                        help='Number of random triples to take from each frame')
    parser.add_argument('--distance_triplets', default=7, type=int,
                        help='Maximum distance for triplet sampling (ZAugNet+)')
    parser.add_argument('--device_ids', default=[0,1], type=list,
                        help='List of GPU device IDs to use for training')
    parser.add_argument('--save_dataset', default=True, type=bool)
    parser.add_argument('--cfg_nb', type=int, default=1, help="Nb of training dataset")


    config = parser.parse_args()    
    return config
