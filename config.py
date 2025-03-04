import argparse
import sys

def is_running_in_jupyter():
    """Check if the script is running inside a Jupyter notebook."""
    try:
        get_ipython()  # This function is available only in Jupyter
        return True
    except NameError:
        return False
    

def config(args=None):
    """
    Define and parse command-line arguments for the ZAugNet experiment.
    Returns:
        argparse.Namespace: Configuration parameters.
    """

    parser = argparse.ArgumentParser(description="ZAugNet Experiment Configuration")
    
    # General experiment settings
    parser.add_argument('--name', default='ZAugNet', help='experiment name')
    parser.add_argument('--dataset', dest='dataset', type=str, default='ascidians')


    # Training settings 
    parser.add_argument('--n_epochs', default=100, type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, 
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
    parser.add_argument('--model_name', default='zaugnet', choices=['zaugnet', 'zaugnet+'],
                        help='Name of the model to use')
    parser.add_argument('--lambda_adv', default=0.001, type=float,
                        help='Weight for adversarial loss')
    parser.add_argument('--lambda_gp', default=10, type=float,
                        help='Weight for gradient penalty')
    parser.add_argument('--patch_size', default=256, type=int,
                        help='Size of input images')
    parser.add_argument('--min_max', default=(0, (2**8-1)*1.0), type=tuple,
                        help='Range for min-max normalization')
    

    # Dataset and hardware settings
    parser.add_argument('--dataset_size', default=142, type=int,
                        help='Number of random triples to take from each frame')
    parser.add_argument('--distance_triplets', default=7, type=int,
                        help='Maximum distance for triplet sampling (ZAugNet+)')
    parser.add_argument('--factor', default=2, type=int,
                        help='Number of slices to add between the two original ones (ZAugNet+)')
    parser.add_argument('--DPM', default=0.5, type=float,
                        help='DPM for (ZAugNet+)')
    parser.add_argument('--device_ids', default="[0]", type=str,
                        help='List of GPU device IDs to use for training')
    parser.add_argument('--save_dataset', default=True, type=bool)

    if is_running_in_jupyter():
        config = parser.parse_args(args=[])
    else:
        config = parser.parse_args()  

    return config
