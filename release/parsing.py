from argparse import ArgumentParser, Namespace
import json
import os
from tempfile import TemporaryDirectory
import pickle

import torch

def add_parsing_args(parser: ArgumentParser):
    """
    Adds predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.


    """
    parser.add_argument('--desc', type=str, default='',
                        help='Experiment description/notes')
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())), default=[],
                        help='Which GPU to use')

    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--n_to_generate', type=int, default=200,
                        help='Number of molecules to generate')
    parser.add_argument('--n_to_draw', type=int, default=20,
                        help='Number of molecules to draw')

    parser.add_argument('--n_policy_replay', type=int, default = 10,
                        help='Number of time a policy ... ?')
    parser.add_argument('--n_policy', type=int, default =15,
                        help='Number of policies')
    parser.add_argument('--n_iterations', type=int, default = 1000,
                        help='Number of iterations')
    parser.add_argument('--trajectory_queue_update_freq', type=int, default=10,
                        help='Update trajectories queue every _ iterations')
    parser.add_argument('--n_estimators_RFR', type=int, default = 250,
                        help='Number of estimators in a Random Forest Regressor')



    parser.add_argument('--n_jobs', type=int, default = 10,
                        help='Number of jobs in RFR')
    parser.add_argument('--n_objectives', type=int, default = 1,
                        help='Number of objectives')

    parser.add_argument('--objectives_names_and_paths', type=list, default = [
        # {'name':'jak1_binary', 'model_class': 'CatBoost', 'interval': None, 'model_type': 'classifier', 'data_path': '/home/dzvinka/ReLeaSE/data/jak1binary_data.csv', 'model_params': {'n_estimators': 250}, 'stats_to_norm': [0,  1], 'stats_to_real': [0, 1]},
        # {'name': 'jak3_binary', 'model_class': 'CatBoost', 'interval': None, 'model_type': 'classifier',
        #  'data_path': '/home/dzvinka/ReLeaSE/data/jak3binary_data.csv', 'model_params': {'n_estimators': 250},
        #  'stats_to_norm': [0, 1], 'stats_to_real': [0, 1]},

        {'name':'IC50',  'model_class': 'RFR', 'interval': None, 'model_type': 'regressor', 'data_path': '/home/dzvinka/ReLeaSE/data/jak2noso2actives_data.csv', 'model_params': {'n_estimators': 1200, 'min_samples_split': 5,
 'min_samples_leaf': 2,
 'max_features': 'sqrt',
 'max_depth': 10,
 'bootstrap': False}, 'stats_to_norm': [6.571506827780664,  0.405556162912055], 'stats_to_real': [0, 1]},
        {'name':'logP', 'model_class': 'RFR', 'interval': [2, 4], 'model_type': 'regressor', 'data_path': '/home/dzvinka/ReLeaSE/data/logP_labels.csv', 'model_params': {'n_estimators': 250, 'n_jobs': 10,}, 'stats_to_norm': [2.3744083807347933,  0.9429912852261682], 'stats_to_real': [0,1]},
        {'name':'mpC', 'model_class': 'RFR', 'interval': [50, 250], 'model_type': 'regressor', 'data_path': '/home/dzvinka/ReLeaSE/data/mpC_data.csv', 'model_params': {'n_estimators': 250, 'n_jobs': 10,}, 'stats_to_norm': [153.76311716715495,  33.472182703355514], 'stats_to_real': [92.0924114641032, 93.92472573003356]},
        {'name': 'mwt',  'model_type': 'regressor', 'interval': [180, 459], 'stats_to_real': [0, 1], 'stats_to_norm': [367.93905744040325, 107.99177665887633]}
    ],
        help='Settings for the corresponding objectives.')


    parser.add_argument('--generator_data_path', type=str, default='/home/dzvinka/ReLeaSE/data/chembl_22_clean_1576904_sorted_std_final.smi',
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--generator_hidden_size', type=int, default=1500,
                        help='Hidden size of the generator')
    parser.add_argument('--generator_stack_width', type=int, default=1500,
                        help='Generator stack width')
    parser.add_argument('--generator_stack_depth', type=int, default=200,
                        help='Generator stack depth')
    parser.add_argument('--generator_layer_type', type=str, default='GRU',
                        help='Generator layer type')
    parser.add_argument('--generator_lr', type=float, default=0.001,
                        help='Generator learning rate')
    parser.add_argument('--generator_model_path', type=str, default='/home/dzvinka/ReLeaSE/checkpoints/generator/checkpoint_biggest_rnn',
                        help='Path with to a generator model\'s checkpoint')

    parser.add_argument('--path_to_predictors', type=str,
                        default='/home/dzvinka/ReLeaSE/checkpoints/predictors/',
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--experiments_general_path', type=str,
                        default='/home/dzvinka/ReLeaSE/experiments/',
                        help='Path to directory with stored visualisations.')
    parser.add_argument('--experiment_name', type=str,
                        default='test_four_rwds_ranges_ic50_max___',
                        help='Path to directory with stored visualisations will be based on this value')
    parser.add_argument('--reward_func', type=str, default='multi_reward_ranges_max_ic50',
                        help='Reward function.')
    parser.add_argument('--end_of_batch_reward_func', type=str, default='',
                        help='Reward function to calculate based on batch of trajectories.')
    parser.add_argument('--store_visuals', type=bool, default=True,
                        help='Indicator whether to store the visualisations during training.')



def parsing() -> Namespace:

    # print("Define path to every corresponding objective, its type and name via corresponding tags: "
    #       "--objective_type_..., --objective_name_..., --path_to_objective_...")

    parser = ArgumentParser()
    add_parsing_args(parser)
    args = parser.parse_args()

    # for obj in range(args.n_objectives):
    #     print("Path to objective #", obj)
    #     parser.add_argument('--objective_type_' + str(obj), type=str, default = 'regressor', required=True,
    #                         help='Type of prediction of corresponding objective.')
    #     parser.add_argument('--objective_name_' + str(obj), type=str, required=True,
    #                         help='Name of corresponding objective.')
    #     parser.add_argument('--path_to_objective_'+str(obj), type=str, required=True,
    #                         help='Path to a data file with objective.')


    args = parser.parse_args()

    return args
