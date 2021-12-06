import time
import datetime
import json
import numpy as np
import os
import torch
import copy
from tqdm import trange

from rdkit import Chem
from stackRNN import StackAugmentedRNN
import matplotlib
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

from data import GeneratorData
from reinforcement import Reinforcement
from utils import get_fp, simple_moving_average, moving_average, plot_hist, plot_dist, save_smiles, estimate_and_update, draw_smiles, generate, predict_and_plot
from data import PredictorData
from predictor import *

from reward_functions import get_reward_func, get_end_of_batch_reward_func

def run(args):

    print('Loading generator')
    tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

    optimizer_instance = torch.optim.Adadelta


    gen_data = GeneratorData(training_data_path=args.generator_data_path, delimiter='\t',
                         cols_to_read=[0], keep_header=True, tokens=tokens, use_cuda=args.use_cuda, pad_symbol='+')

    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=args.generator_hidden_size,
                                     output_size=gen_data.n_characters, layer_type=args.generator_layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=args.generator_stack_width, stack_depth=args.generator_stack_depth,
                                     use_cuda=args.use_cuda,
                                     optimizer_instance=optimizer_instance, lr=args.generator_lr)

    my_generator.load_model(args.generator_model_path)

    predictors = []

    for i, d in enumerate(args.objectives_names_and_paths):

        obj_parameters = copy.deepcopy(d)

        if obj_parameters['name'] in PROPERTY_PREDICTORS:
            predictor = PropertyPredictor(PROPERTY_PREDICTORS[obj_parameters['name']], obj_parameters['model_type'])
        else:
            model_type = obj_parameters['model_type']
            p_name = obj_parameters['data_path'].split('/')[-1].split('.')[0]
            print(p_name)
            # p_name = obj_parameters['name']

            model_instance = PROPERTY_ESTIMATORS[obj_parameters['model_class']]
            predictor = VanillaQSAR(model_class=obj_parameters['model_class'],
                                    model_instance=model_instance,
                                       model_params=obj_parameters['model_params'],
                                       model_type=model_type)

            if os.path.exists(args.path_to_predictors+f'predictor_{p_name}0.pkl'):
                print(f'Loading predictor {i} from {args.path_to_predictors}predictor_{p_name}')
                predictor.load_model(args.path_to_predictors+f'predictor_{p_name}')
            else:
                print(f'Fitting predictor {i}')
                data = PredictorData(path=obj_parameters['data_path'], get_features=get_fp)

                print(predictor.fit_model(data, cv_split='random'))

                predictor.save_model(args.path_to_predictors+f'predictor_{p_name}')

        predictors.append(predictor)


    # In case we want to have unbiased predictions

    unbiased_predictions = get_unbiased_predictions(args, predictors, my_generator, gen_data)


    RL_multi = Reinforcement(args, my_generator, predictors, get_reward_func(args), get_end_of_batch_reward_func(args))

    training(args, RL_multi, gen_data, predictors, unbiased_predictions)


def training(args, RL_multi, gen_data, predictors,  unbiased_predictions):
    predictors_names = [p['name'] for p in args.objectives_names_and_paths]
    print(predictors_names)
    stats_to_real = [p['stats_to_real'] for p in args.objectives_names_and_paths]
    intervals = [p['interval'] for p in args.objectives_names_and_paths]
    path_to_experiment = args.experiments_general_path+args.experiment_name

    if not os.path.exists(f'{path_to_experiment}'):
        os.mkdir(f'{path_to_experiment}')

    with open(f'{path_to_experiment}/config.json', 'w') as f:
        json.dump(vars(args), f)



    writer = SummaryWriter(f'{path_to_experiment}')


    for step in range(args.n_iterations):

        start = time.time()

        for j in trange(args.n_policy, desc='Policy gradient...'):

            # cur_reward, cur_loss, cur_distinct_rewards = RL_multi.policy_gradient(gen_data, std_smiles=True, get_features=[None] * len(predictors_names))
            cur_reward, cur_loss, cur_distinct_rewards, sampled_from_buff_ratio = RL_multi.policy_gradient(gen_data, std_smiles=False,
                                                                                  n_batch=args.batch_size)

        i = 0
        for p_name in predictors_names:
            if not 'clf' in p_name:
                writer.add_scalar(f'distinct_rewards/{p_name}', cur_distinct_rewards[i], step)
                i += 1

        writer.add_scalar(f'distinct_rewards/diversity', cur_distinct_rewards[-1], step)
        writer.add_scalar('final_reward', cur_reward, step)
        writer.add_scalar('sampled_from_buff_ratio', sampled_from_buff_ratio, step)




        smiles_cur, valid_ratio, unique_ratio = generate(RL_multi.generator, args.n_to_generate, gen_data,
                                                         args.batch_size_for_generate)

        # if step % args.trajectory_queue_update_freq == 0:
        #     RL_multi.update_trajectories(smiles_cur)

        plt.clf()

        for p_name, p, s, intrv, unbiased_preds in zip(predictors_names, predictors, stats_to_real, intervals, unbiased_predictions):

            if p.model_type == 'classifier' and not args.store_classifier_plots:
                continue
            elif p.model_type == 'classifier':
                plot_hist(unbiased_preds, p_name)
            else:
                plot_dist(unbiased_preds, p_name)

            _, prediction_cur = predict_and_plot(smiles_cur,
                                              p,
                                              get_features=get_fp,
                                              p_name=p_name,
                                              mean=s[0],
                                              std=s[1],
                                              interval=intrv
                                              )

            plt.savefig(f'{path_to_experiment}/property_dist_{p_name}.png')
            plt.clf()

            if step % 50 == 0:
                writer.add_histogram(f'{p_name}_distribution', prediction_cur * s[1] + s[0], step)

        writer.add_scalar('valid_smiles_ratio', valid_ratio, step)
        writer.add_scalar('unique_smiles_ratio', unique_ratio, step)

        if step % 50 == 0:
            save_smiles(args, smiles_cur)

            smiles, prediction_ic50 = predict_and_plot(smiles_cur, predictors[predictors_names.index('IC50_reg')], get_features=get_fp,
                                              p_name='IC50')
            img = draw_smiles(args, smiles, prediction_ic50)
            writer.add_image('Generated SMILES', matplotlib.image.pil_to_array(img), step, dataformats='HWC')

        if step % 500 == 0:
            RL_multi.generator.save_model(f'{path_to_experiment}/generator_{step}.pth')

        e = time.time() - start

        print(f'Time per iteration {step}: {datetime.timedelta(seconds=e)}')


    writer.close()


def get_unbiased_predictions(args, predictors, generator, gen_data):

    predictors_names = [i['name'] for i in args.objectives_names_and_paths]
    stats_to_real = [p['stats_to_real'] for p in args.objectives_names_and_paths]

    unbiased_predictions = []
    smiles, valid_ratio, unique_ratio = generate(generator, 500, gen_data, args.batch_size_for_generate)

    for p_name, p, s in zip(predictors_names, predictors, stats_to_real):


        _, prediction_unbiased = predict_and_plot(smiles,
                                          p,
                                          get_features=get_fp,
                                          p_name=p_name,
                                          mean=s[0],
                                          std=s[1]
                                          )

        prediction_unbiased = (prediction_unbiased * s[1]) + s[0]
        unbiased_predictions.append(prediction_unbiased)
    plt.clf()

    return unbiased_predictions
