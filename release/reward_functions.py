import numpy as np
from utils import get_fp, get_fp_and_mol, get_ECFP, normalize_fp, mol2image
from predictor import PROPERTY_PREDICTORS
import Levenshtein

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem


def tanimoto_sim(arr1, arr2):
    if not np.any(arr1) and not np.any(arr2):
        return 0.
    return np.sum((arr2 & arr1)) / np.sum((arr2 | arr1))

def calc_sim_ind_tanimoto(fngps):
    n_of_smiles = len(fngps)
    si_matrix = np.eye(n_of_smiles, dtype=np.float32)

    for i1 in range(n_of_smiles):
        for i2 in range(i1 + 1, n_of_smiles):
            si = tanimoto_sim(fngps[i1], fngps[i2])
            si_matrix[i1, i2] = si_matrix[i2, i1] = si

    #si_vector = (np.sum(si_matrix, axis=0) - 1) / (n_of_smiles - 1)
    si_vector = np.mean(si_matrix, axis=0)

        #smiles_with_si[sm] = round(si_vector[i] * (19.5 - mols[sm].GetNumAtoms())**2, 3)


    return si_vector

def calc_sim_ind_levenstein(smiles):
    n_of_smiles = len(smiles)
    si_matrix = np.eye(n_of_smiles, dtype=np.float32)

    for i1 in range(n_of_smiles):
        for i2 in range(i1 + 1, n_of_smiles):
            si = 1.0 / (1.0 + Levenshtein.distance(smiles[i1], smiles[i2]))
            si_matrix[i1, i2] = si_matrix[i2, i1] = si

    #si_vector = (np.sum(si_matrix, axis=0) - 1) / (n_of_smiles - 1)
    si_vector = np.mean(si_matrix, axis=0)

        #smiles_with_si[sm] = round(si_vector[i] * (19.5 - mols[sm].GetNumAtoms())**2, 3)


    return si_vector


def get_end_of_batch_reward_tanimoto_sim(smiles):
    fngps = [mol2image(Chem.MolFromSmiles(sm)) for sm in smiles]

    avg_sim = calc_sim_ind_tanimoto(fngps)

    rwd = [-np.power(s * 4, 2) - 1 if s > 0.37 else 0 for s in avg_sim]

    return rwd

def get_end_of_batch_reward_levenstein_sim(smiles):
    avg_sim = calc_sim_ind_levenstein(smiles)

    rwd = [-np.power(s * 4, 2) - 1 if s > 0.4 else 0 for s in avg_sim]

    return rwd

def get_end_of_batch_reward_tanimoto_levenstein_sim(smiles):
    fngps = [mol2image(Chem.MolFromSmiles(sm)) for sm in smiles]

    avg_sim_tan = calc_sim_ind_tanimoto(fngps)
    avg_sim_lev = calc_sim_ind_levenstein(smiles)


    rwd = [-np.power(s1 * s2 * 10, 4) - 1 if s1 * s2 > 0.08 else 0 for s1, s2 in zip(avg_sim_tan, avg_sim_lev)]
    # rwd = [-np.power(max(s1, s2) * 4, 2) - 1 if max(s1, s2) > 0.4 else 0 for s1, s2 in zip(avg_sim_tan, avg_sim_lev)]


    return rwd


def get_multi_reward_ranges_multiple_ic50_smiles_clf_and_reg_rings(args, smiles, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = []
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]
    indx_to_predict = np.arange(0, len(smiles))

    fngps = [mol2image(sm) for sm in smiles]
    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):
        if p_name_ in PROPERTY_PREDICTORS:
            mol, prop, nan_smiles = p.predict([smiles[j] for j in indx_to_predict], get_features=None)
        else:
            mol, prop, nan_smiles = p.predict([fngps[j] for j in indx_to_predict], get_features=None)

        if len(nan_smiles) > 0:
            print('NAN smiles in prediction')
            # return invalid_reward, [invalid_reward] * len(predictors)

        if p_name_ == 'IC50_reg':  # ic50
            rwds.append(np.exp((prop - 5) / 3.))

        elif p_name_ == 'jak2_clf':  # binds/doesn't bind to jak1
            dstnctv_rwds = np.zeros((len(prop),), dtype=np.float32)
            dstnctv_rwds[np.where(prop == 0)] = 0.5
            rwds.append(dstnctv_rwds)

            indx_to_predict = np.where(prop == 1.)[0]
        elif p_name_ == 'jak2_reg':  # jak1
            if len(indx_to_predict) > 0:
                rwds[-1][np.array(indx_to_predict)] = -0.5 * np.maximum(np.full((len(indx_to_predict),), -1), np.exp((prop - 5) / 3.))
            indx_to_predict = np.arange(0, len(smiles))


        elif p_name_ == 'jak3_clf':  # binds/doesn't bind to jak3
            dstnctv_rwds = np.zeros((len(prop),), dtype=np.float32)

            dstnctv_rwds[np.where(prop == 0)] = 0.5

            rwds.append(dstnctv_rwds)

            indx_to_predict = np.where(prop == 1.)[0]

        elif p_name_ == 'jak3_reg':  # jak3
            if len(indx_to_predict) > 0:

                rwds[-1][np.array(indx_to_predict)] = -0.5 * np.maximum(np.full((len(indx_to_predict),), -1), np.exp((prop - 5) / 3.))

            indx_to_predict = np.arange(0, len(smiles))


        elif p_name_ == 'logP':
            dstnctv_rwds = np.zeros((len(prop),), dtype=np.float32)
            dstnctv_rwds[(prop > 1.) & (prop < 4.)] = 2.
            rwds.append(dstnctv_rwds)


        elif p_name_ == 'mpC':
            p = (prop * 93.92472573003356) + 92.0924114641032
            dstnctv_rwds = np.zeros((len(p),), dtype=np.float32)
            dstnctv_rwds[(p > 50.) & (p < 250.)] = 2.
            rwds.append(dstnctv_rwds)

        elif p_name_ == 'rings_ok':
            dstnctv_rwds = np.zeros((len(prop),), dtype=np.float32)
            dstnctv_rwds[prop == 1.] = 2.
            rwds.append(dstnctv_rwds)


        else:  # mw
            dstnctv_rwds = np.zeros((len(prop),), dtype=np.float32)
            dstnctv_rwds[(prop > 180.) & (prop < 450.)] = 2.
            dstnctv_rwds[(prop < 180.)] = -1.

            rwds.append(dstnctv_rwds)

    return np.sum(np.array(rwds), axis=0), np.array(rwds).T


def get_multi_reward_ranges_multiple_ic50_smiles_clf_and_reg(args, smiles, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = []
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]
    indx_to_predict = np.arange(0, len(smiles))

    fngps = [mol2image(sm) for sm in smiles]
    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):
        if p_name_ in PROPERTY_PREDICTORS:
            mol, prop, nan_smiles = p.predict([smiles[j] for j in indx_to_predict], get_features=None)
        else:
            mol, prop, nan_smiles = p.predict([fngps[j] for j in indx_to_predict], get_features=None)

        if len(nan_smiles) > 0:
            print('NAN smiles in prediction')
            # return invalid_reward, [invalid_reward] * len(predictors)

        if p_name_ == 'IC50_reg':  # ic50
            rwds.append(np.exp((prop - 5) / 3.))

        elif p_name_ == 'jak2_clf':  # binds/doesn't bind to jak1
            dstnctv_rwds = np.zeros((len(prop),), dtype=np.float32)
            dstnctv_rwds[np.where(prop == 0)] = 0.5
            rwds.append(dstnctv_rwds)

            indx_to_predict = np.where(prop == 1.)[0]
        elif p_name_ == 'jak2_reg':  # jak1
            if len(indx_to_predict) > 0:
                rwds[-1][np.array(indx_to_predict)] = -0.5 * np.maximum(np.full((len(indx_to_predict),), -1), np.exp((prop - 5) / 3.))
            indx_to_predict = np.arange(0, len(smiles))


        elif p_name_ == 'jak3_clf':  # binds/doesn't bind to jak3
            dstnctv_rwds = np.zeros((len(prop),), dtype=np.float32)

            dstnctv_rwds[np.where(prop == 0)] = 0.5

            rwds.append(dstnctv_rwds)

            indx_to_predict = np.where(prop == 1.)[0]

        elif p_name_ == 'jak3_reg':  # jak3
            if len(indx_to_predict) > 0:

                rwds[-1][np.array(indx_to_predict)] = -0.5 * np.maximum(np.full((len(indx_to_predict),), -1), np.exp((prop - 5) / 3.))

            indx_to_predict = np.arange(0, len(smiles))


        elif p_name_ == 'logP':
            dstnctv_rwds = np.zeros((len(prop),), dtype=np.float32)
            dstnctv_rwds[(prop > 1.) & (prop < 4.)] = 2.
            rwds.append(dstnctv_rwds)


        elif p_name_ == 'mpC':
            p = (prop * 93.92472573003356) + 92.0924114641032
            dstnctv_rwds = np.zeros((len(p),), dtype=np.float32)
            dstnctv_rwds[(p > 50.) & (p < 250.)] = 2.
            rwds.append(dstnctv_rwds)


        else:  # mw
            dstnctv_rwds = np.zeros((len(prop),), dtype=np.float32)
            dstnctv_rwds[(prop > 180.) & (prop < 450.)] = 2.
            dstnctv_rwds[(prop < 180.)] = -1.

            rwds.append(dstnctv_rwds)

    return np.sum(np.array(rwds), axis=0), np.array(rwds).T


def calc_sim_ind(sm_list, sim_fngp, normalize_fps):
    avr_si = 0
    # smiles_with_si = {}
    for sm in sm_list:
        if normalize_fps:
            si = np.sum(normalize_fp(get_ECFP(sm)) * sim_fngp)
        else:
            si = np.sum(get_ECFP(sm) * sim_fngp)
        avr_si += si
        # smiles_with_si[sm] = round(si, 3)

    avr_si = round(avr_si / len(sm_list), 3)

    return avr_si


def get_multi_reward_ranges_max_ic50_smiles(args, smiles, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = []
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]


    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):

        mol, prop, nan_smiles = p.predict([smiles], get_features=get_features)

        if len(nan_smiles) == 1:
            return invalid_reward, [invalid_reward] * len(predictors)
        if p_name_ == 'IC50':  # ic50
            rwds.append(np.exp((prop[0] - 5) / 3.))
        elif p_name_ == 'logP':  # logp
            rwds.append(2. if ((prop[0] > 1.) and (prop[0] < 4.)) else 0.)
        elif p_name_ == 'mpC':  # mpt  ??????? == mpC ?????
            p = (prop[0] * 93.92472573003356) + 92.0924114641032
            rwds.append(2. if ((p >= 50.) and (p <= 250.0)) else 0.)
        else:  # mw
            if ((prop[0] > 180.) and (prop[0] < 450.)):
                rwds.append(2)
            elif prop[0] < 180.:
                rwds.append(-1.)
            else:
                rwds.append(0.)

    return np.sum(rwds), rwds


def get_reward_max_ic50_smiles(rl, args, smiles, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = []

    for i, p in enumerate(predictors):

        mol, prop, nan_smiles = p.predict([smiles], get_features=get_features)

        if len(nan_smiles) == 1:
            return invalid_reward, [invalid_reward] * len(predictors)
        rwds.append(np.exp((prop[0] - 5) / 3.))
    #         rwds.append(np.exp(prop[0] / 3))

    return np.sum(rwds), rwds


def get_empty_reward(rl, args, mols, fngps, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = [1] * len(predictors)
    return np.sum(rwds), rwds

def get_reward_func(args):
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    metric = args.reward_func

    if metric == 'reward_max_ic50_smiles': return get_reward_max_ic50_smiles

    if metric == 'multi_reward_ranges_multiple_ic50': return get_multi_reward_ranges_multiple_ic50
    if metric == 'multi_reward_ranges_multiple_ic50_smiles': return get_multi_reward_ranges_multiple_ic50_smiles

    if metric == 'multi_reward_ranges_multiple_ic50_smiles_clf_and_reg': return get_multi_reward_ranges_multiple_ic50_smiles_clf_and_reg

    if metric == 'multi_reward_ranges_multiple_ic50_smiles_clf_and_reg_rings': return  get_multi_reward_ranges_multiple_ic50_smiles_clf_and_reg_rings
    if metric == 'empty_reward': return get_empty_reward
    else: raise ValueError(f'Metric "{metric}" not supported.')


def get_end_of_batch_reward_func(args):
    metric = args.end_of_batch_reward_func
    if metric == 'tan_similarity': return get_end_of_batch_reward_tanimoto_sim
    if metric == 'lev_similarity': return get_end_of_batch_reward_levenstein_sim
    if metric == 'tan_lev_similarity': return get_end_of_batch_reward_tanimoto_levenstein_sim
    else:
        return None

