import numpy as np
from utils import get_fp, get_fp_and_mol, get_ECFP, normalize_fp
from predictor import PROPERTY_PREDICTORS

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

def get_end_of_batch_reward_tanimoto_sim(fngps):

    avg_sim = calc_sim_ind_tanimoto(fngps)

    rwd = [-np.power(s * 4, 2) - 1 if s > 0.37 else 0 for s in avg_sim]

    return rwd



def get_multi_reward_ranges_multiple_ic50(rl, args, mols, fngps, predictors, get_features, invalid_reward=0.0):
    rwds = []
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]

    #
    # if len(invalid) > 0:
    #     return invalid_reward, [invalid_reward] * len(predictors)

    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):
        if p_name_ in PROPERTY_PREDICTORS:
            mol, prop, nan_smiles = p.predict(mols, get_features=get_features[i])
        else:
            mol, prop, nan_smiles = p.predict(fngps, get_features=get_features[i])

        if len(nan_smiles) == 1:
            return invalid_reward, [invalid_reward] * len(predictors)

        if p_name_ == 'IC50':  # ic50 to jak2
            rwds.append(np.exp((prop[0] - 5) / 3.))

        elif p_name_ == 'jak1_binary': #binds/doesn't bind to jak1
            rwds.append(0. if (prop[0] == 0.) else -2.)

        elif p_name_ == 'logP':  # logp
            rwds.append(2. if ((prop[0] > 1.) and (prop[0] < 4.)) else 0.)
        elif p_name_ == 'mpC':
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


def get_multi_reward_ranges_multiple_ic50_smiles(args, smiles, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = []
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]


    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):

        mol, prop, nan_smiles = p.predict([smiles], get_features=get_features)

        if len(nan_smiles) == 1:
            return invalid_reward, [invalid_reward] * len(predictors)
        if p_name_ == 'IC50':  # ic50
            rwds.append(np.exp((prop[0] - 5) / 3.))
        elif p_name_ == 'jak1_binary': #binds/doesn't bind to jak1
            rwds.append(0. if (prop[0] == 0.) else -2.)
        elif p_name_ == 'jak3_binary': #binds/doesn't bind to jak1
            rwds.append(0. if (prop[0] == 0.) else -2.)
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

def get_multi_reward_ranges_max_ic50_similarity_penalty(rl, args, smiles, predictors, invalid_reward=0.0, get_features=get_fp_and_mol):
    rwds = []
    stats = [[6.571506827780664, 0.405556162912055]]  # ic50
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]

    fngps, mols, _, invalid = get_fp_and_mol([smiles])
    if len(invalid) > 0:
        return invalid_reward, [invalid_reward] * len(predictors)

    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):
        if p_name_ in PROPERTY_PREDICTORS:
            mol, prop, nan_smiles = p.predict(mols, get_features=None)
        else:
            mol, prop, nan_smiles = p.predict(fngps, get_features=None)

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
            rwds.append(2. if ((prop[0] > 180.) and (prop[0] < 450.)) else 0.)


    sim = calc_sim_ind(mols, rl.similarity_fingerprint, False)
    if sim < 0.4:
        rwds.append(0)

    else:
        rwds.append(-np.power(sim * 4, 2) - 6)

    return np.sum(rwds), rwds

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


def get_multi_reward_ranges_max_ic50(rl, args, mols, fngps, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = []
    stats = [[6.571506827780664, 0.405556162912055]]  # ic50
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]

    # fngps, mols, _, invalid = get_fp_and_mol([smiles])
    # if len(invalid) > 0:
    #     return invalid_reward, [invalid_reward] * len(predictors)

    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):
        if p_name_ in PROPERTY_PREDICTORS:
            mol, prop, nan_smiles = p.predict(mols, get_features=None)
        else:
            mol, prop, nan_smiles = p.predict(fngps, get_features=None)

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


def get_multi_reward_with_mpt(args, smiles, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = []
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]

    stats = [[6.571506827780664, 0.405556162912055], [2.3744083807347933, 0.9429912852261682],
             [153.76311716715495, 33.472182703355514]]

    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):
        mol, prop, nan_smiles = p.predict([smiles], get_features=get_features)

        if len(nan_smiles) == 1:
            return invalid_reward, [invalid_reward] * len(predictors)
        if p_name_ == 'mpC':  # mpt  ??????? == mpC ?????
            p = (prop[0] * 93.92472573003356) + 92.0924114641032
            rwds.append(np.exp((p - stats[i][0]) / stats[i][1]))
        elif p_name_ == 'IC50':  # ic50
            rwds.append(np.exp(-(prop[0] - stats[i][0]) / stats[i][1]))
        else:
            rwds.append(np.exp((prop[0] - stats[i][0]) / stats[i][1]))
    return np.sum(rwds), rwds


def get_multi_reward_with_mw(args, smiles, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = []
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]

    stats = [[6.571506827780664, 0.405556162912055], [2.3744083807347933, 0.9429912852261682]]

    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):
        # FIX WITH I=... , but name of predictors -->
        mol, prop, nan_smiles = p.predict([smiles], get_features=get_features)

        if len(nan_smiles) == 1:
            return invalid_reward, [invalid_reward] * len(predictors)
        if p_name_ == 'mw':  # mw
            rwds.append(2. if (prop[0] > 180. and prop[0] < 450.) else 0.)
        elif p_name_ == 'IC50':  # ic50
            rwds.append(np.exp(-(prop[0] - stats[i][0]) / stats[i][1]))
        else:
            rwds.append(np.exp((prop[0] - stats[i][0]) / stats[i][1]))
    return np.sum(rwds), rwds


def get_multi_reward(args, smiles, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = []
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]
    stats = [[6.571506827780664, 0.405556162912055], [2.3744083807347933, 0.9429912852261682],
             [153.76311716715495, 33.472182703355514]]
    #     stats = [[0,1]]
    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):
        print(smiles)
        mol, prop, nan_smiles = p.predict([smiles], get_features=get_features)

        if len(nan_smiles) == 1:
            return invalid_reward, [invalid_reward] * len(predictors)
        rwds.append(np.exp((prop[0] - stats[i][0]) / stats[i][1]))
    return np.sum(rwds), rwds


def get_range_reward(args, smiles, predictors, invalid_reward=0.0, get_features=get_fp):

    predictors_names = [i['name'] for i in args.objectives_names_and_paths]
    rwds = []
    for p_name_, p in zip(predictors_names, predictors):

        mol, prop, nan_smiles = p.predict([smiles], get_features=get_features)

        if len(nan_smiles) == 1:
            return invalid_reward, [invalid_reward] * len(predictors)
        rwds.append(2. if (prop[0] > 180. and prop[0] < 450.) else 0.)
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

def get_reward_max_ic50(rl, args, mols, fngps, predictors, invalid_reward=0.0, get_features=get_fp):
    rwds = []
    predictors_names = [i['name'] for i in args.objectives_names_and_paths]

    for i, p_name_, p in zip(range(len(predictors_names)), predictors_names, predictors):
        if p_name_ in PROPERTY_PREDICTORS:
            mol, prop, nan_smiles = p.predict(mols, get_features=None)
        else:
            mol, prop, nan_smiles = p.predict(fngps, get_features=None)

        if len(nan_smiles) == 1:
            return invalid_reward, [invalid_reward] * len(predictors)

        if p_name_ == 'IC50':  # ic50
            rwds.append(np.exp((prop[0] - 5) / 3.))
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
    if metric == 'multi_reward': return get_multi_reward
    if metric == 'range_reward': return get_range_reward
    if metric == 'multi_reward_with_mw': return get_multi_reward_with_mw
    if metric == 'multi_reward_ranges_max_ic50': return get_multi_reward_ranges_max_ic50
    if metric == 'multi_reward_ranges_max_ic50_smiles': return get_multi_reward_ranges_max_ic50_smiles

    if metric == 'multi_reward_with_mpt': return get_multi_reward_with_mpt
    if metric == 'reward_max_ic50': return get_reward_max_ic50
    if metric == 'reward_max_ic50_smiles': return get_reward_max_ic50_smiles
    if metric == 'multi_reward_ranges_max_ic50_similarity_penalty': return get_multi_reward_ranges_max_ic50_similarity_penalty
    if metric == 'multi_reward_ranges_multiple_ic50': return get_multi_reward_ranges_multiple_ic50
    if metric == 'multi_reward_ranges_multiple_ic50_smiles': return get_multi_reward_ranges_multiple_ic50_smiles

    if metric == 'empty_reward': return get_empty_reward
    else: raise ValueError(f'Metric "{metric}" not supported.')


def get_end_of_batch_reward_func(args):
    metric = args.end_of_batch_reward_func
    if metric == 'tan_similarity': return get_end_of_batch_reward_tanimoto_sim
    else:
        return None

