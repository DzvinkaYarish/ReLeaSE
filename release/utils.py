import csv
import time
import math
import numpy as np
import warnings
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw
DrawingOptions.atomLabelFontSize = 50
DrawingOptions.dotsPerAngstrom = 100
DrawingOptions.bondLineWidth = 3

from sklearn.model_selection import KFold, StratifiedKFold

def get_fp(smiles):
    fp = []
    processed_indices = []
    invalid_indices = []
    for i in range(len(smiles)):
        mol = smiles[i]
        tmp = np.array(mol2image(mol, n=2048))
        if np.isnan(tmp[0]):
            invalid_indices.append(i)
        else:
            fp.append(tmp)
            processed_indices.append(i)
    return np.array(fp), processed_indices, invalid_indices

def get_fp_and_mol(smiles, max_path=4):
    fp = []
    processed_indices = []
    invalid_indices = []
    mols = []
    for i in range(len(smiles)):
        mol = smiles[i]
        mols.append(mol)
        tmp = np.array(mol2image(mol, n=2048, max_path=max_path))
        if np.isnan(tmp[0]):
            invalid_indices.append(i)
        else:
            fp.append(tmp)
            processed_indices.append(i)
    return np.array(fp), mols, processed_indices, invalid_indices

def get_desc(smiles, calc):
    desc = []
    processed_indices = []
    invalid_indices = []
    for i in range(len(smiles)):
        sm = smiles[i]
        try:
            mol = Chem.MolFromSmiles(sm)
            tmp = np.array(calc(mol))
            desc.append(tmp)
            processed_indices.append(i)
        except:
            invalid_indices.append(i)

    desc_array = np.array(desc)
    return desc_array, processed_indices, invalid_indices


def get_ECFP(smiles):
    if isinstance(smiles, str):
        try:
            m = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, invariants=[1]*m.GetNumAtoms())
            array = np.zeros((0, ), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            return array
        except Exception as e:
            # print(e)
            empty = np.zeros(2048)
            # empty[:] = np.nan
            return empty
    else:
        try:
            m = smiles
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, invariants=[1] * m.GetNumAtoms())
            array = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            return array
        except Exception as e:
        # print(e)
            empty = np.zeros(2048)
            # empty[:] = np.nan
            return empty


def normalize_fp(fp):
    if np.sum(fp) == 0:
        return fp
    return fp / np.sum(fp)

def normalize_desc(desc_array, desc_mean=None):
    desc_array = np.array(desc_array).reshape(len(desc_array), -1)
    ind = np.zeros(desc_array.shape)
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            try:
                if np.isfinite(desc_array[i, j]):
                    ind[i, j] = 1
            except:
                pass
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            if ind[i, j] == 0:
                desc_array[i, j] = 0
    if desc_mean is None:
        desc_mean = np.mean(desc_array, axis=0)
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            if ind[i, j] == 0:
                desc_array[i, j] = desc_mean[j]
    return desc_array, desc_mean


def mol2image(x, n=2048, max_path=4):
    if isinstance(x, str):
        m = Chem.MolFromSmiles(x)
        if not m:
            return [np.nan]
    else:
        m = x

    fp = Chem.RDKFingerprint(m, maxPath=max_path, fpSize=n)
    res = np.zeros(len(fp), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, res)
    return res




def sanitize_smiles(smiles, canonical=True, throw_warning=False):
    """
    Takes list of SMILES strings and returns list of their sanitized versions.
    For definition of sanitized SMILES check
    http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol

    Parameters
    ----------
    smiles: list
        list of SMILES strings

    canonical: bool (default True)
        parameter specifying whether SMILES will be converted to canonical
        format

    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid

    Returns
    -------
    new_smiles: list
        list of SMILES and NaNs if SMILES string is invalid or unsanitized.
        If canonical is True, returns list of canonical SMILES.

    When canonical is True this function is analogous to:
        canonical_smiles(smiles, sanitize=True).
    """
    new_smiles = []
    for sm in smiles:
        try:
            if canonical:
                new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=True)))
            else:
                new_smiles.append(sm)
        except:
            if throw_warning:
                warnings.warn('Unsanitized SMILES string: ' + sm, UserWarning)
            new_smiles.append('')
    return new_smiles


def canonical_smiles(smiles, sanitize=True, throw_warning=False):
    """
    Takes list of SMILES strings and returns list of their canonical SMILES.

    Parameters
    ----------
    smiles: list
        list of SMILES strings to convert into canonical format

    sanitize: bool (default True)
        parameter specifying whether to sanitize SMILES or not.
            For definition of sanitized SMILES check
            http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol

    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid

    Returns
    -------
    new_smiles: list
        list of canonical SMILES and NaNs if SMILES string is invalid or
        unsanitized (when sanitize is True)

    When sanitize is True the function is analogous to:
        sanitize_smiles(smiles, canonical=True).
    """
    new_smiles = []
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            new_smiles.append(Chem.MolToSmiles(mol))
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
            new_smiles.append('')
    return new_smiles


def save_smi_to_file(filename, smiles, unique=True):
    """
    Takes path to file and list of SMILES strings and writes SMILES to the specified file.

        Args:
            filename (str): path to the file
            smiles (list): list of SMILES strings
            unique (bool): parameter specifying whether to write only unique copies or not.

        Output:
            success (bool): defines whether operation was successfully completed or not.
       """
    if unique:
        smiles = list(set(smiles))
    else:
        smiles = list(smiles)
    f = open(filename, 'w')
    for mol in smiles:
        f.writelines([mol, '\n'])
    f.close()
    return f.closed


def read_smi_file(filename, unique=True, add_start_end_tokens=False):
    """
    Reads SMILES from file. File must contain one SMILES string per line
    with \n token in the end of the line.

    Args:
        filename (str): path to the file
        unique (bool): return only unique SMILES

    Returns:
        smiles (list): list of SMILES strings from specified file.
        success (bool): defines whether operation was successfully completed or not.

    If 'unique=True' this list contains only unique copies.
    """
    f = open(filename, 'r')
    molecules = []
    for line in f:
        if add_start_end_tokens:
            molecules.append('<' + line[:-1] + '>')
        else:
            molecules.append(line[:-1])
    if unique:
        molecules = list(set(molecules))
    else:
        molecules = list(molecules)
    f.close()
    return molecules, f.closed


def tokenize(smiles, tokens=None):
    """
    Returns list of unique tokens, token-2-index dictionary and number of
    unique tokens from the list of SMILES

    Parameters
    ----------
        smiles: list
            list of SMILES strings to tokenize.

        tokens: list, str (default None)
            list of unique tokens

    Returns
    -------
        tokens: list
            list of unique tokens/SMILES alphabet.

        token2idx: dict
            dictionary mapping token to its index.

        num_tokens: int
            number of unique tokens.
    """
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = list(np.sort(tokens))
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def cross_validation_split(x, y, n_folds=5, split='random', folds=None):
    assert(len(x) == len(y))
    x = np.array(x)
    y = np.array(y)
    if split not in ['random', 'stratified', 'fixed']:
        raise ValueError('Invalid value for argument \'split\': '
                         'must be either \'random\', \'stratified\' '
                         'or \'fixed\'')
    if split == 'random':
        cv_split = KFold(n_splits=n_folds, shuffle=True)
        folds = list(cv_split.split(x, y))
    elif split == 'stratified':
        cv_split = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = list(cv_split.split(x, y))
    elif split == 'fixed' and folds is None:
        raise TypeError(
            'Invalid type for argument \'folds\': found None, but must be list')
    cross_val_data = []
    cross_val_labels = []
    if len(folds) == n_folds:
        for fold in folds:
            cross_val_data.append(x[fold[1]])
            cross_val_labels.append(y[fold[1]])
    elif len(folds) == len(x) and np.max(folds) == n_folds:
        for f in range(n_folds):
            left = np.where(folds == f)[0].min()
            right = np.where(folds == f)[0].max()
            cross_val_data.append(x[left:right + 1])
            cross_val_labels.append(y[left:right + 1])

    return cross_val_data, cross_val_labels


def read_object_property_file(path, delimiter=',', cols_to_read=[0, 1],
                              keep_header=False):
    f = open(path, 'r')
    reader = csv.reader(f, delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        data[i] = data_full[start_position:, col]
    f.close()
    if len(cols_to_read) == 1:
        data = data[0]
    return data

def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_dist(prediction, name, **kwargs):
    mean = kwargs.get('mean', 0)
    std = kwargs.get('std', 1)
    interval = kwargs.get('interval', None)

    prediction = prediction * std + mean
    # print("Mean value of predictions:", prediction.mean())
    ax = sns.kdeplot(prediction, shade=True)
    ax.set(xlabel=f'Predicted {name}',
           title=f'Distribution of predicted {name} for generated molecules')
    if interval:
        ax.axvline(x=interval[0], color='r')
        ax.axvline(x=interval[1], color='r')


def plot_hist(prediction, name, **kwargs):
    mean = kwargs.get('mean', 0)
    std = kwargs.get('std', 1)
    interval = kwargs.get('interval', None)

    prediction = prediction * std + mean
    plt.hist(prediction, alpha=0.5)
    plt.xlabel(f'Predicted {name}')
    plt.title(f'Distribution of predicted {name} for generated molecules')


def generate(generator, n_to_generate, gen_data, batch_size):
    generated = []
    pbar = tqdm(range(np.ceil(n_to_generate / batch_size).astype(np.int64)))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.extend(generator.evaluate(gen_data, predict_len=120, batch_size=batch_size))

    generated = [sm[1:-1] for sm in generated[:n_to_generate]]
    sanitized = canonical_smiles(generated, sanitize=True, throw_warning=False)[:-1]

    valid_num = (n_to_generate - sanitized.count(''))
    unique_smiles = list(np.unique([s for s in sanitized if s]))
    unique_num = len(unique_smiles)


    return unique_smiles, valid_num / n_to_generate, unique_num / valid_num


def estimate_and_update(generator, predictor, n_to_generate, gen_data, p_name, **kwargs):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])

    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]

    valid_num = (n_to_generate - sanitized.count(''))
    unique_smiles = list(np.unique([s for s in sanitized if s]))[1:]
    unique_num = len(unique_smiles)
    smiles, prediction, nan_smiles = predictor.predict(unique_smiles, get_features=get_fp)

    if predictor.model_type == 'classifier':
        plot_hist(prediction, p_name, **kwargs)
    else:
        plot_dist(prediction, p_name, **kwargs)

    return smiles, prediction, valid_num / n_to_generate, unique_num / valid_num


def predict_and_plot(smiles, predictor, p_name, **kwargs):
    get_features = kwargs.get('get_features')

    smiles, prediction, nan_smiles = predictor.predict(smiles, get_features=get_features)
    if len(prediction) > 0:
        if predictor.model_type == 'classifier':
            plot_hist(prediction, p_name, **kwargs)
        else:
            plot_dist(prediction, p_name, **kwargs)

    return smiles, prediction


def save_smiles(args, smiles_cur):

    path_to_experiment = args.experiments_general_path+args.experiment_name

    with open(f'{path_to_experiment}/generated_smiles.txt', 'a') as f:
        for sm in smiles_cur:
            f.write(str(sm))
            f.write('\n')
        f.write('\n')


def draw_smiles(args, smiles_cur, prediction_cur, labels):
    print(len(smiles_cur))
    print(len(prediction_cur))
    prediction_cur = np.round(prediction_cur, 2)
    ind = np.random.randint(0, len(smiles_cur), args.n_to_draw)
    mols_to_draw_max = [Chem.MolFromSmiles(smiles_cur[i], sanitize=True) for i in ind]

    legends = ['; '.join([f'{labels[j]}={prediction_cur[i][j]}' for j in range(len(labels))]) for i in ind]

    img = Draw.MolsToGridImage(mols_to_draw_max, molsPerRow=4,
                               subImgSize=(400, 400), legends=legends)
    return img