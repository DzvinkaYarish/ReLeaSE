from running_functions import run
from parsing import parsing
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

args = parsing()
run(args)