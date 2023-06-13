# Librairie

# Our lib
import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

# Import des fonctions de pré-traitement
from pre_processing.pre_process import pre_process
# Import des modèles
from model.rnn_model import *
from model.utils_model import *
from model.xgboost_model import *

if __name__ == '__main__':
    print("Hello world !")
