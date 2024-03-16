import torch
import argparse
import os
from data_2.data_manager import BRSETManager
import torchvision.models as models
import torch.nn as nn
from transformers import ViTForImageClassification
import timm

parser = argparse.ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###### TRAINING SETTINGS ######
parser.add_argument("--p-optimizer", type=str, default="adam")
parser.add_argument("--p-learning-rate", type=float, default=0.001)
parser.add_argument("--p-epochs", type=int, default=20)
parser.add_argument("--p-batch-size", type=int, default=80)
parser.add_argument('--p-schedule', action='store_true')
parser.add_argument("--p-l2", type=float, default=0)

###### DATA AND SAVE SETTINGS ######
parser.add_argument("--model", type=str, default="")
parser.add_argument("--tasks", type=str, default="")
parser.add_argument("--train-tasks", type=str, default="")
parser.add_argument("--save-path", type=str, default="")

###### EVALUATION SETTINGS ######
parser.add_argument("--e-optimizer", type=str, default="adam")
parser.add_argument("--e-learning-rate", type=float, default=0.001)
parser.add_argument("--e-epochs", type=int, default=20)
parser.add_argument("--e-batch-size", type=int, default=80)
parser.add_argument('--e-schedule', action='store_true')
parser.add_argument("--e-l2", type=float, default=0)
parser.add_argument("--e-runs", type=int, default=400)
parser.add_argument("--e-saving-runs", type=int, default=5)

TRAINING_CLASSES = ['diabetic_retinopathy',
                        'scar', 'amd', 'hypertensive_retinopathy', 'drusens', 
                        'myopic_fundus', 'increased_cup_disc', 'other']
TEST_CLASSES = ['hemorrhage', 'vascular_occlusion', 'nevus', 'healthy']

def main():
    args = parser.parse_args()
    print(args)
    # Carregar dados
    manager = BRSETManager(TRAINING_CLASSES, TEST_CLASSES, -1, -1, augment = None, batch_size = args.p_batch_size)

    # Carregar modelo
    model = timm.create_model(args.model, pretrained=True, num_classes=len(TRAINING_CLASSES))

    # Pré-processar os dados de acordo com o modelo
    # Pré-treinar o modelo: Salvar curvas, best checkpoint, last checkpoint e lista de números com os valores
    # Avaliar o modelo: Salvar (curvas, checkpoints e valores) de alguns

    return

if __name__ == "__main__":
    main()