import pandas as pd
import sys
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms.functional import pad, resize
from tqdm import tqdm

CLASS_COLUMNS = ['hemorrhage', 'vascular_occlusion','diabetic_retinopathy',
                 'macular_edema', 'scar', 'nevus', 'amd', 
                 'hypertensive_retinopathy', 'drusens', 
                 'myopic_fundus', 'increased_cup_disc', 
                 'retinal_detachment', 'other']

BRSET_PATH = "YOUR_DATA_PATH" # '/scratch/diogo.alves/datasets/brset/physionet.org/files/brazilian-ophthalmological/1.0.0/fundus_photos/'
LABELS_PATH = "YOUR_LABELS_PATH" # labels.csv

def clean():
    """
    Removes unadequate images and samples 100 healthy ones to compose healthy class.
    """
    brset = pd.read_csv(LABELS_PATH)
    brset = brset[(brset.quality == "Adequate") & (brset[CLASS_COLUMNS].sum(axis=1) <= 1)]
    brset['healthy'] = 0
    brset['to_augment'] = 0

    no_cond = brset[brset[CLASS_COLUMNS].sum(axis=1) == 0]
    healthy = no_cond.sample(n=100, random_state=42)
    to_augment = no_cond[~no_cond.image_id.isin(healthy.image_id)]

    brset.loc[brset.image_id.isin(healthy.image_id), 'healthy'] = 1
    brset.loc[brset.image_id.isin(to_augment.image_id), 'to_augment'] = 1

    brset.to_csv("clean.csv")

def count_diseases_before():
    brset = pd.read_csv(LABELS_PATH)

    print("total", len(brset))

    print("######### CLASSES #########")
    for disease in CLASS_COLUMNS: #+ ['healthy', 'to_augment']:
        print(disease, len(brset[brset[disease] == 1]))
    print("###########################")
    
    print("no_cond", len(brset[brset[CLASS_COLUMNS].sum(axis=1) == 0]))

def count_diseases_after():
    brset = pd.read_csv("clean.csv")

    print("total", len(brset))

    print("######### CLASSES #########")
    for disease in CLASS_COLUMNS + ['healthy', 'to_augment']:
        print(disease, len(brset[brset[disease] == 1]))
    print("###########################")
    
    print("no_cond", len(brset[brset[CLASS_COLUMNS].sum(axis=1) == 0]))


def get_padding(image):
    imsize = image.size()
    max_h = max_w = imsize[1] if imsize[1] > imsize[2] else imsize [2] 
    
    h_padding = (max_w - imsize[2]) / 2
    v_padding = (max_h - imsize[1]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    
    return padding

def do_processing(img):
    cond_1 = (img.mean(axis=(0,2)) >= 3/255)
    cond_2 = (img.mean(axis=(0,1)) >= 3/255)
    bless_img = img[:, cond_1, :][:,:,cond_2]
    
    padded_img = pad(bless_img, get_padding(img))
    
    resized_img = resize(padded_img, [224,224])
    return resized_img


def preprocess():
    brset = pd.read_csv("clean.csv")
    for path in tqdm(brset["image_id"]):
        img = read_image(BRSET_PATH + path + ".jpg")/255
        proc_img = do_processing(img)
        save_image(proc_img, "./imgs/" + path + ".jpg")



if __name__ == "__main__":

    if sys.argv[1] == "p":
        preprocess()
    elif sys.argv[1] == "c":
        clean()
    elif sys.argv[1] == "cdb":
        count_diseases_before()
    elif sys.argv[1] == "cda":
        count_diseases_after()

    