import torch
import argparse
import os
from data.data_manager import BRSETManager
import timm
from utils import test
from tqdm import tqdm
from copy import deepcopy
from torchvision.transforms import v2

parser = argparse.ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###### EXPERIMENT SETTINGS ######
parser.add_argument("--shots", type=int, default=10)
parser.add_argument("--ways", type=int, default=2)
parser.add_argument("--model", type=str, default="swin_s3_tiny_224.ms_in1k")
parser.add_argument("--augment", type=str, default=None)
parser.add_argument("--augment-where", type=str, default='')

###### REPTILE SETTINGS ######
parser.add_argument("--inner-loop-learning-rate", type=float, default=0.0001)
parser.add_argument("--inner-loop-steps", type=int, default=4)
parser.add_argument("--outer-loop-learning-rate", type=float, default=0.1)
parser.add_argument("--outer-loop-updates", type=int, default=1000)
parser.add_argument("--outer-loop-batch-size", type=int, default=5)
parser.add_argument("--alpha", type=float, default=1.0)

###### DATA AND SAVE SETTINGS ######
parser.add_argument("--save-path", type=str, default="")

###### EVALUATION SETTINGS ######
parser.add_argument("--e-learning-rate", type=float, default=0.0001)
parser.add_argument("--e-runs", type=int, default=400)


TRAINING_CLASSES = ['diabetic_retinopathy',
                        'scar', 'amd', 'hypertensive_retinopathy', 'drusens', 
                        'myopic_fundus', 'increased_cup_disc', 'other']
TEST_CLASSES = ['hemorrhage', 'vascular_occlusion', 'nevus', 'healthy']

def main():
    args = parser.parse_args()
    print(args)

    inner_loop_batch_size = 5 * args.ways
    inner_loop_epochs = 5 * args.inner_loop_steps //args.shots 

    print("inner loop batch size", inner_loop_batch_size)
    print("inner loop epochs", inner_loop_epochs)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Carregar modelo
    print("Loading model...", flush=True)
    
    model = timm.create_model(args.model, pretrained=True, num_classes=args.ways)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.model == "resnet50.a3_in1k" or args.model == "swin_s3_tiny_224.ms_in1k": 
        mean_val, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif args.model == "vit_small_patch32_224.augreg_in21k_ft_in1k":
        mean_val, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif args.model == "dumb":
        mean_val, std = (0,0,0),(1,1,1)


    # Carregar dados
    manager = BRSETManager(TRAINING_CLASSES, TEST_CLASSES, args.shots, args.ways, mean_val, std, augment = args.augment, batch_size = inner_loop_batch_size)
    model.train()
    model.to(device)

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    if  args.augment == "mixup":
        aug = v2.MixUp(alpha = args.alpha, num_classes = args.ways)
    elif args.augment == "cutmix":
        aug = v2.CutMix(alpha=args.alpha, num_classes=args.ways)

    # REPTILE LOOP
    for iteration in tqdm(range(args.outer_loop_updates)):
        weights_before = deepcopy(model.state_dict())
        updates = []
        # Generate task
        for _ in range(args.outer_loop_batch_size):
            model.load_state_dict(weights_before)
            train_task = manager.get_train_task()
            # Do ADAM on this task
            optimizer = torch.optim.Adam(model.parameters(), lr=args.inner_loop_learning_rate)
            for _ in range(inner_loop_epochs):
                for image, label in train_task:
                    image = image.to(device)
                    label = label.to(device)
                    
                    # COMMENT NEXT TWO LINES TO DISABLE TRAIN AUG
                    if 'i' in args.augment_where and (args.augment == "mixup" or args.augment == "cutmix"):
                        image,label = aug(image,torch.argmax(label, dim=1))

                    optimizer.zero_grad()
                    out = model(image)
                    
                    loss = criterion(out, label)
                    loss.backward()
                    optimizer.step()
            weights_after = model.state_dict()
            updates.append({name : (weights_after[name] - weights_before[name])  
                        for name in weights_before})
        updated ={}
        with torch.no_grad():
            outerstepsize = args.outer_loop_learning_rate* (1 - iteration / args.outer_loop_updates) # linear schedule
            for name in weights_before:
                grads = sum([w[name] for w in updates]) / len(updates)
                updated[name] = weights_before[name] + outerstepsize * grads
        model.load_state_dict(updated)

    torch.save({
            'model_state_dict': model.state_dict()
            }, args.save_path + "/last_checkpoint.pth")
    # EVALUATION

    reptile_state_dict = deepcopy(model.state_dict())
    n_updates = 52
    epochs = n_updates * 5 // args.shots

    run_test_accs = []

    for run in tqdm(range(args.e_runs)):

        model.load_state_dict(reptile_state_dict)
        train_loader, test_loader, class_names = manager.get_eval_task()
        criterion = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.e_learning_rate)

        for epoch in range(epochs):
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                if 'e' in args.augment_where and (args.augment == "mixup" or args.augment == "cutmix"):
                    images,labels = aug(images,torch.argmax(labels, dim=1))

                opt.zero_grad()
                out = model(images)
                loss = criterion(out, labels)
                loss.backward()
                opt.step()

        with torch.no_grad():
            _, test_acc = test(test_loader, model, criterion, device)
            run_test_accs.append(test_acc)
            print(class_names, test_acc, flush=True)

    print("FINAL ACC", sum(run_test_accs) / args.e_runs)
        
    

if __name__ == "__main__":
    main()
