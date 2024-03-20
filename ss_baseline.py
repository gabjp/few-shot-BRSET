import torch
import argparse
import os
from data_2.data_manager import BRSETManager
import torchvision.models as models
import torch.nn as nn
from transformers import ViTForImageClassification
import timm
from utils import test, save_acc, save_loss
import time
import pickle

parser = argparse.ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###### TRAINING SETTINGS ######
parser.add_argument("--p-learning-rate", type=float, default=0.001)
parser.add_argument("--p-epochs", type=int, default=15)
parser.add_argument("--p-batch-size", type=int, default=16)
parser.add_argument("--p-l2", type=float, default=0)

###### DATA AND SAVE SETTINGS ######
parser.add_argument("--model", type=str, default="")
parser.add_argument("--save-path", type=str, default="")

###### EVALUATION SETTINGS ######
parser.add_argument("--e-learning-rate", type=float, default=0.001)
#parser.add_argument("--e-epochs", type=int, default=20)
#parser.add_argument("--e-batch-size", type=int, default=80)
parser.add_argument("--e-l2", type=float, default=0)
parser.add_argument("--e-runs", type=int, default=50)
#parser.add_argument("--e-saving-runs", type=int, default=400)

TRAINING_CLASSES = ['diabetic_retinopathy',
                        'scar', 'amd', 'hypertensive_retinopathy', 'drusens', 
                        'myopic_fundus', 'increased_cup_disc', 'other']
TEST_CLASSES = ['hemorrhage', 'vascular_occlusion', 'nevus', 'healthy']


def compute_class_weights(loader):
    num_objs = 0
    class_count = torch.tensor([0 for _ in range(len(TRAINING_CLASSES))]).float()

    for _, label in loader:
        num_objs += label.size()[0]
        class_count += label.sum(axis=0)
    
    weights = num_objs /  (class_count * len(TRAINING_CLASSES)) # Use 1/ (freq * n_classes) as weights

    print(f"Class weights: {weights}")
    return weights


def main():
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Carregar modelo
    print("Loading model...", flush=True)
    model = timm.create_model(args.model, pretrained=True, num_classes=len(TRAINING_CLASSES))

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.model == "resnet50.a3_in1k" or args.model == "swin_s3_tiny_224.ms_in1k": 
        mean_val, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif args.model == "vit_small_patch32_224.augreg_in21k_ft_in1k":
        mean_val, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # Carregar dados
    manager = BRSETManager(TRAINING_CLASSES, TEST_CLASSES, -1, -1, mean_val, std, augment = None, batch_size = args.p_batch_size)
    train_loader, val_loader = manager.get_ss_split()
    class_weights = compute_class_weights(train_loader).float().to(device)
    

    # Training setup
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.p_learning_rate, weight_decay=args.p_l2)
    
    model.train()
    model.to(device)

    # Training Loop
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    best_loss = float("inf")

    for epoch in range(args.p_epochs):
        last_time = time.time()

        epoch_train_loss = 0
        total_correct = 0
        total_samples = 0
        num_batch_count = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            out = model(images)

            loss = criterion(out, labels)

            loss.backward()
            opt.step()

            predicted = torch.argmax(out, axis=1)

            epoch_train_loss += loss.item()
            total_correct += (predicted == torch.argmax(labels, axis=1)).sum().item()
            total_samples += labels.size(0)
            num_batch_count +=1

        with torch.no_grad():
        
            print(f"[{epoch+1}/{args.p_epochs}] - Training loss: {epoch_train_loss/num_batch_count} - Training accuracy: {total_correct / total_samples * 100}")
            train_loss.append(epoch_train_loss/num_batch_count)
            train_acc.append(total_correct / total_samples * 100)

            val_loss, val_acc = test(val_loader, model, criterion, device)
            print(f"[{epoch+1}/{args.p_epochs}] - Validation loss: {val_loss} - Validation accuracy: {val_acc}", flush=True)
            test_loss.append(val_loss)
            test_acc.append(val_acc)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': criterion,
                }, args.save_path + "/best_checkpoint.pth")
        
        print(f"[{epoch+1}/{args.p_epochs}] - Epoch took {time.time() - last_time} seconds", flush = True)

    # END TRAINING LOOP
        
    # SAVE MODEL 
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': criterion,
            }, args.save_path + "/last_checkpoint.pth")

    # SAVE PLOTS
    save_acc(args.save_path, train_acc, test_acc)
    save_loss(args.save_path, train_loss, test_loss)

    # SAVE LISTS
    with open(args.save_path + '/train_acc.pkl', 'wb') as f:
        pickle.dump(train_acc, f)
    with open(args.save_path + '/test_acc.pkl', 'wb') as f:
        pickle.dump(test_acc, f)
    with open(args.save_path + '/train_loss.pkl', 'wb') as f:
        pickle.dump(train_loss, f)
    with open(args.save_path + '/test_loss.pkl', 'wb') as f:
        pickle.dump(test_loss, f)
        
    
    # LOAD BEST MODEL ...
    # Avaliar o modelo: Salvar (curvas, checkpoints e valores) de alguns

    n_updates = 52

    results = {}

    # EVALUATION LOOP
    for ways in [2,3]:
        batch_size = 5 * ways
        model = timm.create_model(args.model, pretrained=False, num_classes=ways)

        for shots in [5,10,20]:
            epochs = n_updates * 5 // shots
            print(f"Starting {ways}-way-{shots}-shot - {epochs} epochs")

            results[f"{ways}-way-{shots}-shot"] = []
            manager = BRSETManager(TRAINING_CLASSES, TEST_CLASSES, shots, ways, mean_val, std, augment = None, batch_size = batch_size)

            for run in range(args.e_runs):
                last_time = time.time()

                checkpoint = torch.load(args.save_path + "/best_checkpoint.pth")['model_state_dict']
                current_model_dict = model.state_dict()
                checkpoint = {k:v if v.size()==current_model_dict[k].size()  
                                  else  current_model_dict[k] 
                                  for k,v in checkpoint.items()}
                model.load_state_dict(checkpoint, strict=False)
                model.to(device)

                train_loader, test_loader, class_names = manager.get_eval_task()

                criterion = torch.nn.CrossEntropyLoss()
                opt = torch.optim.Adam(model.parameters(), lr=args.e_learning_rate, weight_decay=args.e_l2)

                run_test_accs = []

                for epoch in range(epochs):
                    for images, labels in train_loader:
                        
                        images = images.to(device)
                        labels = labels.to(device)

                        opt.zero_grad()
                        out = model(images)

                        loss = criterion(out, labels)

                        loss.backward()
                        opt.step()

                        _, test_acc = test(test_loader, model, criterion, device)
                        run_test_accs.append(test_acc)
                
                max_acc = max(run_test_accs)
                results[f"{ways}-way-{shots}-shot"].append(max_acc)
                print(f"[{run + 1}/{args.e_runs}] - {ways}-way-{shots}-shot - Acc: {max_acc} - { time.time() - last_time} s - {class_names}", flush = True)
                #print(run_test_accs)

            final_acc = sum(results[f"{ways}-way-{shots}-shot"])/len(results[f"{ways}-way-{shots}-shot"])
            print(f"{ways}-way-{shots}-shot Average Acc: {final_acc}", flush = True)

    
    with open(args.save_path + '/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("############################ SUMMARY ############################")
    for key, value in results.items():
        print(f"{key} - Acc: {sum(value)/len(value)}")


    return

if __name__ == "__main__":
    main()