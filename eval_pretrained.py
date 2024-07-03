import torch
import argparse
import os
from data.data_manager import BRSETManager
import timm
from utils import test
import time
import pickle

parser = argparse.ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###### DATA AND SAVE SETTINGS ######
parser.add_argument("--model", type=str, default="")
parser.add_argument("--save-path", type=str, default="")

###### EVALUATION SETTINGS ######
parser.add_argument("--e-learning-rate", type=float, default=0.0001)
parser.add_argument("--e-l2", type=float, default=0)
parser.add_argument("--e-runs", type=int, default=400)


TRAINING_CLASSES = ['diabetic_retinopathy',
                        'scar', 'amd', 'hypertensive_retinopathy', 'drusens', 
                        'myopic_fundus', 'increased_cup_disc', 'other']
TEST_CLASSES = ['hemorrhage', 'vascular_occlusion', 'nevus', 'healthy']


def main():
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.model == "resnet50.a3_in1k" or args.model == "swin_s3_tiny_224.ms_in1k": 
        mean_val, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif args.model == "vit_small_patch32_224.augreg_in21k_ft_in1k":
        mean_val, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    n_updates = 52

    results = {}

    # EVALUATION LOOP
    for ways in [2,3]:
        batch_size = 5 * ways

        for shots in [5,10,20]:
            epochs = n_updates * 5 // shots
            print(f"Starting {ways}-way-{shots}-shot - {epochs} epochs")

            results[f"{ways}-way-{shots}-shot"] = []
            manager = BRSETManager(TRAINING_CLASSES, TEST_CLASSES, shots, ways, mean_val, std, augment = None, batch_size = batch_size)

            for run in range(args.e_runs):
                last_time = time.time()
                model = timm.create_model(args.model, pretrained=True, num_classes=ways)
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

                        with torch.no_grad(): 
                            _, test_acc = test(test_loader, model, criterion, device)
                            run_test_accs.append(test_acc)
                
                max_acc = run_test_accs[-1] 
                results[f"{ways}-way-{shots}-shot"].append(max_acc)
                print(f"[{run + 1}/{args.e_runs}] - {ways}-way-{shots}-shot - Acc: {max_acc} - { time.time() - last_time} s - {class_names}", flush = True)
                

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