import torch 
from data.loader import FewShotBRSET
import argparse
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from utils import SmallNet, test, save_multi_plot, VGG16, DumbNet
import os
import random 
from copy import deepcopy
from utils import test

parser = argparse.ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###### INNER LOOP SETTINGS ######
parser.add_argument("--inner-optimizer", type=str, default="sgd")
parser.add_argument("--inner-learning-rate", type=float, default=0.001)
parser.add_argument("--inner-steps", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=80)

###### OUTER LOOP SETTINGS ######
parser.add_argument("--outer-learning-rate", type=float, default=0.001)
parser.add_argument("--outer-steps", type=int, default=20)

###### DATA AND SAVE SETTINGS ######
parser.add_argument("--tasks", type=str, default="")
parser.add_argument("--train-set", type=str, default="")
parser.add_argument("--save-path", type=str, default="")
parser.add_argument("--img-resize", type=str, default="256,256")


def main():
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # LOAD DATA
    img_size = tuple( [int(args.img_resize.split(',')[i]) for i in [0,1]] )
    transform = transforms.Resize(img_size, antialias=False)
    tasks = args.tasks.split(',')

    train_loaders = []
    test_loaders = []

    for task in tasks:

        train_set = FewShotBRSET(transform=transform, tasks=task, split=args.train_set)
        test_set = FewShotBRSET(transform=transform, tasks=task, split='test')

        print(task)
        print(f"TRAIN SET SIZE: {len(train_set)}")
        print(f"TEST SET SIZE: {len(test_set)}")

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        train_loaders.append(train_dataloader)
        test_loaders.append(test_dataloader)
        
    # LOAD MODEL
    model = DumbNet()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TRAINABLE PARAMETERS: {n_trainable}")

    # TRAINING SETUP
    criterion = torch.nn.BCELoss()

    if args.inner_optimizer == "adam":
        opt_f = torch.optim.Adam
    elif args.inner_optimizer == "sgd":
        opt_f = torch.optim.SGD
    
    model.train()
    model.to(device)
        
    # TRAIN

    train_info = []

    for o_step in range(args.outer_steps):
        task_idx = random.randint(0,len(tasks))
        weights_before = deepcopy(model.state_dict())
        opt = opt_f(model.parameters(), lr=args.inner_learning_rate)

        train_loader = train_loaders[task_idx]
        test_loader = test_loaders[task_idx]
        task = tasks[task_idx]

        for i_step in range(args.inner_steps):

            epoch_train_loss = 0
            total_correct = 0
            total_samples = 0
            num_batch_count = 0

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                opt.zero_grad()
                out = model(images)

                loss = criterion(out, labels.float())

                loss.backward()
                opt.step()

                predicted = torch.round(out)
                epoch_train_loss += loss.item()
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                num_batch_count +=1

        with torch.no_grad():
            test_loss, test_acc = test(test_loader, model, criterion, device)
            train_info.append((task, epoch_train_loss, total_correct / total_samples, test_loss, test_acc))

            print(f"[{o_step+1}/{args.outer_steps}] Current task: {task}")
            print(f"[{o_step+1}/{args.outer_steps}] Train - Loss: {epoch_train_loss} Acc: {total_correct / total_samples}")
            print(f"[{o_step+1}/{args.outer_steps}] Validation - Loss: {test_loss} Acc: {test_acc}", flush=True)
        
        weights_after = model.state_dict()
        outerstepsize = args.outer_learning_rate * (1 - o_step / args.outer_steps)

        model.load_state_dict({name : 
            weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize 
            for name in weights_before})
            
    # SAVE MODEL AND FIGS
        
    torch.save({
                'model_state_dict': model.state_dict()
                }, args.save_path + "/checkpoint.pth")
    
    train_loss_dict = {name:[] for name in tasks}
    train_acc_dict = {name:[] for name in tasks}
    test_loss_dict = {name:[] for name in tasks}
    test_acc_dict = {name:[] for name in tasks}

    for i,(task, train_loss, train_acc, test_loss, test_acc) in enumerate(train_info):
        train_loss_dict[task].append((train_loss,i))
        train_acc_dict[task].append((train_acc,i))
        test_loss_dict[task].append((test_loss,i))
        test_acc_dict[task].append((test_acc,i))

    save_multi_plot(args.save_path + "/train_loss.png", train_loss_dict, "Loss", "Train Loss")
    save_multi_plot(args.save_path + "/test_loss.png", test_loss_dict, "Loss", "Test Loss")
    save_multi_plot(args.save_path + "/train_acc.png", train_acc_dict, "Accuracy", "Train Accuracy")
    save_multi_plot(args.save_path + "/test_acc.png", test_acc_dict, "Accuracy", "Test accuracy")

    return

if __name__ == "__main__":
    main()