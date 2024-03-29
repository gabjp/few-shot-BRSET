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
parser.add_argument("--train-l2", type=float, default=0)
parser.add_argument("--eval-l2", type=float, default=0)
parser.add_argument('--transform', action='store_true')

###### OUTER LOOP SETTINGS ######
parser.add_argument("--outer-learning-rate", type=float, default=0.001)
parser.add_argument("--outer-steps", type=int, default=20)

###### DATA AND SAVE SETTINGS ######
parser.add_argument("--train-tasks", type=str, default="")
parser.add_argument("--val-tasks", type=str, default="")
parser.add_argument("--train-set", type=str, default="")
parser.add_argument("--save-path", type=str, default="")
parser.add_argument("--img-resize", type=str, default="256,256")

args = parser.parse_args()
print(args)

def train_on_task(model, opt, criterion, train_loader):
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
    return epoch_train_loss, (total_correct/total_samples)*100

def main():

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # LOAD DATA
    img_size = tuple( [int(args.img_resize.split(',')[i]) for i in [0,1]] )
    transform = transforms.Resize(img_size, antialias=False)
    train_tasks = args.train_tasks.split(',')
    val_tasks = args.val_tasks.split(',')

    train_transform = transform if not args.transform else transforms.Compose([transform,
                                                                                transforms.RandomHorizontalFlip(p=0.5),
                                                                                transforms.RandomVerticalFlip(p=0.5)])

    train_loaders_t = []
    test_loaders_t = []

    print("TRAIN TASKS")

    for task in train_tasks:

        train_set = FewShotBRSET(transform=train_transform, tasks=task, split=args.train_set)
        test_set = FewShotBRSET(transform=transform, tasks=task, split='test')

        print(task)
        print(f"TRAIN SET SIZE: {len(train_set)}")
        print(f"TEST SET SIZE: {len(test_set)}")

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        train_loaders_t.append(train_dataloader)
        test_loaders_t.append(test_dataloader)
    
    print()
    print("VAL TASKS")

    train_loaders_v = []
    test_loaders_v = []

    for task in val_tasks:

        train_set = FewShotBRSET(transform=train_transform, tasks=task, split=args.train_set)
        test_set = FewShotBRSET(transform=transform, tasks=task, split='test')

        print(task)
        print(f"TRAIN SET SIZE: {len(train_set)}")
        print(f"TEST SET SIZE: {len(test_set)}")

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        train_loaders_v.append(train_dataloader)
        test_loaders_v.append(test_dataloader)
        
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
    val_info = []

    best_val_acc = 0

    for o_step in range(args.outer_steps):
        task_idx = random.randint(0,len(train_tasks) - 1)
        weights_before = deepcopy(model.state_dict())
        opt = opt_f(model.parameters(), lr=args.inner_learning_rate, weight_decay=args.train_l2)

        train_loader = train_loaders_t[task_idx]
        test_loader = test_loaders_t[task_idx]
        task = train_tasks[task_idx]

        train_loss, train_acc = train_on_task(model, opt, criterion, train_loader)

        with torch.no_grad():
            test_loss, test_acc = test(test_loader, model, criterion, device)
            train_info.append((task, train_loss, train_acc, test_loss, test_acc))

            print("TRAINING TASK")
            print(f"[{o_step+1}/{args.outer_steps}] Current task: {task}")
            print(f"[{o_step+1}/{args.outer_steps}] Train - Loss: {train_loss} Acc: {train_acc}")
            print(f"[{o_step+1}/{args.outer_steps}] Validation - Loss: {test_loss} Acc: {test_acc}", flush=True)
        
        weights_after = model.state_dict()
        outerstepsize = args.outer_learning_rate * (1 - o_step / args.outer_steps)

        model.load_state_dict({name : 
            weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize 
            for name in weights_before})
        
        if o_step % 10 == 0:

            # CHECKPOINT
            val_acc = 0
            checkpoint = deepcopy(model.state_dict())
            for i in range(len(val_tasks)):
                opt = opt_f(model.parameters(), lr=args.inner_learning_rate, weight_decay=args.eval_l2) # Reset optimizer
                train_loader = train_loaders_v[i]
                test_loader = test_loaders_v[i]
                task = val_tasks[i]

                train_loss, train_acc = train_on_task(model, opt, criterion, train_loader)

                with torch.no_grad():
                    test_loss, test_acc = test(test_loader, model, criterion, device)
                    val_info.append((task, train_loss, train_acc, test_loss, test_acc))

                    print("VALIDATION TASK")
                    print(f"[{o_step+1}/{args.outer_steps}] Current task: {task}")
                    print(f"[{o_step+1}/{args.outer_steps}] Train - Loss: {train_loss} Acc: {train_acc}")
                    print(f"[{o_step+1}/{args.outer_steps}] Validation - Loss: {test_loss} Acc: {test_acc}", flush=True)
                    val_acc += test_acc

                # RESTORE MODEL
                model.load_state_dict(checkpoint)

            val_acc = val_acc / len(val_tasks)
            print(f"Validation accuracy mean {val_acc}")
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save({
                'model_state_dict': model.state_dict()
                }, args.save_path + "/best_checkpoint.pth")
            
    print(f"BEST VALIDATION ACC: {best_val_acc}")
    # SAVE MODEL AND FIGS
        
    torch.save({
                'model_state_dict': model.state_dict()
                }, args.save_path + "/last_checkpoint.pth")
    
    # TRAIN TASKS
    
    train_loss_dict = {name:[] for name in train_tasks}
    train_acc_dict = {name:[] for name in train_tasks}
    test_loss_dict = {name:[] for name in train_tasks}
    test_acc_dict = {name:[] for name in train_tasks}

    for i,(task, train_loss, train_acc, test_loss, test_acc) in enumerate(train_info):
        train_loss_dict[task].append((train_loss,i))
        train_acc_dict[task].append((train_acc,i))
        test_loss_dict[task].append((test_loss,i))
        test_acc_dict[task].append((test_acc,i))

    save_multi_plot(args.save_path + "/train_loss_train_task.png", train_loss_dict, "Loss", "Train Loss")
    save_multi_plot(args.save_path + "/test_loss_train_task.png", test_loss_dict, "Loss", "Test Loss")
    save_multi_plot(args.save_path + "/train_acc_train_task.png", train_acc_dict, "Accuracy", "Train Accuracy")
    save_multi_plot(args.save_path + "/test_acc_train_task.png", test_acc_dict, "Accuracy", "Test accuracy")

    # VALIDATION TASKS

    train_loss_dict = {name:[] for name in val_tasks}
    train_acc_dict = {name:[] for name in val_tasks}
    test_loss_dict = {name:[] for name in val_tasks}
    test_acc_dict = {name:[] for name in val_tasks}

    for i in range(int(len(val_info)/len(val_tasks))):
        for j in range(len(val_tasks)):
            task, train_loss, train_acc, test_loss, test_acc = val_info[i* len(val_tasks) + j]
            train_loss_dict[task].append((train_loss,i))
            train_acc_dict[task].append((train_acc,i))
            test_loss_dict[task].append((test_loss,i))
            test_acc_dict[task].append((test_acc,i))

    #for i,(task, train_loss, train_acc, test_loss, test_acc) in enumerate(val_info):
    #    train_loss_dict[task].append((train_loss,i))
    #    train_acc_dict[task].append((train_acc,i))
    #    test_loss_dict[task].append((test_loss,i))
    #    test_acc_dict[task].append((test_acc,i))

    save_multi_plot(args.save_path + "/train_loss_val_task.png", train_loss_dict, "Loss", "Train Loss")
    save_multi_plot(args.save_path + "/test_loss_val_task.png", test_loss_dict, "Loss", "Test Loss")
    save_multi_plot(args.save_path + "/train_acc_val_task.png", train_acc_dict, "Accuracy", "Train Accuracy")
    save_multi_plot(args.save_path + "/test_acc_val_task.png", test_acc_dict, "Accuracy", "Test accuracy")

    return

if __name__ == "__main__":
    main()