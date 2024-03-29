import torch 
from data.loader import FewShotBRSET
import argparse
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from utils import SmallNet, test, save_acc, save_loss, VGG16, DumbNet
import os

parser = argparse.ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###### TRAINING SETTINGS ######
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--learning-rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=80)
parser.add_argument("--checkpoint-path", type=str, default="")
parser.add_argument('--transform', action='store_true')
parser.add_argument('--schedule', action='store_true')
parser.add_argument("--l2", type=float, default=0)

###### DATA AND SAVE SETTINGS ######
parser.add_argument("--tasks", type=str, default="")
parser.add_argument("--train-set", type=str, default="")
parser.add_argument("--save-path", type=str, default="")
parser.add_argument("--img-resize", type=str, default="394,508")


def main():
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # LOAD DATA
    img_size = tuple( [int(args.img_resize.split(',')[i]) for i in [0,1]] )
    transform = transforms.Resize(img_size, antialias=False)
    tasks = args.tasks.split(',')

    train_transform = transform if not args.transform else transforms.Compose([transform,
                                                                                transforms.RandomHorizontalFlip(p=0.5),
                                                                                transforms.RandomVerticalFlip(p=0.5)])

    train_set = FewShotBRSET(transform=train_transform, tasks=tasks, split=args.train_set)
    test_set = FewShotBRSET(transform=transform, tasks=tasks, split='test')

    print(f"TRAIN SET SIZE: {len(train_set)}")
    print(f"TEST SET SIZE: {len(test_set)}")

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # LOAD MODEL
    model = DumbNet()
    if args.checkpoint_path != "":
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TRAINABLE PARAMETERS: {n_trainable}")

    # TRAINING SETUP
    criterion = torch.nn.BCELoss()

    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    
    model.train()
    model.to(device)

    # TRAINING LOOP

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for epoch in range(args.epochs):

        epoch_train_loss = 0
        total_correct = 0
        total_samples = 0
        num_batch_count = 0

        if args.schedule:
            opt.param_groups[0]['lr'] = args.learning_rate * (0.1 ** (epoch/20))

        for i, (images, labels) in enumerate(train_dataloader):
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
        
            print(f"[{epoch+1}/{args.epochs}] - Training loss: {epoch_train_loss/num_batch_count} - Training accuracy: {total_correct / total_samples * 100}")
            train_loss.append(epoch_train_loss/num_batch_count)
            train_acc.append(total_correct / total_samples * 100)

            val_loss, val_acc = test(test_dataloader, model, criterion, device)
            print(f"[{epoch+1}/{args.epochs}] - Validation loss: {val_loss} - Validation accuracy: {val_acc}", flush=True)
            test_loss.append(val_loss)
            test_acc.append(val_acc)

    # SAVE MODEL 
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': criterion,
                }, args.save_path + "/checkpoint.pth")
    
    # SAVE PLOTS
    save_acc(args.save_path, train_acc, test_acc)
    save_loss(args.save_path, train_loss, test_loss)

    return

if __name__ == "__main__":
    main()