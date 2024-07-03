import torch
from data.data_manager import BRSETManager
import timm
from torchvision.transforms import v2
import argparse
from tqdm import tqdm
from torch import nn
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

###### EXPERIMENT SETTINGS ######
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--ways", type=int, default=None)
parser.add_argument("--shots", type=int, default=None)
parser.add_argument("--augment", type=str, default=None)
parser.add_argument("--save-path", type=str, default=None)


def test(testloader, model, device, ways):
    with torch.no_grad():
        softmax = nn.Softmax(dim=1)

        count = 0

        for (image, label) in testloader:
            image = image.to(device)
            lab = label.cpu()
            out = softmax(model(image)).cpu()
            count += 1
        
        assert count == 1
        
        r = (torch.argmax(out, dim=1) == torch.argmax(label, dim=1))
        rights = out[r].max(dim=1).values
        wrongs = out[~r].max(dim=1).values

    return rights.tolist(), wrongs.tolist()


def main(): 
    args = parser.parse_args()
    print(args)
    

    n_updates = 52
    epochs = n_updates * 5 // args.shots
    
    manager = BRSETManager([], ['hemorrhage', 'vascular_occlusion', 'nevus', 'healthy'], args.shots, args.ways, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), augment = args.augment, batch_size = 5 * args.ways)

    if args.augment == "cutmix":
        aug = v2.CutMix(num_classes=args.ways)
    elif args.augment == "mixup":
        aug = v2.MixUp(num_classes=args.ways)

    wrong = []
    right = []

    
    with tqdm(total=100) as pbar:
        while len(wrong) < 100 or len(right) < 100:
            #Load model
            model = timm.create_model("swin_s3_tiny_224.ms_in1k", pretrained=True, num_classes=args.ways)

            if args.model_path != None:
                checkpoint = torch.load(args.model_path)['model_state_dict']
                model.load_state_dict(checkpoint, strict=False)

            model.to(device)
            model.train()

            #Train model
            train_loader, test_loader, class_names = manager.get_eval_task()
            print(class_names)
            criterion = torch.nn.CrossEntropyLoss()
            opt = torch.optim.Adam(model.parameters(), lr=0.0001)

            for epoch in range(epochs):
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    if args.augment == "cutmix" or args.augment == "mixup":
                        images,labels = aug(images,torch.argmax(labels, dim=1))

                    opt.zero_grad()
                    out = model(images)
                    loss = criterion(out, labels)
                    loss.backward()
                    opt.step()
            
            model.eval()
            r,w = test(test_loader, model, device, args.ways)

            for t in r:
                right.append(t)
            for t in w:
                wrong.append(t)

            pbar.n = min(len(wrong), len(right))
            pbar.refresh()
    
            with open(args.save_path, "w") as outfile: 
                json.dump({"wrong": wrong, "right": right}, outfile)


    return 

if __name__ == "__main__":
    main()