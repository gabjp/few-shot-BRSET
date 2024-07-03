import torch
from torchvision import transforms
import numpy as np
from data.data_manager import BRSETManager
import timm
from torchvision.transforms import v2
from torchvision.io import read_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import argparse
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

###### EXPERIMENT SETTINGS ######
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--ways", type=int, default=None)
parser.add_argument("--shots", type=int, default=None)
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--target-class", type=str, default=None)
parser.add_argument("--classes", type=str, default=None)
parser.add_argument("--augment", type=str, default=None)
parser.add_argument("--save-path", type=str, default=None)


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def main(): 
    args = parser.parse_args()
    print(args)

    orig_image = read_image(f"data/imgs/{args.image}.jpg") / 255 #img06628
    t = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    image = t(orig_image)
    image = image.reshape(1,3,224,224).to(device)
    

    n_updates = 52
    epochs = n_updates * 5 // args.shots
    
    args.classes = args.classes.split(",")
    
    manager = BRSETManager([], args.classes, args.shots, args.ways, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), augment = args.augment, batch_size = 5 * args.ways, remove_img = args.image)

    if args.augment == "cutmix":
        aug = v2.CutMix(num_classes=2)
    elif args.augment == "mixup":
        aug = v2.MixUp(num_classes=2)


    hit_count = 0
    masks = 0

    
    with tqdm(total=100) as pbar:
        while hit_count < 100:
            #Load model
            model = timm.create_model("swin_s3_tiny_224.ms_in1k", pretrained=True, num_classes=args.ways)

            if args.model_path != None:
                checkpoint = torch.load(args.model_path)['model_state_dict']
                current_model_dict = model.state_dict()
                checkpoint = {k:v if v.size()==current_model_dict[k].size()  
                                  else  current_model_dict[k] 
                                  for k,v in checkpoint.items()}
                model.load_state_dict(checkpoint, strict=False)

            model.to(device)
            model.train()

            #Train model
            train_loader, _, class_names = manager.get_eval_task()
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
            pred = int(torch.argmax(model(image)))

            if class_names[pred] == args.target_class:
                target_layers = [model.layers[-1].blocks[-1].norm2]
                cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
                mask = cam(input_tensor=image, targets=None)
                masks = masks + mask
                hit_count += 1
                pbar.update(1)
    

    masks = masks / 100
    final_mask = masks[0, :]

    rgb_img = cv2.imread(f"data/imgs/{args.image}.jpg", 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    cam_image = show_cam_on_image(rgb_img, final_mask)
    cv2.imwrite(args.save_path, cam_image)
    return 

if __name__ == "__main__":
    main()