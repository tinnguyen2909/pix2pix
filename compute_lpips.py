import argparse
import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms

def load_image(path):
    # Load image and convert to RGB
    img = Image.open(path).convert('RGB')
    
    # Resize and transform to tensor in [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to fixed size (optional)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [0,1] -> [-1,1]
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def main():
    parser = argparse.ArgumentParser(description='Compute LPIPS distance between two images.')
    parser.add_argument('--img1', required=True, help='Path to first image')
    parser.add_argument('--img2', required=True, help='Path to second image')
    parser.add_argument('--net', default='vgg', choices=['alex', 'vgg', 'squeeze'], help='Backbone network for LPIPS')
    args = parser.parse_args()

    # Load images
    img1 = load_image(args.img1)
    img2 = load_image(args.img2)

    # Initialize LPIPS loss
    loss_fn = lpips.LPIPS(net=args.net)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
        loss_fn = loss_fn.cuda()

    # Compute distance
    distance = loss_fn(img1, img2)
    print(f"LPIPS ({args.net}) distance: {distance.item():.4f}")

if __name__ == '__main__':
    main()
