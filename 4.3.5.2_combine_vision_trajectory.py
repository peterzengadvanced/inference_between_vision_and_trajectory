import os
import torch
import numpy as np
from torchvision import transforms, models 
from PIL import Image
import cv2

class PoseRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)  # Now works due to fixed import
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 6)

    def forward(self, x):
        return self.resnet(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseRegressor().to(device)
    model.load_state_dict(torch.load("C:/Users/Peter Zeng/Desktop/resnetdata/step2outcome/models/best_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_dir = "C:/Users/Peter Zeng/Desktop/resnetdata/commercialimplement/onecamera/"
    output_img_dir = "C:/Users/Peter Zeng/Desktop/resnetdata/commercialimplement/output/"
    output_pose_dir = "C:/Users/Peter Zeng/Desktop/resnetdata/commercialimplement/output/"
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_pose_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            original = np.array(image)
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                pose = model(img_tensor).cpu().numpy()[0]

            labeled_img = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            text = f"x: {pose[0]:.2f}, y: {pose[1]:.2f}, z: {pose[2]:.2f}"
            text_rot = f"roll: {np.degrees(pose[3]):.1f}°, pitch: {np.degrees(pose[4]):.1f}°, yaw: {np.degrees(pose[5]):.1f}°"
            cv2.putText(labeled_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(labeled_img, text_rot, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_img_dir, f"labeled_{img_name}"), labeled_img)

            with open(os.path.join(output_pose_dir, f"{os.path.splitext(img_name)[0]}_pose.txt"), 'w') as f:
                f.write('\n'.join(map(str, pose)) + '\n')

if __name__ == "__main__":
    main()
