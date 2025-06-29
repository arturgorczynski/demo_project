import torch, torchvision.transforms as T, PIL.Image as Image
from torchvision import transforms
import io
from typing import Union, BinaryIO
import matplotlib.pyplot as plt

IMG_SIZE = 224 
prediction_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                        [0.229,0.224,0.225]),
])

classes = torch.load("checkpoints/best.pt", map_location="cpu")["classes"]
model = torch.jit.load("checkpoints/model_scripted.pt")  # already eval

def predict_photo(
    image_path: str | None = None,
    image_file: Union[BinaryIO, bytes, None] = None,
    *,
    model=model,
    tfms=prediction_tfms
    ) -> str:
    """
    Return a class label from either an imaage path or an uploaded file/
    stream. Exactly one of `image_path` or `image_file` must be given.
    """
    if (image_path is None) == (image_file is None):
        raise ValueError("Pass *either* image_path or image_file")

    # ---------- load image ----------
    if image_path is not None:
        img = Image.open(image_path).convert("RGB")  # path or PathLike

    else:                                           # file-like / bytes
        if hasattr(image_file, "read"):             
            img = Image.open(image_file).convert("RGB")
        else:                                       # or raw bytes
            img = Image.open(io.BytesIO(image_file)).convert("RGB")

    # ---------- predict ----------
    x = tfms(img).unsqueeze(0)           # 1×C×H×W tensor
    pred = model(x).softmax(1).argmax(1).item()
    return classes[pred]     

def system_show_image(path:str, label:str = None):
    img   = Image.open(path)
    plt.imshow(img)
    if label:                          # PIL → RGB image display :contentReference[oaicite:0]{index=0}
        plt.title(f"Predicted: {label}", fontsize=14)
    else:
        plt.title(f"Object used for prediction: {label}", fontsize=14)
    plt.axis("off")                              # cleaner look :contentReference[oaicite:1]{index=1}
    plt.show()


if __name__ == "__main__":
    path = "./data/food-101/images/apple_pie/23893.jpg"
    label = predict_photo(image_path=path)
    label = (" ").join(label.split("_"))
    print(label)
    system_show_image(path, label)
