from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()
device = torch.device("cpu")
model.to(device)

# Test image
image = Image.open("test.jpeg").convert('RGB')  # Use any JPEG image

# Inference
inputs = processor(images=image, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Caption:", caption)
