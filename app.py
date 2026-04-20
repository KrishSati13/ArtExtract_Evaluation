import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import os

# 1. 📍 THE ABSOLUTE PATHS
base_dir = '/Users/krishsati13/Desktop/GSoC_ArtExtract/data/nga'
image_dir = os.path.join(base_dir, 'images')

print("Loading Brain Matrix...")
vectors = np.load(os.path.join(base_dir, 'nga_vectors.npy'))
object_ids = np.load(os.path.join(base_dir, 'nga_vector_ids.npy'))
metadata = pd.read_csv(os.path.join(base_dir, 'nga_full_dataset_metadata.csv'), low_memory=False)

print("Mapping standard Cosine space...")
knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
knn.fit(vectors)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 🎯 THE ENGINE
def find_matches(user_image):
    # Convert Gradio image to PIL
    img = Image.fromarray(user_image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_vector = model(img_tensor).cpu().numpy()
        
    distances, indices = knn.kneighbors(query_vector)
    
    # Gather the top 5 matches to send back to the web UI
    match_images = []
    for i in range(1, 6): # Skip index 0
        idx = indices[0][i]
        match_id = str(object_ids[idx])
        match_path = os.path.join(image_dir, f"{match_id}.jpg")
        
        try:
            title = metadata[metadata['objectid'] == int(match_id)]['title'].values[0]
            artist = metadata[metadata['objectid'] == int(match_id)]['attribution'].values[0]
            label = f"{title[:30]}...\nBy: {artist}"
        except:
            label = "Unknown Artwork"
            
        try:
            match_images.append((Image.open(match_path), label))
        except:
            pass
            
    return match_images

# 3. 🎨 THE WEB UI DESIGN
with gr.Blocks(theme=gr.themes.Soft()) as web_app:
    gr.Markdown("# 🏛️ The GSoC Neural Art Engine")
    gr.Markdown("Upload any image. The AI will extract its mathematical visual features and search through **22,842** National Gallery of Art paintings to find the closest matches in milliseconds.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Your Query Image")
            search_button = gr.Button("🔍 Search Database", variant="primary")
            
        with gr.Column(scale=2):
            gallery_output = gr.Gallery(
                label="Top 5 Neural Matches", 
                show_label=True, 
                elem_id="gallery", 
                columns=5, 
                height="auto"
            )

    search_button.click(fn=find_matches, inputs=image_input, outputs=gallery_output)

# 4. LAUNCH!
if __name__ == "__main__":
    web_app.launch()