import pandas as pd
from pathlib import Path

def build_metadata(raw_data_path='../data/raw'):
    print(f"Scanning {raw_data_path} for paintings...")
    
    data_list = []
    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    all_image_paths = []
    
    for ext in extensions:
        all_image_paths.extend(Path(raw_data_path).rglob(ext))
        
    for img_path in all_image_paths:
        try:
            
            style = img_path.parent.name
            artist = img_path.name.split('_')[0]
            
            data_list.append({
                'path': str(img_path),
                'style': style,
                'artist': artist
            })
        except Exception:
            continue 
            
    df = pd.DataFrame(data_list)
    
   
    csv_path = '../data/metadata.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Success! Mapped {len(df)} images to {csv_path}.")
    return df