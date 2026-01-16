'''
1. Print topo for comparing ten correct samples when imagining 'f' and 'k' by attention and SHAP

2. For 'k' compare ten correct and incorrect samples when imagining 'k'
'''

from config import get_config
from data_loader import get_loader
from random import shuffle

import models
import matplotlib.pyplot as plt
import mne
import shap
import torch
from random import random
import numpy as np

random_name = str(random())
random_seed = 336
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

#load Conformer for Sub 

test_config = get_config(mode='test')

test_config.subject=8

model = getattr(models, 'Conformer')(test_config)

sub = test_config.subject
model_weight_path='./checkpoints/model_'+str(sub)+'.std'

model.load_state_dict(torch.load(model_weight_path))
model = model

model.eval()

target_layers = [model[0], model[1]]



test_data_loader = get_loader(test_config, shuffle=False)

from sklearn.metrics import accuracy_score
attentions=[]

preds=[]
labels = []

correct_features=[]

for feature, label in test_data_loader:
    feature=feature
    label = label
    out = model(feature)
    pred = torch.argmax(out,dim=1)

    for i in range(len(pred)):
        if pred[i]==label[i]:
            correct_features.append((feature[i], label[i]))
    

#f: 0, k: 1
shuffle(correct_features)

f_cnt=0
k_cnt=0

f_features=[]
k_features=[]

for sample in correct_features:
    feature, label = sample

    if label==0:
        f_features.append(feature)
    elif label==1:
        k_features.append(feature)

f_features=torch.from_numpy(np.asarray(f_features))
k_features=torch.from_numpy(np.asarray(k_features))

f_atts=target_layers[0](f_features[:10])
k_atts=target_layers[0](k_features[:10])

f_atts= f_atts.detach().cpu().numpy()
k_atts= k_atts.detach().cpu().numpy()

print('=====att map=====')

print(f_atts.shape, k_atts.shape)

background = f_features[:50]  # background examples
explainer = shap.GradientExplainer(model, background)
f_shap_values = explainer.shap_values(f_features[:10])[:,:,0]

background = k_features[:50]  # background examples
explainer = shap.GradientExplainer(model, background)
k_shap_values = explainer.shap_values(k_features[:10])[:,:,1]


CAND_CHANNELS=['F7', 'F3', 'F4', 'F8', 'FC5', 'FC6', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2', 'AF3', 'AF4']

inter1020_montage = mne.channels.make_standard_montage('standard_1020')

cand_channels = [ch for ch in inter1020_montage.ch_names if ch in CAND_CHANNELS ]
info = mne.create_info(ch_names = cand_channels, sfreq=256., ch_types='eeg')





def plot_row(atts, savename, label):
    fig,axes=plt.subplots(1,10,figsize=(14,5), gridspec_kw={'hspace': 0.7})
    col=0
    for att in atts:
        evoked = mne.EvokedArray(np.expand_dims(att,axis=1), info)
        evoked.set_montage(inter1020_montage)    
        im, cn = mne.viz.plot_topomap(att, evoked.info, show=False, axes=axes[col], res=1200)
        col+=1

    for i in range(1,11):
        axes[i-1].set_title('Test img '+str(i), fontsize=8)
        axes[i-1].axis('off')


    fig.text(0.5, 0.35, label, ha='center', va='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(savename, bbox_inches='tight')

plot_row(f_atts, 'f_att.png', '(a) Visualization of the attention output from the Transformer for imagined \'f\'')
plot_row(k_atts, 'k_att.png', '(b) Visualization of the attention output from the Transformer for imagined \'k\'')
plot_row(f_shap_values, 'f_shap.png', '(c) Visualization of the SHAP interpretation for imagined \'f\'')
plot_row(k_shap_values, 'k_shap.png', '(d) Visualization of the SHAP interpretation for imagined \'k\'')


print('f_shap_values:', 'k_shap_values:')
print(f_shap_values.shape)

import numpy as np
import matplotlib.pyplot as plt
import mne

# Example: replace this with your actual SHAP array
# shape: (10 samples × 14 channels)
shap_values = np.random.randn(10, 14) * 0.2  # example random data

# Compute average across samples


ch_names = CAND_CHANNELS



fig, axes = plt.subplots(1, 2, figsize=(9, 4))

im = axes[0].imshow(f_shap_values, aspect='auto', vmin=np.min(f_shap_values), vmax=np.max(f_shap_values))

axes[0].set_title('SHAP scores for \'f\'', fontsize=12)


axes[0].set_xlabel('Channel')
axes[0].set_ylabel('Sample')
axes[0].set_xticks(np.arange(len(ch_names)))
axes[0].set_xticklabels(ch_names, fontsize=9)

# --- 2. Heatmap of SHAP values per sample ---
im = axes[1].imshow(k_shap_values, aspect='auto')
axes[1].set_title('SHAP scores for \'k\'', fontsize=12)
axes[1].set_xlabel('Channel')
axes[1].set_ylabel('Sample')
axes[1].set_xticks(np.arange(len(ch_names)))
axes[1].set_xticklabels(ch_names, fontsize=9)

# Add colorbar for heatmap
cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
cbar.set_label('SHAP value', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('shap_scores.png')

from nilearn import plotting, datasets, surface


# -------------------------------
# 1. Fetch template cortical surface
# -------------------------------
fsaverage = datasets.fetch_surf_fsaverage()

# Load the left inflated surface mesh
# surf_mesh[0] = coordinates, surf_mesh[1] = faces
coords, faces = surface.load_surf_mesh(fsaverage.infl_left)

# -------------------------------
# 2. Load group-level data
# -------------------------------
n_vertices = coords.shape[0]
group_map = np.random.randn(n_vertices)  # Replace with your actual data

# -------------------------------
# 3. Plot cortical overlay
# -------------------------------
view = plotting.view_surf(
    fsaverage.infl_left,           # Surface mesh file path
    group_map,                     # Data values per vertex
    hemi='left',                   # Left hemisphere
    cmap='cold_hot',               # Colormap
    threshold=2.0,                 # Optional threshold
    colorbar=True,                 # Show colorbar
    bg_map=fsaverage.sulc_left,    # Background sulcal depth
)
view  # In Jupyter, this shows the interactive view



from PIL import Image

# List your image filenames
images = [Image.open('f_att.png'), Image.open('k_att.png'), Image.open('f_shap.png'), Image.open('k_shap.png')]

# Find the max width and total height
widths, heights = zip(*(img.size for img in images))
max_width = max(widths)
total_height = sum(heights)

# Create a new blank image
combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))

# Paste each image vertically
y_offset = 0
for img in images:
    combined.paste(img, (0, y_offset))
    y_offset += img.height

# Save the combined image
combined.save('combined_vertical.png')
print("✅ Saved as combined_vertical.png")
