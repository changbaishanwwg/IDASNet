"""Point-prompted SAM segmentation, berry TSS inference and KDE (Figure 9).

Input is a calibrated H x W x 151 reflectance cube; SAM and IDASNet weights
are supplied explicitly. No model weights are downloaded automatically.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from data.build import preprocess_roi_pixels, PROCESSING_METHODS, WAVELENGTHS
from idasnet.inference import load_checkpoint, predict_spectra


def pseudo_rgb(cube, wavelengths):
    indices = [list(wavelengths).index(value) for value in [770,745,730]]
    image = np.asarray(cube[:,:,indices], dtype=float)
    lo, hi = np.nanpercentile(image,[1,99],axis=(0,1))
    return (np.clip((image-lo)/(hi-lo+1e-8),0,1)*255).astype(np.uint8)


def berry_spectra(cube, mask):
    if mask.shape != cube.shape[:2]:
        raise ValueError('Mask and cube image dimensions differ')
    pixels = np.asarray(cube[mask], dtype=np.float32)
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    if len(pixels) < 1:
        raise ValueError('The selected mask contains no finite spectral pixels')
    channels = preprocess_roi_pixels(pixels)
    return np.stack([channels[name].mean(axis=0) for name in PROCESSING_METHODS])[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cube', type=Path, required=True, help='Calibrated .npy, shape H x W x 151, 350–1100 nm in 5 nm steps')
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--sam-checkpoint', type=Path, required=True)
    parser.add_argument('--sam-model', choices=['vit_b','vit_l','vit_h'], default='vit_b')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    # The interactive application requires a desktop with Tk support.
    import matplotlib
    matplotlib.use('TkAgg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as error:
        raise RuntimeError('Install the optional Segment Anything dependencies described in README.md') from error
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    cube = np.load(args.cube, mmap_mode='r', allow_pickle=False)
    if cube.ndim != 3 or cube.shape[-1] != 151:
        raise ValueError('Expected calibrated reflectance (height,width,151)')
    rgb = pseudo_rgb(cube,WAVELENGTHS)
    sam = sam_model_registry[args.sam_model](checkpoint=str(args.sam_checkpoint)).to(args.device)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)
    models = []
    for path in args.checkpoints:
        model, saved = load_checkpoint(path,args.device)
        info = saved.get('input_metadata',{})
        if info.get('processing_methods') != PROCESSING_METHODS or info.get('wavelengths') != WAVELENGTHS:
            raise ValueError('Checkpoint preprocessing channel order/wavelengths do not match the image pipeline')
        models.append(model)
    figure, axes = plt.subplots(1,2,figsize=(13,6))
    axes[0].imshow(rgb)
    axes[0].set_title('Click the centre of a berry; close the window to finish')
    axes[1].set(xlabel='Predicted berry TSS (°Brix)',ylabel='Density')
    records, masks, values = [], [], []
    colour = plt.get_cmap('YlOrRd')
    scale = Normalize(8,22.4)

    def on_click(event):
        if event.inaxes is not axes[0] or event.button != 1 or event.xdata is None:
            return
        x,y = int(round(event.xdata)),int(round(event.ydata))
        if not (0<=x<cube.shape[1] and 0<=y<cube.shape[0]):
            return
        try:
            candidates,scores,_ = predictor.predict(point_coords=np.array([[x,y]]),point_labels=np.array([1]),multimask_output=True)
            valid = [i for i,mask in enumerate(candidates) if mask[y,x] and mask.any()]
            if not valid:
                raise ValueError('SAM returned no mask containing the selected point')
            mask = candidates[max(valid,key=lambda i: scores[i])]
            if any(mask[y,x] for mask in masks):
                axes[0].set_title('This berry was already selected')
                figure.canvas.draw_idle()
                return
            spectra = berry_spectra(cube,mask)
            value = float(predict_spectra(models,spectra,args.device)[0])
            masks.append(mask); values.append(value)
            records.append({'berry_id':len(values),'point_x':x,'point_y':y,'mask_pixels':int(mask.sum()),'predicted_TSS':value})
            overlay = np.zeros((*mask.shape,4))
            overlay[mask] = colour(scale(value))
            overlay[:,:,3] *= .45
            axes[0].imshow(overlay)
            axes[0].text(x,y,f'{value:.2f}',color='black',ha='center',fontsize=9,bbox={'facecolor':'white','alpha':.65,'pad':1})
            axes[0].set_title(f'{len(values)} berries selected')
            axes[1].clear()
            axes[1].set(xlabel='Predicted berry TSS (°Brix)',ylabel='Density')
            if len(values)>=2 and np.std(values)>1e-8:
                span=max(np.ptp(values)*.15,.25)
                grid=np.linspace(min(values)-span,max(values)+span,200)
                density=gaussian_kde(values)(grid)
                axes[1].plot(grid,density)
                pd.DataFrame({'predicted_TSS':grid,'density':density}).to_csv(args.output/'KDE.csv',index=False)
            else:
                axes[1].axvline(value)
            pd.DataFrame(records).to_csv(args.output/'berry_predictions.csv',index=False)
            np.savez_compressed(args.output/'berry_masks.npz',masks=np.stack(masks))
            figure.savefig(args.output/'Figure9.png',dpi=300,bbox_inches='tight')
            figure.canvas.draw_idle()
        except (ValueError,RuntimeError) as error:
            axes[0].set_title(str(error))
            figure.canvas.draw_idle()

    figure.canvas.mpl_connect('button_press_event',on_click)
    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
