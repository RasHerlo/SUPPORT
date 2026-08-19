import numpy as np
import torch
import skimage.io as skio
import os
import os.path
from tqdm import tqdm
from src.utils.dataset import DatasetSUPPORT_test_stitch
from model.SUPPORT import SUPPORT
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from src.utils.fft_metrics import (
    evaluate_denoise_pair,
    infer_channel_letter,
    resolve_signature,
    write_metrics_json,
)

def validate(test_dataloader, model):
    """
    Validate a model with a test data
    
    Arguments:
        test_dataloader: (Pytorch DataLoader)
            Should be DatasetSUPPORT_test_stitch!
        model: (Pytorch nn.Module)

    Returns:
        denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
    """
    with torch.no_grad():
        model.eval()
        # initialize denoised stack to NaN array.
        denoised_stack = np.zeros(test_dataloader.dataset.noisy_image.shape, dtype=np.float32)
        
        # stitching denoised stack
        for _, (noisy_image, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="Processing")):
            noisy_image = noisy_image.cuda() #[b, z, y, x]
            noisy_image_denoised = model(noisy_image)
            T = noisy_image.size(1)
            for bi in range(noisy_image.size(0)): 
                stack_start_w = int(single_coordinate['stack_start_w'][bi])
                stack_end_w = int(single_coordinate['stack_end_w'][bi])
                patch_start_w = int(single_coordinate['patch_start_w'][bi])
                patch_end_w = int(single_coordinate['patch_end_w'][bi])

                stack_start_h = int(single_coordinate['stack_start_h'][bi])
                stack_end_h = int(single_coordinate['stack_end_h'][bi])
                patch_start_h = int(single_coordinate['patch_start_h'][bi])
                patch_end_h = int(single_coordinate['patch_end_h'][bi])

                stack_start_s = int(single_coordinate['init_s'][bi])
                
                denoised_stack[stack_start_s+(T//2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

        # change nan values to 0 and denormalize
        denoised_stack = denoised_stack * test_dataloader.dataset.std_image.numpy() + test_dataloader.dataset.mean_image.numpy()

        return denoised_stack

def process_file(input_file, output_dir, model, patch_size, patch_interval, batch_size,
                 include_first_and_last="mirror", model_path=None, score=True,
                 signature=None, channel_letter=None, score_mean_only=False,
                 score_frame_stride=1, score_max_frames=None):
    """Process a single .tif file"""
    print(f'\nDEBUG: Loading file from: {input_file}')
    
    # Load and prepare the image
    raw_image = skio.imread(input_file)
    print(f'DEBUG: Raw image shape: {raw_image.shape}')
    print(f'DEBUG: Raw image dtype: {raw_image.dtype}')
    print(f'DEBUG: Raw image min/max: {raw_image.min()}/{raw_image.max()}')
    
    # Check if image dimensions are smaller than patch size
    if (raw_image.shape[0] < patch_size[0] or 
        raw_image.shape[1] < patch_size[1] or 
        raw_image.shape[2] < patch_size[2]):
        print(f"WARNING: Skipping file {input_file} - Image dimensions {raw_image.shape} are smaller than patch size {patch_size}")
        return
    
    demo_tif = torch.from_numpy(raw_image.astype(np.float32)).type(torch.FloatTensor)
    print(f'DEBUG: Converted to tensor shape: {demo_tif.shape}')
    print(f'DEBUG: Tensor dtype: {demo_tif.dtype}')
    print(f'DEBUG: Tensor min/max: {demo_tif.min()}/{demo_tif.max()}')
    
    if include_first_and_last == "repeat":
        print('Padding temporal edges by repeating boundary frames (full-length output).')
        demo_tif = torch.cat([
                demo_tif[0, :, :].unsqueeze(0).repeat((patch_size[0] // 2, 1, 1)),
                demo_tif,
                demo_tif[-1, :, :].unsqueeze(0).repeat((patch_size[0] // 2, 1, 1)),
            ])
    elif include_first_and_last == "mirror":
        print('Padding temporal edges by mirroring boundary frames (full-length output).')
        demo_tif = torch.cat([
                demo_tif[1:(patch_size[0] // 2)+1, :, :].flip(0),
                demo_tif,
                demo_tif[-1 * (patch_size[0] // 2)-1:-1, :, :].flip(0),
            ])

    # Create dataset and dataloader
    testset = DatasetSUPPORT_test_stitch(demo_tif, patch_size=patch_size, patch_interval=patch_interval)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    
    # Process the image
    denoised_stack = validate(testloader, model)

    if include_first_and_last in ["repeat", "mirror"]:
        denoised_stack = denoised_stack[patch_size[0] // 2:-1 * (patch_size[0] // 2)]
    else:
        trim = (patch_size[0] - 1) // 2
        denoised_stack = denoised_stack[trim:-trim]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full denoised stack
    output_file = os.path.join(output_dir, 'denoised.tif')
    print(f'Saving full stack to: {output_file}')
    skio.imsave(output_file, denoised_stack, metadata={'axes': 'TYX'})
    
    # Save the cut version (removing first and last 30 frames) — legacy artifact
    cut_stack = denoised_stack[30:-30] if denoised_stack.shape[0] > 60 else denoised_stack
    output_file_cut = os.path.join(output_dir, 'denoised_cut.tif')
    print(f'Saving cut stack to: {output_file_cut} (frames 30 to {denoised_stack.shape[0]-30})')
    skio.imsave(output_file_cut, cut_stack, metadata={'axes': 'TYX'})
    
    # Create Z-projection (average) of the cut stack
    z_projection = np.mean(cut_stack, axis=0)
    z_proj_path = os.path.join(output_dir, 'denoised_cut_avr.tif')
    skio.imsave(z_proj_path, z_projection)
    print(f"Saving Z-projection to: {z_proj_path}")
    
    # Create and save visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(z_projection, cmap='coolwarm')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    png_path = os.path.join(output_dir, 'denoised_cut_avr.png')
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"Saving visualization to: {png_path}")

    if score:
        letter = channel_letter or infer_channel_letter(input_file, output_dir)
        try:
            sig, sig_path = resolve_signature(
                channel=letter, signature=signature, search_from=input_file
            )
            result = evaluate_denoise_pair(
                raw_image,
                denoised_stack,
                sig,
                signature_path=sig_path,
                frame_stride=score_frame_stride,
                max_frames=score_max_frames,
                skip_frames=score_mean_only,
            )
            result["pre_path"] = os.path.abspath(input_file)
            result["post_path"] = os.path.abspath(output_file)
            result["channel"] = letter
            if model_path is not None:
                result["model_path"] = os.path.abspath(model_path)
            metrics_path = os.path.join(output_dir, "fft_metrics.json")
            write_metrics_json(result, metrics_path)
            m = result["mean"]
            print(
                f"FFT metrics ({metrics_path}): verdict={result['verdict']}  "
                f"cell_ratio={m['cell_power_ratio']:.4f}  "
                f"fringe_ratio={m['fringe_power_ratio']:.4f}"
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"Skipping FFT metrics: {exc}")

def find_and_process_stacks(parent_folder, channel, model_path, patch_size, patch_interval,
                            batch_size, bs_size, include_first_and_last, score=True,
                            signature=None, score_mean_only=False, score_frame_stride=1,
                            score_max_frames=None):
    """Walk DATA folders and denoise channel stack TIFFs."""
    model = SUPPORT(in_channels=patch_size[0],
                   mid_channels=[16, 32, 64, 128, 256],
                   depth=5,
                   blind_conv_channels=64,
                   one_by_one_channels=[32, 16],
                   last_layer_channels=[64, 32, 16],
                   bs_size=bs_size).cuda()
    model.load_state_dict(torch.load(model_path))
    channel_letter = "A" if channel.endswith("A") else "B"

    for root, dirs, files in os.walk(parent_folder):
        if os.path.basename(root) != "DATA":
            continue
        channel_path = os.path.join(root, channel)
        if not os.path.exists(channel_path):
            print(f"Warning: No {channel} folder found in {root}")
            continue
        stack_file = os.path.join(channel_path, f"{channel}_stk.tif")
        if not os.path.exists(stack_file):
            print(f"Warning: No stack file found in {channel_path}")
            continue
        support_dir = os.path.join(root, f"SUPPORT_{channel}")
        if os.path.exists(support_dir):
            print(f"\nSkipping {root} - SUPPORT_{channel} folder already exists")
            continue
        print(f"\nFound stack in: {stack_file}")
        process_file(
            stack_file, support_dir, model,
            patch_size, patch_interval, batch_size, include_first_and_last,
            model_path=model_path, score=score, signature=signature,
            channel_letter=channel_letter, score_mean_only=score_mean_only,
            score_frame_stride=score_frame_stride, score_max_frames=score_max_frames,
        )

def main():
    parser = argparse.ArgumentParser(description='Batch denoise .tif stacks using SUPPORT')
    parser.add_argument('--parent_folder', type=str, required=True,
                       help='Parent folder containing DATA folders with channel subfolders')
    parser.add_argument('--channel', type=str, required=True, choices=['ChanA', 'ChanB'],
                       help='Channel to process (ChanA or ChanB)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.pth)')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[61, 64, 64],
                       help='Patch size [t, x, y]')
    parser.add_argument('--patch_interval', type=int, nargs=3, default=[1, 32, 32],
                       help='Patch interval [t, x, y]')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing')
    parser.add_argument('--bs_size', type=int, default=3,
                       help='BS size parameter')
    parser.add_argument('--include_first_last', type=str,
                       choices=['mirror', 'repeat', 'none'], default='mirror',
                       help='Pad temporal edges so output keeps full length '
                            '(default: mirror). Use none to drop ±(T//2) frames.')
    parser.add_argument('--signature', type=str, default=None,
                       help='defringe signature.json for FFT QC')
    parser.add_argument('--no_score', action='store_true',
                       help='Skip signature FFT metrics after denoise')
    parser.add_argument('--score_mean_only', action='store_true',
                       help='Only score temporal means (skip per-frame tails)')
    parser.add_argument('--score_frame_stride', type=int, default=1)
    parser.add_argument('--score_max_frames', type=int, default=None)

    args = parser.parse_args()
    edge_mode = None if args.include_first_last == 'none' else args.include_first_last

    find_and_process_stacks(
        args.parent_folder,
        args.channel,
        args.model,
        args.patch_size,
        args.patch_interval,
        args.batch_size,
        args.bs_size,
        edge_mode,
        score=not args.no_score,
        signature=args.signature,
        score_mean_only=args.score_mean_only,
        score_frame_stride=args.score_frame_stride,
        score_max_frames=args.score_max_frames,
    )

if __name__ == '__main__':
    main()
