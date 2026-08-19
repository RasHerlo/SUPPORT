import argparse
import os

import numpy as np
import skimage.io as skio
import torch
from tqdm import tqdm

from model.SUPPORT import SUPPORT
from src.utils.dataset import DatasetSUPPORT_test_stitch
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
        denoised_stack = np.zeros(test_dataloader.dataset.noisy_image.shape, dtype=np.float32)

        for _, (noisy_image, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="Processing")):
            noisy_image = noisy_image.cuda()
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

                denoised_stack[stack_start_s + (T // 2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

        denoised_stack = denoised_stack * test_dataloader.dataset.std_image.numpy() \
            + test_dataloader.dataset.mean_image.numpy()

        return denoised_stack


def default_output_path(stack_path):
    base, ext = os.path.splitext(stack_path)
    return f"{base}_denoised{ext or '.tif'}"


def maybe_score(raw_image, denoised_stack, stack_path, output_path, args):
    if args.no_score:
        return None
    channel = args.channel.upper() if args.channel else infer_channel_letter(stack_path, output_path)
    try:
        sig, sig_path = resolve_signature(
            channel=channel,
            signature=args.signature,
            search_from=stack_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Skipping FFT metrics: {exc}")
        return None

    result = evaluate_denoise_pair(
        raw_image,
        denoised_stack,
        sig,
        signature_path=sig_path,
        frame_stride=args.score_frame_stride,
        max_frames=args.score_max_frames,
        skip_frames=args.score_mean_only,
    )
    result["pre_path"] = os.path.abspath(stack_path)
    result["post_path"] = os.path.abspath(output_path)
    result["channel"] = channel
    result["model_path"] = os.path.abspath(args.model)
    metrics_path = os.path.splitext(output_path)[0] + "_fft_metrics.json"
    write_metrics_json(result, metrics_path)
    m = result["mean"]
    print(
        f"FFT metrics ({metrics_path}): verdict={result['verdict']}  "
        f"cell_ratio={m['cell_power_ratio']:.4f}  fringe_ratio={m['fringe_power_ratio']:.4f}"
    )
    if "frames" in result:
        f = result["frames"]
        print(
            f"  frames: fringe median={f['fringe_power_ratio_median']:.4f}  "
            f"p90={f['fringe_power_ratio_p90']:.4f}  "
            f"frac>1.05={f['frac_frames_fringe_ratio_gt_1_05']:.3f}"
        )
    return metrics_path


def main():
    parser = argparse.ArgumentParser(description='Denoise a single .tif stack using SUPPORT')
    parser.add_argument('--stack', type=str, required=True,
                        help='Path to input .tif stack')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file (.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path for denoised output .tif (default: <stack>_denoised.tif)')
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
                        help='defringe signature.json for FFT QC (auto from ChanA/B if omitted)')
    parser.add_argument('--channel', choices=['A', 'B', 'a', 'b'], default=None,
                        help='Channel letter for default signature lookup')
    parser.add_argument('--no_score', action='store_true',
                        help='Skip signature FFT metrics after denoise')
    parser.add_argument('--score_mean_only', action='store_true',
                        help='Only score temporal means (skip per-frame tails)')
    parser.add_argument('--score_frame_stride', type=int, default=1,
                        help='Stride for per-frame fringe/cell ratios')
    parser.add_argument('--score_max_frames', type=int, default=None,
                        help='Optional cap on scored frames')
    args = parser.parse_args()

    stack_path = os.path.abspath(args.stack)
    model_path = os.path.abspath(args.model)
    output_path = os.path.abspath(args.output or default_output_path(stack_path))
    edge_mode = None if args.include_first_last == 'none' else args.include_first_last

    if not os.path.isfile(stack_path):
        raise FileNotFoundError(f"Stack not found: {stack_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = SUPPORT(
        in_channels=args.patch_size[0],
        mid_channels=[16, 32, 64, 128, 256],
        depth=5,
        blind_conv_channels=64,
        one_by_one_channels=[32, 16],
        last_layer_channels=[64, 32, 16],
        bs_size=args.bs_size,
    ).cuda()
    model.load_state_dict(torch.load(model_path))

    raw_image = skio.imread(stack_path)
    print(f"Input stack: {stack_path}")
    print(f"Shape: {raw_image.shape}, dtype: {raw_image.dtype}")
    print(f"include_first_last: {args.include_first_last}")

    if (raw_image.shape[0] < args.patch_size[0]
            or raw_image.shape[1] < args.patch_size[1]
            or raw_image.shape[2] < args.patch_size[2]):
        raise ValueError(
            f"Stack dimensions {raw_image.shape} are smaller than patch size {args.patch_size}"
        )

    demo_tif = torch.from_numpy(raw_image.astype(np.float32)).type(torch.FloatTensor)

    if edge_mode == "repeat":
        print('Padding temporal edges by repeating boundary frames (full-length output).')
        demo_tif = torch.cat([
            demo_tif[0, :, :].unsqueeze(0).repeat((args.patch_size[0] // 2, 1, 1)),
            demo_tif,
            demo_tif[-1, :, :].unsqueeze(0).repeat((args.patch_size[0] // 2, 1, 1)),
        ])
    elif edge_mode == "mirror":
        print('Padding temporal edges by mirroring boundary frames (full-length output).')
        demo_tif = torch.cat([
            demo_tif[1:(args.patch_size[0] // 2) + 1, :, :].flip(0),
            demo_tif,
            demo_tif[-1 * (args.patch_size[0] // 2) - 1:-1, :, :].flip(0),
        ])

    testset = DatasetSUPPORT_test_stitch(
        demo_tif,
        patch_size=args.patch_size,
        patch_interval=args.patch_interval,
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size)
    denoised_stack = validate(testloader, model)

    trim = (model.in_channels - 1) // 2
    if edge_mode in ["repeat", "mirror"]:
        denoised_stack = denoised_stack[args.patch_size[0] // 2:-1 * (args.patch_size[0] // 2)]
    else:
        denoised_stack = denoised_stack[trim:-trim, :, :]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving denoised stack to: {output_path}")
    print(f"Output shape: {denoised_stack.shape}")
    skio.imsave(output_path, denoised_stack, metadata={'axes': 'TYX'})

    maybe_score(raw_image, denoised_stack, stack_path, output_path, args)


if __name__ == '__main__':
    main()
