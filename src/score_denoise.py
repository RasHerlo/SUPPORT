"""Score defringed vs SUPPORT-denoised stacks with signature FFT metrics."""

from __future__ import annotations

import argparse
import os

import skimage.io as skio

from src.utils.fft_metrics import (
    evaluate_denoise_pair,
    infer_channel_letter,
    resolve_signature,
    write_metrics_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Score pre (defringed) vs post (SUPPORT) with PMT-family FFT masks"
    )
    parser.add_argument("--pre", required=True, help="Pre stack (e.g. defringed_v21)")
    parser.add_argument("--post", required=True, help="Post stack (SUPPORT output)")
    parser.add_argument("--signature", default=None, help="Path to signature.json")
    parser.add_argument("--channel", choices=["A", "B", "a", "b"], default=None)
    parser.add_argument("--output", default=None, help="Metrics JSON path")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_frames", action="store_true",
                        help="Only score temporal means (faster)")
    args = parser.parse_args()

    pre_path = os.path.abspath(args.pre)
    post_path = os.path.abspath(args.post)
    channel = args.channel.upper() if args.channel else infer_channel_letter(pre_path, post_path)
    sig, sig_path = resolve_signature(
        channel=channel, signature=args.signature, search_from=pre_path
    )
    pre = skio.imread(pre_path)
    post = skio.imread(post_path)
    result = evaluate_denoise_pair(
        pre,
        post,
        sig,
        signature_path=sig_path,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
    )
    result["pre_path"] = pre_path
    result["post_path"] = post_path
    result["channel"] = channel

    out = args.output or (os.path.splitext(post_path)[0] + "_fft_metrics.json")
    write_metrics_json(result, out)
    m = result["mean"]
    print(f"Wrote {out}")
    print(
        f"verdict={result['verdict']}  "
        f"cell_ratio={m['cell_power_ratio']:.4f}  "
        f"fringe_ratio={m['fringe_power_ratio']:.4f}"
    )
    if "frames" in result:
        f = result["frames"]
        print(
            f"frames: fringe median={f['fringe_power_ratio_median']:.4f}  "
            f"p90={f['fringe_power_ratio_p90']:.4f}  "
            f"frac>1.05={f['frac_frames_fringe_ratio_gt_1_05']:.3f}"
        )


if __name__ == "__main__":
    main()
