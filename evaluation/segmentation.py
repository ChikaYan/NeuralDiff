"""
    Evaluate segmentation capacity of model via mAP,
    also includes renderings of segmentations and PSNR evaluation.
"""
import os
from collections import defaultdict

import git
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score
from pathlib import Path
import json
import cv2
import imageio
from PIL import Image
import torchvision

from . import metrics, utils


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = torchvision.transforms.ToTensor()(x_)  # (3, H, W)
    return x_

def evaluate_sample(
    ds,
    sample_id,
    t=None,
    visualise=True,
    gt_masked=None,
    model=None,
    mask_targ=None,
    save=False,
    pose=None,
):
    """
    Evaluate one sample of a dataset (ds). Calculate PSNR and mAP,
    and visualise different model components for this sample. Additionally,
    1) a different timestep (`t`) can be chosen, which can be different from the
    timestep of the sample (useful for rendering the same view over different
    timesteps).
    """
    if pose is None:
        sample = ds[sample_id]
    else:
        sample = ds.__getitem__(sample_id, pose)
    results = model.render(sample, t=t)
    figure = None

    output_person = "person_weights_sum" in results
    output_transient = "_rgb_fine_transient" in results

    img_wh = tuple(sample["img_wh"].numpy())
    img_gt = ds.x2im(sample["rgbs"], type_="pt")
    img_pred = ds.x2im(results["rgb_fine"][:, :3], type_="pt")

    mask_stat = ds.x2im(results["_rgb_fine_static"][:, 3])
    if output_transient:
        mask_transient = ds.x2im(results["_rgb_fine_transient"][:, 4])
        mask_pred = mask_transient
        if output_person:
            mask_person = ds.x2im(results["_rgb_fine_person"][:, 5])
            mask_pred = mask_pred + mask_person
        else:
            mask_person = np.zeros_like(mask_transient)

    beta = ds.x2im(results["beta"])
    img_pred_static = ds.x2im(results["rgb_fine_static"][:, :3], type_="pt")
    img_pred_transient = ds.x2im(results["_rgb_fine_transient"][:, :3])
    img_pred_transient_white = ds.x2im(results["transient_rgb_map_white"][:, :3])
    if output_person:
        img_pred_person = ds.x2im(results["_rgb_fine_person"][:, :3])

    if mask_targ is not None:
        average_precision = average_precision_score(
            mask_targ.reshape(-1), mask_pred.reshape(-1)
        )

    depth = visualize_depth(results["depth_fine"].view(img_wh[1], img_wh[0])) 
    static_depth = visualize_depth(results["depth_fine_static"].view(img_wh[1], img_wh[0])) 

    psnr = metrics.psnr(img_pred, img_gt).item()
    psnr_static = metrics.psnr(img_pred_static, img_gt).item()

    if visualise:

        figure, ax = plt.subplots(figsize=(8, 5))
        figure.suptitle(f"Sample: {sample_id}.\n")
        plt.tight_layout()
        plt.subplot(331)
        plt.title("GT")
        if gt_masked is not None:
            plt.imshow(torch.from_numpy(gt_masked))
        else:
            plt.imshow(img_gt)
        plt.axis("off")
        plt.subplot(332)
        plt.title(f"Pred. PSNR: {psnr:.2f}")
        plt.imshow(img_pred.clamp(0, 1))
        plt.axis("off")
        plt.subplot(333)
        plt.axis("off")

        plt.subplot(334)
        plt.title(f"Static. PSNR: {psnr_static:.2f}")
        plt.imshow(img_pred_static)
        plt.axis("off")
        plt.subplot(335)
        plt.title(f"Transient")
        plt.imshow(img_pred_transient)
        plt.axis("off")
        if "_rgb_fine_person" in results:
            plt.subplot(336)
            plt.title("Person")
            plt.axis("off")
            plt.imshow(img_pred_person)
        else:
            plt.subplot(336)
            plt.axis("off")

        plt.subplot(337)
        if mask_targ is not None:
            plt.title(f"Mask. AP: {average_precision:.4f}")
        else:
            plt.title("Mask.")
        plt.imshow(mask_pred)
        plt.axis("off")
        plt.subplot(338)
        plt.title(f"Mask: Transient.")
        plt.imshow(mask_transient)
        plt.axis("off")
        plt.subplot(339)
        plt.title(f"Mask: Person.")
        plt.imshow(mask_person)
        plt.axis("off")

    if visualise and not save:
        plt.show()

    results = {}

    results["figure"] = figure
    results["im_tran"] = img_pred_transient
    results["im_tran_white"] = img_pred_transient_white
    results["im_stat"] = img_pred_static
    results["im_pred"] = img_pred
    results["im_targ"] = img_gt
    results["psnr"] = psnr
    results["mask_pred"] = mask_pred
    results["mask_stat"] = mask_stat
    results["im_depth"] = depth.permute([1,2,0])
    results["im_depth_static"] = static_depth.permute([1,2,0])
    if output_person:
        results["mask_pers"] = mask_person
        results["im_pers"] = img_pred_person
    results["mask_tran"] = mask_transient
    if mask_targ is not None:
        results["average_precision"] = average_precision

    for k in results:
        if k == "figure":
            continue
        if type(results[k]) == torch.Tensor:
            results[k] = results[k].to("cpu")

    return results


def evaluate(
    dataset,
    model,
    mask_loader,
    vis_i=5,
    save_dir="results/test",
    save=False,
    vid=None,
    epoch=None,
    timestep_const=None,
    image_ids=None,
    # freeze_time=None,
):
    """
    Like `evaluate_sample`, but evaluates over all selected image_ids.
    Saves also visualisations and average scores of the selected samples.
    """

    results = {
        k: []
        for k in [
            "avgpre",
            "psnr",
            "masks",
            "out",
            "hp",
        ]
    }

    if image_ids is None:
        image_ids = dataset.img_ids_test

    for i, sample_id in utils.tqdm(enumerate(image_ids), total=len(image_ids)):

        do_visualise = i % vis_i == 0

        mask_targ, im_masked = mask_loader[sample_id]
        # # ignore evaluation if no mask available
        # if mask_targ.sum() == 0:
        #     print(f"No annotations for frame {sample_id}, skipping.")
        #     continue

        results["hp"] = model.hparams
        results["hp"]["git_eval"] = git.Repo(
            search_parent_directories=True
        ).head.object.hexsha

        if timestep_const is not None:
            timestep = sample_id
            sample_id = timestep_const
        else:
            timestep = sample_id

        # load meta and check if time_ids can be used
        with (Path(dataset.root) / 'meta.json').open('r') as f:
            meta_json = json.load(f)
        if "time_ids" in meta_json:
            timestep = meta_json["time_ids"][str(sample_id)]

        # if freeze_time is not None:
        #     timestep = freeze_time

        tqdm.tqdm.write(f"Test sample {i}. Frame {sample_id}. Time {timestep}")

        out = evaluate_sample(
            dataset,
            sample_id,
            model=model,
            t=timestep,
            visualise=do_visualise,
            gt_masked=im_masked,
            mask_targ=mask_targ,
            save=save,
        )

        # save_tags = ['figure', 'im_tran', 'im_stat', 'im_pred', 'im_targ', 'mask_pred', 'mask_stat', 'mask_pers', 'im_pers', 'mask_tran']
        save_tags = ['figure', 'im_stat', 'mask_pred', 'im_pred', 'im_tran_white']  # 'im_depth', 'im_depth_static'] #, 'im_stat', 'im_pred', 'mask_pred']

        if save and do_visualise:
            for save_tag in save_tags:
                if save_tag == 'figure':
                    results_im = utils.plt_to_im(out[save_tag])
                else:
                    results_im = out[save_tag]
                    if torch.is_tensor(results_im):
                        results_im = results_im.cpu().numpy()
                    if results_im.ndim < 3:
                        results_im = results_im[...,None] * np.ones([1,3])
                os.makedirs(f"{save_dir}/per_sample/{save_tag}", exist_ok=True)
                path = f"{save_dir}/per_sample/{save_tag}/{sample_id:05d}.png"
                # import pdb
                # pdb.set_trace()
                if save_tag != 'figure':
                    results_im = np.clip(results_im, 0., 1.)
                plt.imsave(path, results_im)
            
            # save a gif of figure
            # imageio.mimsave(f"{save_dir}/per_sample/figures.gif", figures, fps=30)


        mask_pred = out["mask_pred"]

        results["avgpre"].append(out["average_precision"])

        results["psnr"].append(out["psnr"])
        results["masks"].append([mask_targ, mask_pred])
        results["out"].append(out)

    with imageio.get_writer(f'{save_dir}/per_sample/figures.gif', fps=30, mode='I') as writer:
        for rgb_path in sorted(Path(f"{save_dir}/per_sample/figure/").glob('*.png')):
            image = imageio.imread(rgb_path)
            writer.append_data(image)

    metrics_ = {
        "avgpre": {},
        "psnr": {},
    }
    for metric in metrics_:
        metrics_[metric] = np.array(
            [x for x in results[metric] if not np.isnan(x)]
        ).mean()

    results["metrics"] = metrics_

    if save:
        with open(f"{save_dir}/metrics.txt", "a") as f:
            lines = utils.write_summary(results)
            f.writelines(f"Epoch: {epoch}.\n")
            f.writelines(lines)

    print(f"avgpre: {results['metrics']['avgpre']}, PSNR: {results['metrics']['psnr']}")

    return results
