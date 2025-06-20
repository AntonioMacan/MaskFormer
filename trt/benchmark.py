import os
from pathlib import Path
import argparse
import numpy as np
import torch
from time import perf_counter
from torch.nn.functional import interpolate
from torchmetrics import JaccardIndex
from panopticapi.evaluation import PQStat

from trt.engines.tensorrt_engine import TensorRTEngine
from trt.engines.tensorrt_onnx_engine import TensorRTONNXEngine
from trt.engines.pytorch_engine import PyTorchEngine
from trt.data.utils import prepare_data


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark MaskFormer inference')
    parser.add_argument('--engine', type=str, choices=['pytorch', 'tensorrt', 'tensorrt-onnx'], default='pytorch')
    parser.add_argument('--weights', type=str, default="weights/maskformer_R101_bs16_90k/model_final_38c00c.pkl")
    parser.add_argument('--dataset_path', type=str, default='datasets/cityscapes')
    parser.add_argument('--num_images', type=int, default=250)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--engine_cache_dir', type=str, default='trt/engine_cache')
    parser.add_argument('--eval_pq', action='store_true', help='Evaluate Panoptic Quality (PQ)')
    return parser.parse_args()


def compute_pixel_accuracy(pred1, pred2):
    assert pred1.shape == pred2.shape, "Shape mismatch"
    return np.mean(pred1 == pred2)


def evaluate_miou(engine, data_loader, num_classes=19):
    jaccard = JaccardIndex(task='multiclass', num_classes=num_classes).to('cuda')
    total_pixels = 0
    correct_pixels = 0
    print("[INFO] Evaluating mIoU on validation set...")

    for images, paths, labels in data_loader:
        input_data = engine.prepare_input(images)
        output = engine.run(input_data)
        logits_batch = engine.get_logits_from_output(output)

        for logits, gt in zip(logits_batch, labels):
            pred = torch.from_numpy(np.argmax(logits, axis=0)).to(torch.int64)
            gt = gt.to(torch.int64)

            if pred.shape != gt.shape:
                pred = interpolate(
                    torch.from_numpy(logits).unsqueeze(0),
                    size=gt.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                pred = torch.argmax(pred, dim=0)

            pred = pred.flatten().to('cuda')
            gt = gt.flatten().to('cuda')
            valid = gt != 255
            if valid.sum() == 0:
                continue

            pred_valid = pred[valid]
            gt_valid = gt[valid]

            pred_valid = torch.clamp(pred_valid, 0, num_classes - 1)
            gt_valid = torch.clamp(gt_valid, 0, num_classes - 1)

            jaccard.update(pred_valid, gt_valid)
            correct_pixels += (pred_valid == gt_valid).sum().item()
            total_pixels += valid.sum().item()

    miou = jaccard.compute().item() * 100
    pixel_acc = (correct_pixels / total_pixels) * 100
    return miou, pixel_acc


def evaluate_pq(engine, data_loader, num_classes=19, ignore_label=255):
    print("[INFO] Evaluating PQ on validation set...")
    pq_stat = PQStat()
    OFFSET = 256 * 256 * 256

    for images, paths, labels in data_loader:
        input_data = engine.prepare_input(images)
        output = engine.run(input_data)
        logits_batch = engine.get_logits_from_output(output)

        for logits, gt in zip(logits_batch, labels):
            pred = np.argmax(logits, axis=0).astype(np.int64)
            gt = gt.cpu().numpy().astype(np.int64)

            if pred.shape != gt.shape:
                pred = torch.from_numpy(logits).unsqueeze(0)
                pred = interpolate(pred, size=gt.shape, mode='nearest')
                pred = torch.argmax(pred.squeeze(0), dim=0).numpy().astype(np.int64)

            valid = gt != ignore_label
            pred[~valid] = ignore_label

            gt_ids = np.unique(gt[valid])
            pred_ids = np.unique(pred[valid])

            gt_ann = {'segments_info': []}
            for id_ in gt_ids:
                mask = gt == id_
                gt_ann['segments_info'].append({
                    "id": id_,
                    "category_id": id_,
                    "area": mask.sum(),
                    "iscrowd": 0
                })

            pred_ann = {'segments_info': []}
            for id_ in pred_ids:
                mask = pred == id_
                pred_ann['segments_info'].append({
                    "id": id_,
                    "category_id": id_,
                    "area": mask.sum(),
                    "iscrowd": 0
                })

            pan_gt_pred = gt.astype(np.uint64) * OFFSET + pred.astype(np.uint64)
            intersect_ids, intersect_counts = np.unique(pan_gt_pred, return_counts=True)

            gt_segms = {el['id']: el for el in gt_ann['segments_info']}
            pred_segms = {el['id']: el for el in pred_ann['segments_info']}

            matched_gt = set()
            matched_pred = set()

            for label, count in zip(intersect_ids, intersect_counts):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                if gt_id == ignore_label or pred_id == ignore_label:
                    continue
                if gt_id not in gt_segms or pred_id not in pred_segms:
                    continue
                if gt_id != pred_id:
                    continue
                union = gt_segms[gt_id]['area'] + pred_segms[pred_id]['area'] - count
                iou = count / union
                if iou > 0.5:
                    pq_stat[gt_id].tp += 1
                    pq_stat[gt_id].iou += iou
                    matched_gt.add(gt_id)
                    matched_pred.add(pred_id)

            for gt_id in gt_segms:
                if gt_id not in matched_gt:
                    pq_stat[gt_id].fn += 1
            for pred_id in pred_segms:
                if pred_id not in matched_pred:
                    pq_stat[pred_id].fp += 1

    pq_results, _ = pq_stat.pq_average(
        {i: {"id": i, "isthing": 0, "name": str(i)} for i in range(num_classes)},
        isthing=False
    )
    return pq_results


def run_benchmark(engine, data_loader, n_warmup=50, n_inference=180):
    loader_iter = iter(data_loader)

    print(f"[INFO] Running {n_warmup} warm-up inferences (not measured).")
    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                images, _, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                images, _, _ = next(loader_iter)
            input_data = engine.prepare_input(images)
            _ = engine.run(input_data)

    torch.cuda.synchronize()

    print(f"[INFO] Measuring time for {n_inference} inferences...")
    times = []
    with torch.no_grad():
        for _ in range(n_inference):
            try:
                images, _, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                images, _, _ = next(loader_iter)

            input_data = engine.prepare_input(images)
            start_time = perf_counter()
            _ = engine.run(input_data)
            end_time = perf_counter()
            times.append(end_time - start_time)

    times = np.array(times)
    return {
        'total_time': times.sum(),
        'mean_time': times.mean(),
        'fps': 1.0 / times.mean()
    }


def main():
    args = parse_args()
    sizes = [(128, 256), (256, 512), (512, 1024)]
    results = []

    for size in sizes:
        H, W = size
        print(f"\n=== Benchmarking {args.engine} at resolution {H}x{W} ===")

        model_path = (
            f"trt/onnx_models/trt_model_{H}x{W}.onnx"
            if args.engine in ['tensorrt', 'tensorrt-onnx']
            else args.weights
        )

        if args.engine == 'pytorch':
            engine = PyTorchEngine()
            build_time = engine.load_model(model_path, image_size=size)
        elif args.engine == 'tensorrt':
            cache_path = os.path.join(args.engine_cache_dir, f"{args.engine}_{H}x{W}.plan")
            engine, build_time = TensorRTEngine.load_or_build(model_path, cache_path)
        elif args.engine == 'tensorrt-onnx':
            engine = TensorRTONNXEngine()
            build_time = engine.load_model(model_path)
        else:
            raise ValueError("Unsupported engine")

        print(f"[INFO] Engine: {engine.name}, built in {build_time:.2f}s")

        data_loader = prepare_data(
            args.dataset_path,
            'val',
            num_images=args.num_images,
            image_size=size
        )

        bench_result = run_benchmark(engine, data_loader, args.warmup, args.iterations)
        bench_result['size'] = size

        if args.engine != 'pytorch':
            print("[INFO] Comparing predictions with PyTorch baseline...")
            torch_engine = PyTorchEngine()
            torch_engine.load_model(args.weights, image_size=size)

            torch_loader = prepare_data(args.dataset_path, 'val', image_size=size)
            images, paths = next(iter(torch_loader))[:2]

            input_torch = torch_engine.prepare_input(images)
            out_torch = torch_engine.run(input_torch)
            preds_torch = [np.argmax(t.cpu().numpy(), axis=0) for t in out_torch]

            input_target = engine.prepare_input(images)
            out_target = engine.run(input_target)
            logits_target = engine.get_logits_from_output(out_target)
            preds_target = [np.argmax(t, axis=0) for t in logits_target]

            pixel_accs = [
                compute_pixel_accuracy(p1, p2)
                for p1, p2 in zip(preds_torch, preds_target)
            ]
            bench_result['engine_agreement'] = sum(pixel_accs) / len(pixel_accs)

        eval_loader = prepare_data(
            args.dataset_path, 'val',
            image_size=size,
            batch_size=1,
        )
        # miou, pixacc = evaluate_miou(engine, eval_loader)
        # bench_result['miou'] = miou
        # bench_result['pixel_acc_eval'] = pixacc

        if args.eval_pq:
            pq_res = evaluate_pq(engine, eval_loader)
            bench_result['pq'] = pq_res['pq'] * 100
            bench_result['sq'] = pq_res['sq'] * 100
            bench_result['rq'] = pq_res['rq'] * 100

        results.append(bench_result)
        print("=== Done ===")

    for result in results:
        print(f"\n[RESULT] Image size: {result['size']}")
        print(f"Inference time: {result['total_time']:.2f} s")
        print(f"Mean time: {result['mean_time'] * 1000:.2f} ms")
        print(f"FPS: {result['fps']:.2f}")
        # print(f"mIoU: {result['miou']:.2f}%")
        # if 'engine_agreement' in result:
            # print(f"Pixel Agreement vs PyTorch: {result['engine_agreement'] * 100:.2f}%")
        if 'pq' in result:
            print(f"PQ: {result['pq']:.2f}%, SQ: {result['sq']:.2f}%, RQ: {result['rq']:.2f}%")


if __name__ == "__main__":
    main()
