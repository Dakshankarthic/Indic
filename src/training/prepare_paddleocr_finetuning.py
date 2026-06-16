"""
Prepare PaddleOCR Recognition Fine-Tuning from Labeled Data.

After you label the CSV from extract_line_crops_for_labeling.py,
this script:
1. Splits data into train/val sets
2. Generates the character dictionary (dict.txt)
3. Creates PaddleOCR-format ground truth files
4. Creates the training YAML config
5. Prints the exact training command to run

Usage:
    python prepare_paddleocr_finetuning.py --labeled_csv labeling_data/labels.csv --output finetune_data
"""

import os
import sys
import csv
import yaml
import random
import shutil
import argparse
from pathlib import Path
from collections import Counter


def load_labeled_data(csv_path):
    """Load labeled CSV, skip empty labels."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '').strip()
            if text:  # Only include labeled rows
                rows.append(row)
    return rows


def build_dictionary(rows):
    """Extract all unique characters from labeled text."""
    chars = set()
    for row in rows:
        for ch in row['text']:
            chars.add(ch)
    # Sort for deterministic output
    return sorted(chars)


def main():
    parser = argparse.ArgumentParser(description="Prepare PaddleOCR fine-tuning data")
    parser.add_argument("--labeled_csv", required=True, help="Path to labeled CSV file")
    parser.add_argument("--crops_dir", default=None, help="Directory with line crop images (default: same dir as CSV / line_crops)")
    parser.add_argument("--output", default="finetune_data", help="Output directory for training data")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio (default 0.1)")
    args = parser.parse_args()

    csv_path = Path(args.labeled_csv)
    if args.crops_dir:
        crops_dir = Path(args.crops_dir)
    else:
        crops_dir = csv_path.parent / "line_crops_raw"  # Use raw color crops for training
    
    out_dir = Path(args.output)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load labeled data
    rows = load_labeled_data(csv_path)
    if not rows:
        print("ERROR: No labeled rows found! Open labels.csv and type the text for each line crop.")
        sys.exit(1)

    print(f"Found {len(rows)} labeled line crops.")

    # 2. Build character dictionary
    chars = build_dictionary(rows)
    dict_path = out_dir / "dict.txt"
    with open(dict_path, 'w', encoding='utf-8') as f:
        for ch in chars:
            f.write(ch + '\n')
    print(f"Dictionary: {len(chars)} unique characters -> {dict_path}")

    # 3. Shuffle and split into train/val
    random.seed(42)
    random.shuffle(rows)
    val_count = max(1, int(len(rows) * args.val_split))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]

    print(f"Train: {len(train_rows)} samples, Val: {len(val_rows)} samples")

    # 4. Copy images and create ground truth files
    def write_gt(rows_subset, target_dir, gt_path):
        with open(gt_path, 'w', encoding='utf-8') as f:
            for row in rows_subset:
                fname = row['file_name']
                text = row['text'].strip()
                src = crops_dir / fname
                dst = target_dir / fname
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                elif not src.exists():
                    # Try the clean crops dir
                    alt_src = csv_path.parent / "line_crops" / fname
                    if alt_src.exists():
                        shutil.copy2(alt_src, dst)
                    else:
                        print(f"  Warning: Image not found: {src}")
                        continue
                f.write(f"{fname}\t{text}\n")

    train_gt = out_dir / "rec_gt_train.txt"
    val_gt = out_dir / "rec_gt_val.txt"
    write_gt(train_rows, train_dir, train_gt)
    write_gt(val_rows, val_dir, val_gt)

    # 5. Create PaddleOCR training config YAML
    config = {
        'Global': {
            'use_gpu': True,
            'epoch_num': 100,
            'log_smooth_window': 20,
            'print_batch_step': 10,
            'save_model_dir': str(out_dir / "trained_model"),
            'save_epoch_step': 10,
            'eval_batch_step': [0, 50],
            'cal_metric_during_train': True,
            'pretrained_model': None,  # Will be filled by command line
            'checkpoints': None,
            'save_inference_dir': str(out_dir / "inference_model"),
            'use_visualdl': False,
            'infer_img': str(val_dir),
            'character_dict_path': str(dict_path),
            'character_type': 'ch',
            'max_text_length': 100,
            'use_space_char': True,
        },
        'Optimizer': {
            'name': 'Adam',
            'beta1': 0.9,
            'beta2': 0.999,
            'lr': {
                'name': 'Cosine',
                'learning_rate': 0.0005,
                'warmup_epoch': 5,
            },
            'regularizer': {
                'name': 'L2',
                'factor': 0.00001,
            }
        },
        'Architecture': {
            'model_type': 'rec',
            'algorithm': 'SVTR_LCNet',
            'Transform': None,
            'Backbone': {
                'name': 'MobileNetV1Enhance',
                'scale': 0.5,
                'last_conv_stride': [1, 2],
                'last_pool_type': 'avg',
            },
            'Head': {
                'name': 'MultiHead',
                'head_list': [
                    {
                        'CTCHead': {
                            'Neck': {'name': 'svtr', 'dims': 64, 'depth': 2, 'hidden_dims': 120, 'use_guide': True},
                            'Head': {'fc_decay': 0.00001}
                        }
                    },
                    {
                        'SARHead': {
                            'enc_dim': 512,
                            'max_text_length': 100,
                        }
                    }
                ]
            }
        },
        'Loss': {
            'name': 'MultiLoss',
            'loss_config_list': [
                {'CTCLoss': None},
                {'SARLoss': None}
            ]
        },
        'PostProcess': {
            'name': 'CTCLabelDecode'
        },
        'Metric': {
            'name': 'RecMetric',
            'main_indicator': 'acc',
        },
        'Train': {
            'dataset': {
                'name': 'SimpleDataSet',
                'data_dir': str(train_dir),
                'label_file_list': [str(train_gt)],
                'transforms': [
                    {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                    {'RecConAug': {'prob': 0.5, 'ext_data_num': 0, 'image_shape': [48, 320, 3]}},
                    {'RecAug': {}},
                    {'MultiLabelEncode': None},
                    {'RecResizeImg': {'image_shape': [3, 48, 320]}},
                    {'KeepKeys': {'keep_keys': ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']}},
                ]
            },
            'loader': {
                'shuffle': True,
                'batch_size_per_card': 32,
                'drop_last': True,
                'num_workers': 4,
            }
        },
        'Eval': {
            'dataset': {
                'name': 'SimpleDataSet',
                'data_dir': str(val_dir),
                'label_file_list': [str(val_gt)],
                'transforms': [
                    {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                    {'MultiLabelEncode': None},
                    {'RecResizeImg': {'image_shape': [3, 48, 320]}},
                    {'KeepKeys': {'keep_keys': ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']}},
                ]
            },
            'loader': {
                'shuffle': False,
                'drop_last': False,
                'batch_size_per_card': 32,
                'num_workers': 4,
            }
        }
    }

    config_path = out_dir / "rec_finetune.yml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # 6. Print instructions
    print(f"\n{'='*70}")
    print("PaddleOCR Fine-Tuning Data Ready!")
    print(f"{'='*70}")
    print(f"\n  Config:      {config_path}")
    print(f"  Dictionary:  {dict_path} ({len(chars)} chars)")
    print(f"  Train GT:    {train_gt} ({len(train_rows)} samples)")
    print(f"  Val GT:      {val_gt} ({len(val_rows)} samples)")
    print(f"\n  TO START TRAINING, RUN:")
    print(f"  ─────────────────────────────────────")
    print(f"  First, clone PaddleOCR if not done:")
    print(f"    git clone https://github.com/PaddlePaddle/PaddleOCR.git")
    print(f"")
    print(f"  Then download the pretrained Hindi rec model:")
    print(f"    # From PaddleOCR model list, get the hi (Hindi) rec model")
    print(f"")
    print(f"  Then run training:")
    print(f"    cd PaddleOCR")
    print(f"    python tools/train.py \\")
    print(f"        -c {config_path.resolve()} \\")
    print(f"        -o Global.pretrained_model=./pretrained_hi_rec/best_accuracy")
    print(f"")
    print(f"  AFTER TRAINING, update paddle_ocr_step2.py to use your model:")
    print(f"    ocr = PaddleOCR(rec_model_dir='{out_dir / 'trained_model'}', ...)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
