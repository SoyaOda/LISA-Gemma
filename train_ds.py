import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor

from model.GemmaLISA import GemmaLISAForCausalLM, LISAForCausalLM
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA-Gemma3 Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="google/gemma-3-4b-it", type=str,
        help="Gemma3モデルのパス"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff",  # mapillaryを除外
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="H:/download/LISA-dataset/dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa-gemma3", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="gemma_v1",
        type=str,
        choices=["gemma_v1", "llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--deepspeed_config", default="ds_config.json", type=str)
    
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    print(f"モデル名: {args.version}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    print(f"[SEG]トークンのインデックス: {seg_token_idx}")

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
        print("画像トークンを追加しました")

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    print("モデルを初期化しています...")
    
    # 量子化設定
    if args.load_in_8bit or args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        
        if args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        model_args["quantization_config"] = quantization_config
    
    # モデル読み込み
    try:
        model = LISAForCausalLM.from_pretrained(
            args.version, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            **model_args
        )
        
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # モデルのサイズを確認
        model_size = sum(p.numel() for p in model.parameters())
        print(f"モデルのパラメータ数: {model_size:,}")
        
        if not args.eval_only:
            # LoRAの設定
            lora_r = args.lora_r
            if lora_r > 0:
                print(f"LoRAを適用しています (r={lora_r}, alpha={args.lora_alpha})...")
                
                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in [
                                        "visual_model",
                                        "vision_tower",
                                        "mm_projector",
                                        "text_hidden_fcs",
                                    ]
                                ]
                            )
                            and any([x in name for x in args.lora_target_modules.split(",")])
                        ):
                            lora_module_names.add(name)
                    return sorted(list(lora_module_names))

                lora_target_modules = find_linear_layers(model, args.lora_target_modules.split(","))
                print(f"LoRA適用レイヤー: {lora_target_modules}")
                
                peft_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, peft_config)
        
        # データセットの作成
        print("データセットを初期化しています...")
        
        if not args.eval_only:
            train_dataset = HybridDataset(
                args=args,
                tokenizer=tokenizer,
                vis_processor=None,
                vis_processor_gemma=None,
                conv_type=args.conv_type,
                task_list=args.dataset.split("||"),
                sample_rate=[float(s) for s in args.sample_rates.split(",")],
            )
            print(f"トレーニングデータセットのサイズ: {len(train_dataset)}")
        else:
            train_dataset = None

        if args.val_dataset:
            val_dataset = ValDataset(
                args=args,
                tokenizer=tokenizer,
                vis_processor=None,
                vis_processor_gemma=None,
                conv_type=args.conv_type,
                task=args.val_dataset.split("|")[0],
                split=args.val_dataset.split("|")[1],
            )
            print(f"検証データセットのサイズ: {len(val_dataset)}")
        else:
            val_dataset = None

        # データローダーの設定
        if not args.eval_only:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                shuffle=True,
                seed=42,
                drop_last=True,
                rank=args.local_rank,
                num_replicas=torch.cuda.device_count(),
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                sampler=train_sampler,
                collate_fn=partial(
                    collate_fn, tokenizer=tokenizer, conv_type=args.conv_type
                ),
            )
        else:
            train_loader = None
            train_sampler = None

        if val_dataset is not None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                shuffle=False,
                seed=42,
                drop_last=False,
                rank=args.local_rank,
                num_replicas=torch.cuda.device_count(),
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn, tokenizer=tokenizer, conv_type=args.conv_type
                ),
            )
        else:
            val_loader = None
            val_sampler = None

        # オプティマイザーの設定
        if not args.eval_only:
            opt = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                betas=(args.beta1, args.beta2),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, args.epochs * args.steps_per_epoch
            )
        else:
            opt = None
            scheduler = None

        # チェックポイント読み込み
        if args.resume:
            if os.path.isfile(args.resume):
                print(f"=> チェックポイントを読み込んでいます '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location="cpu")
                args.start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
                opt.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                print(f"=> エポック {checkpoint['epoch']} から再開します")
            else:
                print(f"=> チェックポイントが見つかりません '{args.resume}'")
        elif args.auto_resume:
            latest_checkpoint = os.path.join(args.log_dir, "checkpoint_latest.pt")
            if os.path.isfile(latest_checkpoint):
                print(f"=> 最新のチェックポイントを読み込んでいます '{latest_checkpoint}'")
                checkpoint = torch.load(latest_checkpoint, map_location="cpu")
                args.start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
                opt.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                print(f"=> エポック {checkpoint['epoch']} から再開します")

        # DeepSpeedの設定
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.grad_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "betas": [args.beta1, args.beta2],
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": args.lr,
                    "warmup_num_steps": 100,
                    "total_num_steps": args.epochs * args.steps_per_epoch,
                },
            },
            "fp16": {
                "enabled": args.precision == "fp16",
            },
            "bf16": {
                "enabled": args.precision == "bf16",
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "reduce_scatter": True,
                "contiguous_gradients": True,
            },
        }

        # DeepSpeedでモデルを初期化
        model, opt, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=opt,
            config=ds_config,
            lr_scheduler=scheduler,
            dist_init_required=True,
        )

        # 評価のみの場合
        if args.eval_only:
            evaluate(val_loader, model, tokenizer, args, val_dataset, 0, writer)
            return

        # トレーニングループ
        print("トレーニングを開始します...")
        for epoch in range(args.start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # トレーニング
            train_one_epoch(
                train_loader,
                model,
                tokenizer,
                opt,
                scheduler,
                epoch,
                args,
                writer,
            )

            # チェックポイント保存
            if args.local_rank == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    False,
                    args.log_dir,
                )

            # 評価
            if not args.no_eval and val_loader is not None:
                evaluate(val_loader, model, tokenizer, args, val_dataset, epoch, writer)

        print("トレーニングが完了しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


def train_one_epoch(train_loader, model, tokenizer, optimizer, scheduler, epoch, args, writer):
    """1エポックのトレーニング"""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CE_Loss", ":.4f")
    mask_bce_losses = AverageMeter("BCE_Loss", ":.4f")
    mask_dice_losses = AverageMeter("Dice_Loss", ":.4f")
    mask_losses = AverageMeter("Mask_Loss", ":.4f")
    
    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, data_time, losses, ce_losses, mask_bce_losses, mask_dice_losses, mask_losses],
        prefix=f"Epoch: [{epoch}]",
    )
    
    # モデルをトレーニングモードに設定
    model.train()
    
    end = time.time()
    for i, input_dict in enumerate(train_loader):
        if i >= args.steps_per_epoch:
            break
        
        # 計測：データロード時間
        data_time.update(time.time() - end)
        
        # データをGPUに転送
        input_dict = dict_to_cuda(input_dict)
        
        # フォワードパス
        outputs = model(**input_dict)
        loss = outputs["loss"]
        ce_loss = outputs["ce_loss"] if "ce_loss" in outputs else 0
        mask_bce_loss = outputs["mask_bce_loss"] if "mask_bce_loss" in outputs else 0
        mask_dice_loss = outputs["mask_dice_loss"] if "mask_dice_loss" in outputs else 0
        mask_loss = outputs["mask_loss"] if "mask_loss" in outputs else 0
        
        # バックワードパス
        model.backward(loss)
        model.step()
        
        # ロス値を記録
        losses.update(loss.item(), input_dict["input_ids"].size(0))
        ce_losses.update(ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss, input_dict["input_ids"].size(0))
        mask_bce_losses.update(mask_bce_loss.item() if torch.is_tensor(mask_bce_loss) else mask_bce_loss, input_dict["input_ids"].size(0))
        mask_dice_losses.update(mask_dice_loss.item() if torch.is_tensor(mask_dice_loss) else mask_dice_loss, input_dict["input_ids"].size(0))
        mask_losses.update(mask_loss.item() if torch.is_tensor(mask_loss) else mask_loss, input_dict["input_ids"].size(0))
        
        # 計測：バッチ処理時間
        batch_time.update(time.time() - end)
        end = time.time()
        
        # 進捗表示
        if i % args.print_freq == 0:
            progress.display(i)
        
        # TensorBoardにロス値を記録
        if writer is not None and args.local_rank == 0:
            step = epoch * args.steps_per_epoch + i
            writer.add_scalar("train/loss", losses.val, step)
            writer.add_scalar("train/ce_loss", ce_losses.val, step)
            writer.add_scalar("train/mask_bce_loss", mask_bce_losses.val, step)
            writer.add_scalar("train/mask_dice_loss", mask_dice_losses.val, step)
            writer.add_scalar("train/mask_loss", mask_losses.val, step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)


def evaluate(val_loader, model, tokenizer, args, val_dataset, epoch, writer):
    """モデルの評価"""
    batch_time = AverageMeter("Time", ":6.3f")
    
    # 評価指標
    intersection_meter = AverageMeter("Intersection", ":6.3f")
    union_meter = AverageMeter("Union", ":6.3f")
    
    # モデルを評価モードに設定
    model.eval()
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, intersection_meter, union_meter],
        prefix="Eval: ",
    )
    
    with torch.no_grad():
        end = time.time()
        for i, input_dict in enumerate(val_loader):
            # データをGPUに転送
            input_dict = dict_to_cuda(input_dict)
            
            # 推論
            input_ids = input_dict["input_ids"]
            pixel_values = input_dict["pixel_values"]
            images = input_dict["images"]
            labels = input_dict.get("labels", None)
            
            # リサイズサイズと元のサイズを取得
            resize_list = input_dict.get("resize_list", None)
            original_size_list = input_dict.get("original_size_list", None)
            
            # 評価用の推論
            results = model.evaluate(
                pixel_values=pixel_values,
                images=images,
                input_ids=input_ids,
                resize_list=resize_list, 
                original_size_list=original_size_list,
                max_new_tokens=128,
                tokenizer=tokenizer,
            )
            
            # 予測結果を取得
            pred_masks = results["masks"] if "masks" in results else None
            
            if pred_masks is not None and "gt_masks" in input_dict:
                # セグメンテーション評価指標の計算
                gt_masks = input_dict["gt_masks"]
                for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                    if pred_mask is not None and gt_mask is not None:
                        pred_mask = pred_mask.bool()
                        gt_mask = gt_mask.bool()
                        intersection, union = intersectionAndUnionGPU(
                            pred_mask.float(), gt_mask.float(), 2
                        )
                        intersection_meter.update(intersection[1].item())
                        union_meter.update(union[1].item())
            
            # 計測：バッチ処理時間
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 進捗表示
            if i % args.print_freq == 0:
                progress.display(i)
        
        # IoU計算
        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        
        # 結果表示
        print(f"Validation IoU: {iou:.4f}")
        
        # TensorBoardに評価結果を記録
        if writer is not None and args.local_rank == 0:
            writer.add_scalar("val/IoU", iou, epoch)
            
        return iou


def save_checkpoint(state, is_best, log_dir, filename="checkpoint_latest.pt"):
    """チェックポイントの保存"""
    torch.save(state, os.path.join(log_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(log_dir, filename), os.path.join(log_dir, "checkpoint_best.pt")
        )


if __name__ == "__main__":
    main(sys.argv[1:])
