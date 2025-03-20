import argparse
import os
import shutil
import sys
import time
from functools import partial
import logging

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoProcessor, 
    AutoTokenizer, 
    TrainingArguments, 
    default_data_collator,
    set_seed
)

from model import LISAForCausalLM
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.trainer import GemmaLISATrainer
from utils.utils import (
    DEFAULT_IM_END_TOKEN, 
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    AverageMeter, 
    ProgressMeter, 
    Summary, 
    dict_to_cuda,
    intersectionAndUnionGPU
)

from torchvision.transforms import Compose, ToTensor, RandomResizedCrop, Normalize, InterpolationMode

# 初期化メッセージとログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# GPU/CPU環境の確認
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    logger.info(f"=== GPU環境で実行: 利用可能なGPU {device_count}台 ===")
    for i, name in enumerate(device_names):
        logger.info(f"GPU {i}: {name}")
    logger.info(f"CUDA バージョン: {torch.version.cuda}")
else:
    logger.info("=== CPU環境で実行: GPUが利用できません ===")
    logger.info(f"PyTorch バージョン: {torch.__version__}")

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
    
    # デバッグ・テストモード用のオプションを追加
    parser.add_argument("--debug", action="store_true", default=False, 
                      help="デバッグモード：ADE20kの最初の10例のみを使用")
    parser.add_argument("--debug_samples", type=int, default=10,
                      help="デバッグモードで使用するサンプル数（デフォルト: 10）")
    parser.add_argument("--debug_dataset", type=str, default="ade20k",
                      help="デバッグモードで使用するデータセット（デフォルト: ade20k）")
    
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # シード値を設定して再現性を確保
    set_seed(42)
    
    # デバイス情報の詳細表示
    logger.info("=== トレーニング環境詳細 ===")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        logger.info(f"現在のアクティブデバイス: GPU {current_device} ({torch.cuda.get_device_name(current_device)})")
        logger.info(f"GPU メモリ使用量: {torch.cuda.memory_allocated(current_device)/1024**3:.2f} GB (割り当て済み)")
        logger.info(f"GPU メモリ予約量: {torch.cuda.memory_reserved(current_device)/1024**3:.2f} GB (予約済み)")
        logger.info(f"CUDA バージョン: {torch.version.cuda}")
        
        # 利用可能な最大メモリも表示（可能な場合）
        try:
            gpu_properties = torch.cuda.get_device_properties(current_device)
            total_memory = gpu_properties.total_memory / 1024**3
            logger.info(f"GPU 総メモリ: {total_memory:.2f} GB")
        except:
            pass
    else:
        logger.info("CPU環境での実行: GPU機能は利用できません")
        # CPUスレッド数など、可能であれば表示
        import multiprocessing
        logger.info(f"CPU コア数: {multiprocessing.cpu_count()}")
    logger.info("==========================")

    # トークナイザーの初期化
    logger.info(f"トークナイザーを初期化: {args.version}")
    tokenizer = AutoTokenizer.from_pretrained(
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
    logger.info(f"[SEG]トークンのインデックス: {seg_token_idx}")

    # Gemma3の画像トークンを追加
    gemma_image_token = "<start_of_image>"
    num_added = tokenizer.add_tokens([gemma_image_token], special_tokens=True)
    if num_added > 0:
        gemma_image_token_id = tokenizer.convert_tokens_to_ids(gemma_image_token)
        logger.info(f"Gemma3画像トークン '{gemma_image_token}' を追加しました。ID: {gemma_image_token_id}")
    else:
        logger.info(f"Gemma3画像トークン '{gemma_image_token}' は既に存在しています")
    
    # 標準の画像トークンも追加
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN], 
            special_tokens=True
        )
        logger.info("標準画像トークンを追加しました")

    # モデル引数の設定
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
    
    # 精度設定
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    logger.info("モデルを初期化しています...")
    
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
            
        # トークナイザーでモデルの埋め込み層をリサイズ
        model.resize_token_embeddings(len(tokenizer))
        
        # モデルのサイズを確認
        model_size = sum(p.numel() for p in model.parameters())
        logger.info(f"モデルのパラメータ数: {model_size:,}")
        
        # モデルのデバイス情報を表示
        device_info = next(model.parameters()).device
        logger.info(f"現在のモデルデバイス: {device_info}")
        
        # LISA モジュールを初期化
        logger.info("LISA モジュールを初期化しています...")
        model.initialize_lisa_modules(args)
        
        if not args.eval_only:
            # LoRAの設定
            lora_r = args.lora_r
            if lora_r > 0:
                logger.info(f"LoRAを適用しています (r={lora_r}, alpha={args.lora_alpha})...")
                
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
                logger.info(f"LoRA適用レイヤー: {lora_target_modules}")
                
                peft_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
        
        # デバッグモードの場合、設定を調整
        if args.debug:
            logger.info(f"==== デバッグモード有効: {args.debug_dataset}の最初の{args.debug_samples}例のみを使用 ====")
            # デバッグモードでは少ないステップと短いエポックで学習
            args.epochs = min(args.epochs, 2)
            args.steps_per_epoch = 5
            args.grad_accumulation_steps = 1  # 勾配蓄積を無効化
            # データセット設定を指定されたデバッグデータセットのみに変更
            if args.debug_dataset == "ade20k":
                args.dataset = "sem_seg"
                args.sem_seg_data = "ade20k"
                args.sample_rates = "1"
                args.val_dataset = "ReasonSeg|val"  # 検証データセットも設定
            elif args.debug_dataset == "cocostuff":
                args.dataset = "sem_seg"
                args.sem_seg_data = "cocostuff"
                args.sample_rates = "1"
                args.val_dataset = "ReasonSeg|val"  # 検証データセットも設定
            else:
                # デフォルトはade20k
                args.dataset = "sem_seg"
                args.sem_seg_data = args.debug_dataset
                args.sample_rates = "1"
                args.val_dataset = "ReasonSeg|val"  # 検証データセットも設定
            # 学習率とバッチサイズ調整
            args.batch_size = 1
            logger.info(f"デバッグ設定: エポック={args.epochs}, ステップ数={args.steps_per_epoch}, バッチ={args.batch_size}")
        
        # データセットの作成
        logger.info("データセットを初期化しています...")
        
        if not args.eval_only:
            # トレーニング用画像変換
            train_transform = Compose([
                ToTensor(),
                RandomResizedCrop(
                    size=(896, 896),  # Gemma3の推奨サイズ (SiGLIP ViT用)
                    scale=(0.9, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                          std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            
            train_dataset = HybridDataset(
                args=args,
                tokenizer=tokenizer,
                vision_tower=None,  # Gemma3ではvision_towerは不要
                samples_per_epoch=args.steps_per_epoch * args.batch_size * args.grad_accumulation_steps,
                debug_mode=args.debug,  # デバッグモードのフラグを渡す
                debug_samples=args.debug_samples,  # デバッグサンプル数
                transform=train_transform,
            )
            logger.info(f"トレーニングデータセットのサイズ: {len(train_dataset)}")
        else:
            train_dataset = None

        if args.val_dataset and not args.no_eval:
            # 文字列であることを確認
            val_dataset_str = args.val_dataset
            if isinstance(val_dataset_str, int):
                # 整数の場合は文字列に変換（ReasonSeg|val形式を期待）
                val_dataset_str = "ReasonSeg|val"
                logger.info(f"val_datasetが整数値でした。デフォルト値 '{val_dataset_str}' を使用します。")
            
            # 検証用画像変換
            val_transform = Compose([
                ToTensor(),
                RandomResizedCrop(
                    size=(896, 896),  # Gemma3の推奨サイズ (SiGLIP ViT用)
                    scale=(0.9, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                          std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            
            # 一時的にコメントアウト - 評価データセットは現在Trainer APIと互換性がありません
            # val_dataset = ValDataset(
            #     args.dataset_dir, tokenizer, args.version, val_dataset_str, args.image_size, transform=val_transform
            # )
            # logger.info(f"検証データセットのサイズ: {len(val_dataset)}")
            
            # 評価データセットの問題を解決するまで一時的にNoneに設定
            val_dataset = None
            logger.warning("評価データセットが一時的に無効化されています。")
        else:
            val_dataset = None
            
        # TrainingArgumentsの設定
        deepspeed_config = None
        if DEEPSPEED_AVAILABLE and os.path.exists(args.deepspeed_config) and args.deepspeed_config != "None":
            deepspeed_config = args.deepspeed_config

        training_args = TrainingArguments(
            output_dir=args.log_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            gradient_accumulation_steps=args.grad_accumulation_steps,
            learning_rate=args.lr,
            weight_decay=0.0,
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_dir=os.path.join(args.log_dir, "logs"),
            logging_steps=1,
            save_steps=args.steps_per_epoch // 2,
            save_total_limit=3,
            # 一時的に評価を無効化
            evaluation_strategy="no",  # "steps" if val_dataset is not None else "no",
            eval_steps=None,  # args.steps_per_epoch // 2 if val_dataset is not None else None,
            fp16=args.precision == "fp16",
            bf16=args.precision == "bf16",
            dataloader_num_workers=args.workers,
            local_rank=args.local_rank,
            remove_unused_columns=False,  # 画像データを保持するため
            report_to=["tensorboard"],
            label_names=["labels", "mask_labels"],  # モデルへの入力として渡す特別なキー
            deepspeed=deepspeed_config,
        )
        
        # カスタムTrainerのインスタンス化
        trainer = GemmaLISATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=partial(collate_fn, tokenizer=tokenizer),
            tokenizer=tokenizer,
            # カスタム損失の重み
            ce_loss_weight=args.ce_loss_weight,
            bce_loss_weight=args.bce_loss_weight,
            dice_loss_weight=args.dice_loss_weight,
        )
        
        # 学習またはモデルのロード
        # チェックポイントの復元
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info(f"チェックポイントを読み込みます: {args.resume}")
                # Trainer.train() に前回のチェックポイントを渡して続きから学習
                trainer_path = args.resume
            else:
                logger.warning(f"チェックポイントが見つかりません: {args.resume}")
                trainer_path = None
        elif args.auto_resume:
            # 自動的に最新のチェックポイントを探す
            latest = trainer._get_checkpoint_path()
            if latest:
                logger.info(f"最新のチェックポイントを読み込みます: {latest}")
                trainer_path = latest
            else:
                logger.info("チェックポイントが見つからないため、初めから学習します")
                trainer_path = None
        else:
            trainer_path = None
            
        # 評価モードまたはトレーニングモードを実行
        if args.eval_only:
            if val_dataset:
                logger.info("評価を開始します...")
                metrics = trainer.evaluate(eval_dataset=val_dataset)
                logger.info(f"評価結果: {metrics}")
            else:
                logger.warning("検証データセットが指定されていないため、評価をスキップします")
        else:
            logger.info("トレーニングを開始します...")
            
            # トレーニング開始直前のデバイス・メモリ情報を表示
            train_device = next(model.parameters()).device
            logger.info(f"トレーニング実行デバイス: {train_device}")
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i} メモリ状態: "
                               f"割り当て済み={torch.cuda.memory_allocated(i)/1024**3:.2f}GB, "
                               f"予約済み={torch.cuda.memory_reserved(i)/1024**3:.2f}GB")
            
            # トレーニング実行
            trainer.train(resume_from_checkpoint=trainer_path)
            
            # 最終モデルを保存
            trainer.save_model()
            logger.info(f"最終モデルを保存しました: {args.log_dir}")
            
    except Exception as e:
        logger.exception(f"エラーが発生しました: {e}")
        raise e


if __name__ == "__main__":
    main(sys.argv[1:])
