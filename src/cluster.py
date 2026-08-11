

import argparse
import random
import os
from timeit import default_timer as timer

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel 


from utils import data_setup, model, engine, utils, loss_function, data_preprocess, augmentation_utils

from utils.data_setup import KmerTokenizer 


os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Argument Parser
    parser = argparse.ArgumentParser()
    # --- 原有参数 ---
    parser.add_argument("--dataset", default="01_Cypriniformes.fasta", type=str, help="choose a fasta file in data directory")
    parser.add_argument("--k", default=6, type=int, help="k-mer size for FCGR (Visual)")
    parser.add_argument("--weak_mutation_rate", default=1e-4, type=float, help="Weak mutation rate")
    parser.add_argument("--strong_mutation_rate", default=1e-2, type=float, help="Strong mutation rate")
    parser.add_argument("--weak_fragmentation_perc", default=None, type=float, help="Weak frag perc")
    parser.add_argument("--strong_fragmentation_perc", default=None, type=float, help="Strong frag perc")
    parser.add_argument("--number_of_pairs", default=1, type=int, help="Number of augmented data pairs")
    parser.add_argument("--number_of_models", default=5, type=int, help="number of models")
    
    # --- 学习率相关参数 ---
    parser.add_argument("--lr", default=7e-5, type=float, help="Base learning rate (CNN)")
    parser.add_argument("--lora_lr", default=1e-4, type=float, help="Learning rate for LoRA parameters") 
    
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--temp_ins", default=0.1, type=float, help="instance temperature")
    parser.add_argument("--temp_clu", default=1.0, type=float, help="cluster temperature")
    
    # --- Epochs ---
    parser.add_argument("--warmup_epochs", default=20, type=int, help="Epochs to freeze LLM and train only visual branch")
    parser.add_argument("--num_epochs", default=150, type=int, help="Total joint training epochs")
    parser.add_argument("--finetune_llm", action="store_true", help="If set, unfreeze and train LoRA parameters.")
    parser.add_argument("--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("--embedding_dim", default=512, type=int, help="embedding dimension")
    parser.add_argument("--feature_dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--random_seed", default=0)
    parser.add_argument("--weight", default=0.7)
    
    # --- LLM 参数 ---
    parser.add_argument("--llm_path", default="InstaDeepAI/nucleotide-transformer-500m-human-ref", type=str)
    parser.add_argument("--max_len", default=50, type=int, help="max token seq len for LLM") 

    # --- Fusion 参数 ---
    parser.add_argument("--fusion_dim", default=512, type=int, help="Dimension after fusing Visual + Text")
    parser.add_argument("--fusion_type", choices=["gate","cross_attn"], default='gate', help="Fusion module type")
    
    # --- 开关参数 ---
    parser.add_argument("--use_llm", action="store_true", help="Enable LLM fusion with LoRA.")

    # --- Pretrained Visual Loading ---
    parser.add_argument("--pretrained_visual", default=None, type=str, help="Path to a saved visual checkpoint")
    parser.add_argument("--freeze_visual", action="store_true", help="If set, freeze backbone after loading")

    args = parser.parse_args()
    print(args)

    # ####################################################################################################################
    print("Reading the data... ")
    records_df = data_preprocess.read_fasta("data/" + args.dataset)
    class_names = sorted(records_df.label.unique())
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    
    raw_sequences_all = records_df['sequence'].tolist()

    # ####################################################################################################################
    # Generate Augmented Data Pairs
    print("Generating Augmented Data Pairs...")
    random.seed(42)
    np.random.seed(42)
    
    if args.weak_mutation_rate is not None:
        X_train, X_test, y_test, X_train_seqs = augmentation_utils.generate_pairs(
            data=records_df,
            class_to_idx=class_to_idx,
            k=args.k, 
            number_of_pairs=args.number_of_pairs,
            mutation_rate_weak=args.weak_mutation_rate,
            mutation_rate_strong=args.strong_mutation_rate
        )
    elif args.weak_fragmentation_perc is not None:
        X_train, X_test, y_test, X_train_seqs = augmentation_utils.generate_pairs(
            data=records_df,
            class_to_idx=class_to_idx,
            k=args.k,
            number_of_pairs=args.number_of_pairs,
            frag_perc_weak=args.weak_fragmentation_perc,
            frag_perc_strong=args.strong_fragmentation_perc
        )
    else:
        raise ValueError("Specify either mutation rates or fragmentation percentages for augmentation.")

    X_train, X_test = utils.data_normalization(X_train, X_test)
    print(f"Class names: {class_names}")
    
    X_test_seqs = raw_sequences_all

    # ####################################################################################################################
 
    tokenizer = None
    if args.use_llm:
        print(f"[Init] LLM Fusion Enabled. Initializing Tokenizer from: {args.llm_path}")
        try:
            print("[Init] Loading original vocabulary...")
            original_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
            original_vocab = original_tokenizer.get_vocab() 
            print("[Init] Initializing Custom KmerTokenizer...")
            tokenizer = KmerTokenizer(k=6, stride=6, vocab=original_vocab)
        except Exception as e:
            print(f"Error loading Tokenizer: {e}")
            exit()
    else:
        print("[Init] LLM Fusion Disabled. Running Pure Visual Baseline.")

    # ####################################################################################################################
    # Create Datasets and DataLoaders
    random.seed(args.random_seed)
    NUM_WORKERS = 1 

    train_data = data_setup.PairSeqData(
        train_pairs=X_train, 
        raw_sequences=X_train_seqs,
        tokenizer=tokenizer, 
        max_len=args.max_len,
        transform=None
    )
    
    test_data = data_setup.SeqData(
        fcgr_images=X_test,
        raw_sequences=X_test_seqs,
        labels=y_test,
        classes=class_names,
        class_to_idx=class_to_idx,
        tokenizer=tokenizer,
        max_len=args.max_len,
        transform=None
    )
    
    train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=NUM_WORKERS, drop_last=False, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=NUM_WORKERS, shuffle=False)

    # ####################################################################################################################
    y_preds = []
    y_probs = []
    
 
    for i in range(args.number_of_models):
        print(f"\n{'='*20} Training model #{i + 1} {'='*20}")
        torch.manual_seed(args.random_seed + i)
        torch.cuda.manual_seed(args.random_seed + i)
        
        # =========================================================================

        # =========================================================================
        dna_llm = None
        if args.use_llm:
            print(f"[Init] Loading FRESH DNA LLM for Model #{i+1}...")
            try:
                dna_llm = AutoModel.from_pretrained(args.llm_path, trust_remote_code=True)
              
                # print(f"🔴 Model max pos: {dna_llm.config.max_position_embeddings} | Args max_len: {args.max_len}")
            except Exception as e:
                print(f"Error loading LLM: {e}")
                exit()
        
      
        backbone_model = model.BackBoneModel(input_shape=1, output_shape=args.embedding_dim)
        

        projector_model = model.Network(
            backbone=backbone_model,
            dna_llm=dna_llm, 
            rep_dim=args.embedding_dim,
            feature_dim=args.feature_dim,
            class_num=len(class_names),
            use_llm=args.use_llm,
            fusion_dim=args.fusion_dim,
            fusion_type=args.fusion_type,
        ).to(device)

        print(f"[Init] Fusion type: {args.fusion_type} | fusion_dim: {args.fusion_dim}")

   
        current_warmup_epochs = args.warmup_epochs

    
        if hasattr(args, 'pretrained_visual') and args.pretrained_visual is not None:
            print(f"[Init] Loading visual checkpoint from: {args.pretrained_visual}")

            
            try:
                ckpt = torch.load(args.pretrained_visual, map_location=device)
            except Exception:
                
                ckpt = torch.load(args.pretrained_visual, map_location=device, weights_only=False)

            sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            model_sd = projector_model.state_dict()
            
           
            loaded_keys = []
            for k, v in sd.items():
                if k.startswith('backbone.') or k.startswith('instance_projector.') or k.startswith('cluster_projector.'):
                    if k in model_sd and model_sd[k].shape == v.shape:
                        model_sd[k] = v
                        loaded_keys.append(k)
            projector_model.load_state_dict(model_sd, strict=False) 
            print(f"[Init] Loaded {len(loaded_keys)} keys into visual stream.")

            if hasattr(args, 'freeze_visual') and args.freeze_visual:
                for name, p in projector_model.named_parameters():
                    if name.startswith('backbone.') or name.startswith('instance_projector.') or name.startswith('cluster_projector.'):
                        p.requires_grad = False
              
                current_warmup_epochs = 0
                print("[Init] Visual stream frozen and warmup skipped.")

        criterion_instance = loss_function.InstanceLoss(args.batch_size, args.temp_ins, device).to(device)
        criterion_cluster = loss_function.ClusterLoss(len(class_names), args.temp_clu, device).to(device)

        start_time = timer()

 
        if args.use_llm and current_warmup_epochs > 0:
            print(f"\n>>> [Phase 1] Pure Visual Warmup (LLM Disconnected) for {current_warmup_epochs} epochs...")
            
           
            for n, p in projector_model.named_parameters():
                if "lora" in n:
                    p.requires_grad = False
            
         
            warmup_params = [p for p in projector_model.parameters() if p.requires_grad]
            optimizer_warmup = torch.optim.Adam(
                warmup_params,
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            
          
            engine.train(
                model=projector_model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer_warmup,
                scheduler=None,
                weight=float(args.weight),
                criterion_instance=criterion_instance,
                criterion_cluster=criterion_cluster,
                epochs=current_warmup_epochs,
                device=device,
                run_visual_only=True 
            )
            print(">>> Warmup Finished.")

        # =========================================================
        # ✅ [New Step] Verify Baseline (Visual Only)
        # =========================================================
     
        temp_mode = projector_model.use_llm
        projector_model.use_llm = False 
        _, _, _, baseline_acc = engine.model_evaluation(
            model=projector_model, 
            dataloader=test_dataloader,
            y_test=y_test, 
            device=device
        )
        print(f"✅ [Baseline Check] Visual-Only Accuracy: {baseline_acc * 100:.2f}%")
        projector_model.use_llm = temp_mode # 恢复

        # =========================================================
        # ✅ [Phase 2] Joint Training: Visual + LLM Fusion
        # =========================================================
        print(f"\n>>> [Phase 2] Joint Training Stage for {args.num_epochs} epochs...")
        

        if args.use_llm:
            if args.finetune_llm:
                print("🔓 [Status] Unfreezing LoRA parameters for fine-tuning...")
                for n, p in projector_model.named_parameters():
                    if "lora" in n:
                        p.requires_grad = True
            else:
                print("🔒 [Status] Keeping LoRA parameters FROZEN...")
                for n, p in projector_model.named_parameters():
                    if "lora" in n:
                        p.requires_grad = False
        

        param_groups = []
        if args.use_llm and args.finetune_llm:
            lora_params = [p for n, p in projector_model.named_parameters() if "lora" in n and p.requires_grad]
            if len(lora_params) > 0:
                param_groups.append({
                    "params": lora_params,
                    "lr": args.lora_lr,
                    "weight_decay": args.weight_decay
                })
                print(f"   -> Added {len(lora_params)} LoRA parameters (lr={args.lora_lr})")

        base_params = [p for n, p in projector_model.named_parameters() if "lora" not in n and p.requires_grad]
        if len(base_params) > 0:
            param_groups.append({
                "params": base_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay
            })
            print(f"   -> Added {len(base_params)} Base parameters (lr={args.lr})")
        
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

     
        total_trainable = sum(p.numel() for p in projector_model.parameters() if p.requires_grad)
        print(f"📊 Total Trainable Params: {total_trainable:,}")

        engine.train(
            model=projector_model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            weight=float(args.weight),
            criterion_instance=criterion_instance,
            criterion_cluster=criterion_cluster,
            epochs=args.num_epochs, 
            device=device,
            run_visual_only=False 
        )

        end_time = timer()
        print(f"Total training time: {(end_time - start_time):.3f} seconds")


        print(f"Evaluating model #{i+1}...")
        y_prob, y_pred, ind, acc = engine.model_evaluation(model=projector_model, 
                                                           dataloader=test_dataloader,
                                                           y_test=y_test, 
                                                           device=device)
        print(f"Accuracy of model: {acc * 100:.2f}%")
        
        d = {}
        for j, k in ind:
            d[j] = k
        for j in range(len(y_pred)):
            y_pred[j] = d[y_pred[j]]
        y_preds.append(y_pred)
        
        y_prob_hungarian = np.zeros_like(y_prob)
        for j in range(len(d.keys())):
            y_prob_hungarian[:, d[j]] = y_prob[:, j]
        y_probs.append(y_prob_hungarian)
        print("#" * 100)

   
        del projector_model
        del dna_llm
        del optimizer
        torch.cuda.empty_cache()

    # End of Loop
    
    # Hard voting
    y_preds = np.array(y_preds)
    mode, counts = stats.mode(y_preds, axis=0)
    if hasattr(mode, 'squeeze'): mode = mode.squeeze()
         
    w = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for i in range(y_test.shape[0]): w[y_test[i], mode[i]] += 1
    print(f"Accuracy of hard voting of {args.number_of_models} models: {100 * np.sum(np.diag(w) / np.sum(w)):.2f}")
    print("Confusion matrix:"); print(w)

    # Soft voting
    y_probs = np.array(y_probs)
    y_prob = []
    for i in range(y_probs.shape[1]):
        prob = np.zeros(y_probs.shape[2])
        for j in range(y_probs.shape[0]): prob += y_probs[j][i]
        prob /= (y_probs.shape[0])
        y_prob.append(prob)

    w = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for i in range(y_test.shape[0]): w[y_test[i], y_prob[i].argmax()] += 1
    print(f"Accuracy of soft voting of {args.number_of_models} models: {100 * np.sum(np.diag(w) / np.sum(w)):.2f}")
    print("Confusion matrix:"); print(w)