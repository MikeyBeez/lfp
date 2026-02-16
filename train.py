"""Training loop for all three regimes."""

import os
import time
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from config import ExperimentConfig, TrainingRegime, ModelConfig, MCR2Config
from model import SmallTransformer
from data import load_wikitext, build_dataloaders
from losses import CrossEntropyLoss, MCR2Loss, CombinedOrganismLoss
from losses.mcr2 import LayerWiseMCR2Loss
from evaluate import run_all_metrics


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step: int, cfg):
    """Cosine decay with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.learning_rate * 0.1
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: ExperimentConfig, resume_path: str = None):
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print(f"Loading {cfg.training.dataset}...")
    splits, tokenizer = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, cfg.training.batch_size)

    # Model
    model = SmallTransformer(cfg.model).to(device)
    print(f"Model parameters: {model.param_count():,}")

    # Loss functions
    regime = cfg.training.regime
    dual_beta = None  # will be set if using dual mode
    if regime == TrainingRegime.CE_ONLY:
        ce_loss_fn = CrossEntropyLoss()
    elif regime == TrainingRegime.MCR2_ONLY:
        mcr2_loss_fn = LayerWiseMCR2Loss(cfg.mcr2, layer_indices=cfg.training.mcr2_layers)
        # Still need CE for perplexity evaluation
        ce_eval_fn = CrossEntropyLoss()
    elif regime == TrainingRegime.ORGANISM:
        if cfg.training.dual_beta:
            # Dual/adversarial mode: beta is NOT a parameter — it's a plain scalar
            # updated via Lagrangian dual ascent based on representation quality.
            # Model minimizes: CE + beta * MCR2
            # Beta updates:    beta += lr * (MCR2 - target)
            #   -> beta rises when representations are poor (MCR2 > target)
            #   -> beta falls when representations are good (MCR2 < target)
            combined_loss_fn = CombinedOrganismLoss(
                alpha_ce=cfg.training.alpha_ce,
                mcr2_cfg=cfg.mcr2,
                layer_indices=cfg.training.mcr2_layers,
            )
            dual_beta = cfg.training.beta_mcr2  # initial value
            print(f"Dual beta mode: init={dual_beta}, target={cfg.training.mcr2_target}, "
                  f"lr={cfg.training.beta_lr}, bounds=[{cfg.training.beta_min}, {cfg.training.beta_max}]")
        else:
            combined_loss_fn = CombinedOrganismLoss(
                alpha_ce=cfg.training.alpha_ce,
                beta_mcr2=cfg.training.beta_mcr2,
                mcr2_cfg=cfg.mcr2,
                learn_beta=cfg.training.learn_beta,
                per_layer_beta=cfg.training.per_layer_beta,
                n_layers=cfg.model.n_layers,
                layer_indices=cfg.training.mcr2_layers,
                beta_min=cfg.training.beta_min,
            )
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Optimizer — include learned beta params if applicable (NOT for dual mode)
    param_groups = [{"params": model.parameters()}]
    if regime == TrainingRegime.ORGANISM and cfg.training.learn_beta and not cfg.training.dual_beta:
        beta_params = list(combined_loss_fn.parameters())
        if beta_params:
            param_groups.append({
                "params": beta_params,
                "weight_decay": 0.0,  # no decay on the beta parameters
            })
            beta_names = [n for n, _ in combined_loss_fn.named_parameters()]
            print(f"Learnable beta params: {beta_names}")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # Mixed precision
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Output directory
    run_dir = Path(cfg.training.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "regime": regime.value,
            "model": cfg.model.__dict__,
            "mcr2": cfg.mcr2.__dict__,
            "training": {k: v.value if hasattr(v, 'value') else v
                         for k, v in cfg.training.__dict__.items()},
            "seed": cfg.seed,
        }, f, indent=2)

    # Resume from checkpoint
    start_step = 0
    log_history = []
    mcr2_satisfied = False
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        if "dual_beta" in ckpt and dual_beta is not None:
            dual_beta = ckpt["dual_beta"]
        if "mcr2_satisfied" in ckpt:
            mcr2_satisfied = ckpt["mcr2_satisfied"]
        print(f"Resumed from {resume_path} at step {start_step}")

    # Training loop
    train_loader = loaders["train"]
    train_iter = iter(train_loader)

    model.train()
    step = start_step
    accum_loss = 0.0
    accum_ce = 0.0
    accum_mcr2 = 0.0
    t0 = time.time()

    while step < cfg.training.max_steps:
        optimizer.zero_grad()

        for micro_step in range(cfg.training.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            # Forward pass: collect layer representations (not attention weights) for MCR2
            needs_layer_reps = regime in (TrainingRegime.MCR2_ONLY, TrainingRegime.ORGANISM)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(x, collect_layer_reps=needs_layer_reps)

                if regime == TrainingRegime.CE_ONLY:
                    loss = ce_loss_fn(output.logits, y)
                    ce_val = loss.detach()
                    mcr2_val = torch.tensor(0.0)

                elif regime == TrainingRegime.MCR2_ONLY:
                    loss = mcr2_loss_fn(output.layer_representations, y)
                    mcr2_val = loss.detach()
                    with torch.no_grad():
                        ce_val = ce_eval_fn(output.logits, y)

                elif regime == TrainingRegime.ORGANISM:
                    # MCR2 scale: 0 during satisfaction, ramps 0->1 during warmup, 1 otherwise
                    if mcr2_satisfied:
                        mcr2_scale = 0.0
                    elif cfg.training.mcr2_warmup_steps > 0 and step < cfg.training.mcr2_warmup_steps:
                        mcr2_scale = step / cfg.training.mcr2_warmup_steps
                    else:
                        mcr2_scale = 1.0

                    loss_dict = combined_loss_fn(
                        output.logits, y, output.layer_representations,
                        mcr2_scale=mcr2_scale,
                        beta_override=dual_beta,  # None unless dual mode
                    )
                    loss = loss_dict["loss"]
                    ce_val = loss_dict["ce_loss"]
                    mcr2_val = loss_dict["mcr2_loss"]

                loss = loss / cfg.training.grad_accum_steps

            loss.backward()

        # Accumulate once per optimizer step (average across micro-steps)
        accum_loss += (loss.item() * cfg.training.grad_accum_steps)  # undo the division
        accum_ce += ce_val.item()    # last micro-step CE (representative)
        accum_mcr2 += mcr2_val.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

        # Update LR
        lr = get_lr(step, cfg.training)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()
        step += 1

        # Phase switching (curriculum or alternating)
        if regime in (TrainingRegime.MCR2_ONLY, TrainingRegime.ORGANISM):
            mcr2_obj = None
            if regime == TrainingRegime.ORGANISM:
                mcr2_obj = combined_loss_fn.mcr2_core
            elif regime == TrainingRegime.MCR2_ONLY:
                mcr2_obj = mcr2_loss_fn.mcr2

            if mcr2_obj is not None:
                # Alternating: cycle expansion/full every N steps
                if cfg.training.alternating_period > 0:
                    phase = (step // cfg.training.alternating_period) % 2
                    want_expansion = (phase == 0)
                    if want_expansion != mcr2_obj.expansion_only:
                        mcr2_obj.expansion_only = want_expansion
                        mcr2_obj.selective_expansion = False
                        mode = "expansion-only" if want_expansion else "full MCR2"
                        print(f"\n*** Alternating switch at step {step}: -> {mode} ***\n")
                # One-time curriculum switch
                elif (cfg.training.curriculum_switch_step > 0
                      and step == cfg.training.curriculum_switch_step):
                    mcr2_obj.expansion_only = False
                    mcr2_obj.selective_expansion = False
                    print(f"\n*** Curriculum switch at step {step}: expansion-only -> full MCR2 ***\n")

        # Dual ascent: update beta when MCR2 is active
        # Frozen during warmup and when MCR2 is satisfied (representations good enough).
        if dual_beta is not None and mcr2_scale >= 1.0:
            dual_beta += cfg.training.beta_lr * (mcr2_val.item() - cfg.training.mcr2_target)
            dual_beta = max(cfg.training.beta_min, min(cfg.training.beta_max, dual_beta))

        # Logging
        if step % cfg.training.log_interval == 0:
            avg_loss = accum_loss / cfg.training.log_interval
            avg_ce = accum_ce / cfg.training.log_interval
            avg_mcr2 = accum_mcr2 / cfg.training.log_interval
            elapsed = time.time() - t0
            ppl = math.exp(min(avg_ce, 20))  # cap to avoid overflow

            entry = {
                "step": step,
                "loss": avg_loss,
                "ce_loss": avg_ce,
                "mcr2_loss": avg_mcr2,
                "ppl": ppl,
                "lr": lr,
                "time": elapsed,
            }

            # Log beta value for organism regime
            beta_str = ""
            if regime == TrainingRegime.ORGANISM:
                if dual_beta is not None:
                    entry["beta"] = dual_beta
                    beta_str = f" | beta {dual_beta:.6f}"
                elif cfg.training.learn_beta:
                    betas = combined_loss_fn._get_betas()
                    if cfg.training.per_layer_beta:
                        entry["betas"] = betas.detach().cpu().tolist()
                        beta_str = f" | beta_mean {betas.mean().item():.4f}"
                    else:
                        entry["beta"] = betas.item()
                        beta_str = f" | beta {betas.item():.4f}"
                else:
                    entry["beta"] = cfg.training.beta_mcr2

            log_history.append(entry)

            print(
                f"step {step:6d} | loss {avg_loss:.4f} | CE {avg_ce:.4f} | "
                f"MCR2 {avg_mcr2:.4f} | ppl {ppl:.1f} | lr {lr:.2e}{beta_str} | "
                f"{elapsed:.1f}s"
            )

            accum_loss = 0.0
            accum_ce = 0.0
            accum_mcr2 = 0.0

        # Evaluation
        if step % cfg.training.eval_interval == 0:
            print(f"\n--- Evaluation at step {step} ---")
            model.eval()
            metrics = run_all_metrics(
                model, loaders["validation"], device, amp_dtype,
                max_batches=10,
            )
            model.train()

            # Log metrics + beta at eval time
            metrics["step"] = step
            if regime == TrainingRegime.ORGANISM:
                if dual_beta is not None:
                    metrics["beta"] = dual_beta
                elif cfg.training.learn_beta:
                    betas = combined_loss_fn._get_betas()
                    if cfg.training.per_layer_beta:
                        metrics["betas"] = betas.detach().cpu().tolist()
                    else:
                        metrics["beta"] = betas.item()
                else:
                    metrics["beta"] = cfg.training.beta_mcr2

            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics, default=str) + "\n")

            beta_eval_str = ""
            if "beta" in metrics:
                beta_eval_str = f"  beta: {metrics['beta']:.6f}\n"
            elif "betas" in metrics:
                beta_eval_str = f"  betas: {metrics['betas']}\n"

            print(f"  val_ppl: {metrics.get('val_ppl', 'N/A'):.1f}")
            print(f"  eff_rank (layer avg): {metrics.get('effective_rank_mean', 'N/A'):.1f}")
            print(f"  cosine_sim_mean: {metrics.get('cosine_sim_mean', 'N/A'):.4f}")
            print(f"  attn_entropy_mean: {metrics.get('attn_entropy_mean', 'N/A'):.3f}")
            print(f"  sink_ratio_mean: {metrics.get('sink_ratio_mean', 'N/A'):.4f}")
            print(f"  separation_ratio: {metrics.get('separation_ratio', 'N/A'):.2f}")
            if beta_eval_str:
                print(beta_eval_str, end="")

            # MCR2 satisfaction check: stop MCR2 when representations are organized enough
            # Re-engage if they degrade. Either threshold being met triggers shutoff.
            if regime == TrainingRegime.ORGANISM and (
                cfg.training.mcr2_rank_target > 0 or cfg.training.mcr2_sim_target > 0
            ):
                rank = metrics.get("effective_rank_mean", 0.0)
                sim = metrics.get("cosine_sim_mean", 1.0)
                rank_ok = cfg.training.mcr2_rank_target > 0 and rank >= cfg.training.mcr2_rank_target
                sim_ok = cfg.training.mcr2_sim_target > 0 and sim <= cfg.training.mcr2_sim_target

                if (rank_ok or sim_ok) and not mcr2_satisfied:
                    mcr2_satisfied = True
                    metrics["mcr2_satisfied"] = True
                    reasons = []
                    if rank_ok:
                        reasons.append(f"rank {rank:.1f} >= {cfg.training.mcr2_rank_target}")
                    if sim_ok:
                        reasons.append(f"sim {sim:.4f} <= {cfg.training.mcr2_sim_target}")
                    print(f"  MCR2 satisfied ({', '.join(reasons)}) — disabling MCR2")
                elif not rank_ok and not sim_ok and mcr2_satisfied:
                    mcr2_satisfied = False
                    metrics["mcr2_satisfied"] = False
                    print(f"  MCR2 re-engaged (rank {rank:.1f}, sim {sim:.4f})")

            print()

        # Save checkpoint
        if step % cfg.training.save_interval == 0:
            ckpt_path = run_dir / f"checkpoint_{step}.pt"
            ckpt_data = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "mcr2_satisfied": mcr2_satisfied,
            }
            if dual_beta is not None:
                ckpt_data["dual_beta"] = dual_beta
            torch.save(ckpt_data, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Save final
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, run_dir / "final.pt")

    with open(run_dir / "log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\nTraining complete. Results saved to {run_dir}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, default="ce_only",
                        choices=["ce_only", "mcr2_only", "organism"])
    parser.add_argument("--dataset", type=str, default="wikitext-2")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beta-mcr2", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    # Model size
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    # Learned scaling
    parser.add_argument("--learn-beta", action="store_true",
                        help="Make MCR2 beta a learnable parameter")
    parser.add_argument("--per-layer-beta", action="store_true",
                        help="Learn separate beta per layer (requires --learn-beta)")
    parser.add_argument("--mcr2-warmup-steps", type=int, default=0,
                        help="Steps before MCR2 reaches full strength (curriculum)")
    parser.add_argument("--beta-min", type=float, default=0.001,
                        help="Floor for beta (prevents MCR2 shutdown)")
    parser.add_argument("--beta-max", type=float, default=1.0,
                        help="Ceiling for beta")
    # Dual/adversarial beta
    parser.add_argument("--dual-beta", action="store_true",
                        help="Use Lagrangian dual ascent for beta (adversarial)")
    parser.add_argument("--beta-lr", type=float, default=1e-4,
                        help="Learning rate for dual beta update")
    parser.add_argument("--mcr2-target", type=float, default=-1.0,
                        help="Target MCR2 loss for dual ascent (beta rises when MCR2 > target)")
    # MCR2 satisfaction thresholds
    parser.add_argument("--mcr2-rank-target", type=float, default=0.0,
                        help="Stop MCR2 when effective_rank_mean >= this (0 = disabled)")
    parser.add_argument("--mcr2-sim-target", type=float, default=0.0,
                        help="Stop MCR2 when cosine_sim_mean <= this (0 = disabled)")
    # Layer-selective MCR2
    parser.add_argument("--mcr2-layers", type=str, default=None,
                        help="Comma-separated layer indices for MCR2 (e.g., '0,1,2')")
    parser.add_argument("--expansion-only", action="store_true",
                        help="Use only the expansion term of MCR2 (no compression)")
    parser.add_argument("--selective-expansion", action="store_true",
                        help="Expand class centroids only (frequency-neutral, no compression)")
    parser.add_argument("--compression-only", action="store_true",
                        help="Use only the compression term of MCR2 (no expansion)")
    parser.add_argument("--curriculum-switch-step", type=int, default=0,
                        help="Step to switch from expansion-only to full MCR2 (0 = disabled)")
    parser.add_argument("--alternating-period", type=int, default=0,
                        help="Alternate expansion/full MCR2 every N steps (0 = disabled)")
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Root directory for run outputs and checkpoints")
    parser.add_argument("--save-interval", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from checkpoint. No value = auto-detect latest; or pass a path")
    args = parser.parse_args()

    mcr2_layers = None
    if args.mcr2_layers:
        mcr2_layers = [int(x) for x in args.mcr2_layers.split(",")]

    run_name = args.run_name or args.regime

    cfg = ExperimentConfig(
        model=ModelConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
        ),
        mcr2=MCR2Config(
            expansion_only=args.expansion_only,
            selective_expansion=args.selective_expansion,
            compression_only=args.compression_only,
        ),
        training=__import__("config").TrainingConfig(
            regime=TrainingRegime(args.regime),
            dataset=args.dataset,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            beta_mcr2=args.beta_mcr2,
            learning_rate=args.lr,
            learn_beta=args.learn_beta,
            per_layer_beta=args.per_layer_beta,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            mcr2_warmup_steps=args.mcr2_warmup_steps,
            dual_beta=args.dual_beta,
            beta_lr=args.beta_lr,
            mcr2_target=args.mcr2_target,
            mcr2_rank_target=args.mcr2_rank_target,
            mcr2_sim_target=args.mcr2_sim_target,
            mcr2_layers=mcr2_layers,
            curriculum_switch_step=args.curriculum_switch_step,
            alternating_period=args.alternating_period,
            output_dir=args.output_dir,
            save_interval=args.save_interval,
        ),
        seed=args.seed,
        run_name=run_name,
    )

    # Resolve --resume: auto-detect latest checkpoint if no path given
    resume_path = None
    if args.resume == "auto":
        run_dir = Path(args.output_dir) / run_name
        ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            resume_path = str(ckpts[-1])
            print(f"Auto-detected checkpoint: {resume_path}")
        else:
            print("No checkpoints found, starting fresh")
    elif args.resume is not None:
        resume_path = args.resume

    train(cfg, resume_path=resume_path)
