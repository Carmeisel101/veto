"""Supervised fine-tuning of MiniLM for sequence classification (plain PyTorch loop)."""

from __future__ import annotations

import copy

import numpy as np

from veto.training.config import TrainConfig, set_seed


def predict_proba(model, tokenizer, texts: list[str], device: str, cfg: TrainConfig,
                  batch_size: int = 32) -> np.ndarray:
    """Probability of the positive class."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i : i + batch_size], truncation=True,
                            max_length=cfg.max_length, padding=True, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def fit_finetune(train_texts, train_y, cfg: TrainConfig, device: str,
                 val_texts=None, val_y=None):
    """Fine-tune and return (model, tokenizer, history).

    With validation data, keeps the best epoch by validation PR-AUC (ties -> earlier epoch)
    and stops after ``early_stopping_patience`` epochs without improvement. Without
    validation data, trains for exactly ``cfg.epochs``.
    """
    import torch
    from sklearn.metrics import average_precision_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, num_labels=2)
    model.to(device)
    y = np.asarray(train_y, dtype=int)
    n_pos = max(int(y.sum()), 1)
    pos_weight = min(cfg.max_pos_weight, (len(y) - n_pos) / n_pos)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(y) / cfg.batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    gen = torch.Generator().manual_seed(cfg.seed)

    history, best_state, best_score, stale = [], None, -1.0, 0
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(y), generator=gen).numpy()
        total = 0.0
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            enc = tokenizer([train_texts[j] for j in idx], truncation=True,
                            max_length=cfg.max_length, padding=True, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            loss = loss_fn(logits, torch.tensor(y[idx], device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += float(loss) * len(idx)
        rec = {"epoch": epoch + 1, "train_loss": round(total / len(y), 4)}
        if val_texts is not None and len(set(val_y)) > 1:
            p = predict_proba(model, tokenizer, list(val_texts), device, cfg)
            score = float(average_precision_score(val_y, p))
            rec["val_pr_auc"] = round(score, 4)
            if score > best_score:
                best_score, stale = score, 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                rec["best"] = True
            else:
                stale += 1
        history.append(rec)
        if val_texts is not None and stale >= cfg.early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, history
