"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
"""稳健训练配方：warmup cosine + EMA，适配小模型短训练"""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


# ==================== EMA 指数移动平均 ====================
class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')

    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)

    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 优化器超参（稳健版） ====================
    PEAK_LR = 0.0025
    WEIGHT_DECAY = 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)

    # ==================== EMA 初始化 ====================
    ema = EMA(model, decay=0.995)

    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.

    # ==================== 学习率调度 ====================
    WARMUP_STEPS = 200
    MIN_LR_RATIO = 0.05

    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]

        # warmup + cosine 衰减
        if step < WARMUP_STEPS:
            warmup_factor = (step + 1) / WARMUP_STEPS
        else:
            warmup_factor = 1.0
        progress = step / max(1, args.steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        decay_factor = MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine_factor
        learning_rate = PEAK_LR * warmup_factor * decay_factor

        for group in optimizer.param_groups:
            group['lr'] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 每步更新 EMA
        ema.step()

        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {
                'step': step + 1,
                'loss': loss.item(),
                'lr': learning_rate,
                'seconds': time.perf_counter() - started - intermediate_validation_seconds
            }
            history.append(row)
            print(json.dumps(row), flush=True)

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            ema.apply()
            intermediate = score(model, *data['validation'], device, 'fp32')
            ema.restore()

            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step': step + 1, **intermediate})
            print(json.dumps({'validation': validation_history[-1]}), flush=True)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    train_seconds = time.perf_counter() - started - intermediate_validation_seconds

    # 最终验证与保存使用 EMA 权重
    ema.apply()

    validation = score(model, *data['validation'], device, 'fp32')
    validation.pop('window_nll_nats')

    checkpoint = args.run_dir / 'checkpoint.pt'
    torch.save({
        'protocol': PROTOCOL,
        'implementation': args.implementation,
        'config': config,
        'model': model.cpu().state_dict(),
        'seed': args.seed,
        'train_tokens': args.steps * args.batch_size * 256
    }, checkpoint)

    ema.restore()

    result = {
        'protocol': PROTOCOL,
        'implementation': args.implementation,
        'config': config,
        'seed': args.seed,
        'parameters': sum(p.numel() for p in model.parameters()),
        'precision': precision,
        'train_tokens': args.steps * args.batch_size * 256,
        'preparation_seconds': preparation_seconds,
        'train_seconds': train_seconds,
        'validation': validation,
        'history': history,
        'validation_history': validation_history,
        'intermediate_validation_seconds': intermediate_validation_seconds,
        'process_seconds': time.perf_counter() - total_started,
        'torch_version': str(torch.__version__),
        'threads': args.threads,
        'checkpoint_sha256': sha(checkpoint),
        'implementation_sha256': implementation_sha,
        **device_metrics(device)
    }
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result | {'history': []}, indent=2), flush=True)


if __name__ == '__main__':
    main()
