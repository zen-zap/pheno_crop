"""
PhenoCrop — Computational Complexity Profiler
==============================================
Computes parameter counts and inference latency for all DL models.

Run:   python scripts/compute_complexity.py
Output: prints markdown table + saves complexity_results.json
"""

import math, time, json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ───────────────────────────────────────────────────────
MAX_T       = 40
S1_FEATS    = 3
S2_FEATS    = 8
FUSED_FEATS = S1_FEATS + S2_FEATS + 1   # 12 (baselines use cloud_pct)
NUM_CLASSES = 5
BATCH       = 1
DEVICE      = "cpu"
N_WARMUP    = 20
N_RUNS      = 100

# ═══════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

# ── BiLSTM ────────────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, in_f=12, hid=128, nl=2, nc=5, drop=0.3):
        super().__init__()
        self.rnn = nn.LSTM(in_f, hid, nl, batch_first=True,
                           dropout=drop if nl > 1 else 0., bidirectional=True)
        d = hid * 2
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(d, d//2), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(d//2, nc))

    def forward(self, x, mask=None):
        out, _ = self.rnn(x)
        out = self.norm(out).mean(1)
        return self.head(self.drop(out))


# ── TempCNN ───────────────────────────────────────────────────────
class TempCNN(nn.Module):
    def __init__(self, in_f=12, nc=5, ch=64, drop=0.3):
        super().__init__()
        def cb(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k//2),
                                 nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(drop))
        self.enc = nn.Sequential(cb(in_f, ch, 5), cb(ch, ch, 5), cb(ch, ch*2, 3))
        self.head = nn.Sequential(nn.Linear(ch*2, ch), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(ch, nc))

    def forward(self, x, mask=None):
        return self.head(self.enc(x.transpose(1, 2)).mean(-1))


# ── TCN ───────────────────────────────────────────────────────────
class _TCNBlock(nn.Module):
    def __init__(self, ic, oc, ks, dil, drop=0.2):
        super().__init__()
        pad = (ks - 1) * dil
        self.c1 = nn.Conv1d(ic, oc, ks, padding=pad, dilation=dil)
        self.c2 = nn.Conv1d(oc, oc, ks, padding=pad, dilation=dil)
        self.bn1, self.bn2 = nn.BatchNorm1d(oc), nn.BatchNorm1d(oc)
        self.drop = nn.Dropout(drop)
        self.ds = nn.Conv1d(ic, oc, 1) if ic != oc else None
        self.pad = pad

    def forward(self, x):
        res = x if self.ds is None else self.ds(x)
        o = F.relu(self.bn1(self.c1(x)[..., :-self.pad] if self.pad else self.c1(x)))
        o = F.relu(self.bn2(self.c2(self.drop(o))[..., :-self.pad] if self.pad else self.c2(self.drop(o))))
        return F.relu(self.drop(o) + res)


class TCN(nn.Module):
    def __init__(self, in_f=12, nc=5, chs=None, ks=3, drop=0.2):
        super().__init__()
        chs = chs or [64, 64, 128, 128]
        layers, ic = [], in_f
        for i, oc in enumerate(chs):
            layers.append(_TCNBlock(ic, oc, ks, 2**i, drop)); ic = oc
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(ic, ic//2), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(ic//2, nc))

    def forward(self, x, mask=None):
        return self.head(self.net(x.transpose(1, 2)).mean(-1))


# ── Vanilla Transformer ──────────────────────────────────────────
def _sinusoid(n, d):
    pos = torch.arange(n).float().unsqueeze(1)
    dim = torch.arange(d).float().unsqueeze(0)
    a = pos / torch.pow(10000, 2*(dim//2)/d)
    a[:, 0::2] = torch.sin(a[:, 0::2])
    a[:, 1::2] = torch.cos(a[:, 1::2])
    return a

def _month_tab(d):
    ang = torch.arange(12).float() / 12 * 2 * math.pi
    return torch.cat([torch.sin(ang).unsqueeze(1).expand(-1, d//2),
                      torch.cos(ang).unsqueeze(1).expand(-1, d//2)], -1)


class VanillaTransformer(nn.Module):
    def __init__(self, in_f=12, dm=128, nh=8, nl=3, nc=5, dff=256, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(in_f, dm)
        self.register_buffer("pe", _sinusoid(128, dm))
        enc = nn.TransformerEncoderLayer(dm, nh, dff, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, nl, nn.LayerNorm(dm))
        self.head = nn.Sequential(nn.Linear(dm, dm//2), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(dm//2, nc))

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        x = self.proj(x) + self.pe[:T]
        x = self.enc(x).mean(1)
        return self.head(x)


# ── PhenoCrop-Presto ─────────────────────────────────────────────
class _TFBlock(nn.Module):
    def __init__(self, d, nh, mr=2.0, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, int(d*mr)), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(int(d*mr), d), nn.Dropout(drop))

    def forward(self, x, kpm=None):
        h, _ = self.attn(self.n1(x), self.n1(x), self.n1(x),
                         key_padding_mask=kpm, need_weights=False)
        x = x + h
        return x + self.mlp(self.n2(x))


class PhenoCropPresto(nn.Module):
    def __init__(self, dm=128, depth=4, nh=8, mr=2.0, drop=0.1, nc=5):
        super().__init__()
        self.dm = dm
        self.s1p = nn.Linear(S1_FEATS, dm)
        self.s2p = nn.Linear(S2_FEATS, dm)
        self.ch_emb = nn.Embedding(2, dm//4)
        self.register_buffer("pos_tab", _sinusoid(121, dm//2))
        self.register_buffer("month_tab", _month_tab(dm//4))
        self.blocks = nn.ModuleList([_TFBlock(dm, nh, mr, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dm)
        self.head = nn.Sequential(nn.Linear(dm, dm//2), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(dm//2, nc))

    def forward(self, s1, s2, days, s1m, s2m):
        B, T, _ = s1.shape
        dev = s1.device
        pe = self.pos_tab[days.clamp(0, 120)]
        me = self.month_tab[(1 - (days//30).clamp(0, 11)) % 12]
        toks, masks = [], []
        for gi, (p, f, m) in enumerate([(self.s1p, s1, s1m), (self.s2p, s2, s2m)]):
            ch = self.ch_emb(torch.tensor(gi, device=dev)).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            toks.append(p(f) + torch.cat([pe, ch, me], -1))
            masks.append(m)
        x = torch.cat(toks, 1)
        kpm = torch.cat(masks, 1)
        for blk in self.blocks:
            x = blk(x, kpm)
        x = self.norm(x)
        v = (~kpm).unsqueeze(-1).float()
        return self.head((x * v).sum(1) / v.sum(1).clamp(min=1))


# ── PhenoCrop-BiMamba-Transformer ────────────────────────────────
class _MambaCore(nn.Module):
    def __init__(self, dm, ds=16, exp=2, dc=4):
        super().__init__()
        di = dm * exp
        self.ds = ds
        self.di = di
        self.inp = nn.Linear(dm, di*2, bias=False)
        self.conv = nn.Conv1d(di, di, dc, padding=dc-1, groups=di)
        self.xp = nn.Linear(di, ds*2 + di, bias=False)
        self.dtp = nn.Linear(di, di)
        A = torch.arange(1, ds+1, dtype=torch.float32).unsqueeze(0).expand(di, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out = nn.Linear(di, dm, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.inp(x)
        x_, z = xz.chunk(2, -1)
        x_ = self.conv(x_.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_ = F.silu(x_)
        tmp = self.xp(x_)
        Bs, Cs = tmp[..., :self.ds], tmp[..., self.ds:2*self.ds]
        dt = F.softplus(self.dtp(tmp[..., 2*self.ds:]))
        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * Bs.unsqueeze(2)
        h = torch.zeros(B, self.di, self.ds, device=x.device)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_[:, t].unsqueeze(-1)
            ys.append((h * Cs[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1)
        return self.out((y + x_ * self.D) * F.silu(z))


class _BiMamba(nn.Module):
    def __init__(self, dm, ds=16, exp=2, dc=4, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dm)
        self.fwd = _MambaCore(dm, ds, exp, dc)
        self.bwd = _MambaCore(dm, ds, exp, dc)
        self.merge = nn.Linear(dm*2, dm, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.norm(x)
        return x + self.drop(self.merge(torch.cat([self.fwd(h), self.bwd(h.flip(1)).flip(1)], -1)))


class PhenoCropBiMamba(nn.Module):
    def __init__(self, dm=128, md=2, td=4, nh=8, ds=16, mr=2.0, drop=0.1, nc=5):
        super().__init__()
        self.s1p = nn.Linear(S1_FEATS, dm)
        self.s2p = nn.Linear(S2_FEATS, dm)
        self.ch_emb = nn.Embedding(2, dm//4)
        self.register_buffer("pos_tab", _sinusoid(131, dm//2))
        self.register_buffer("month_tab", _month_tab(dm//4))
        self.s1m = nn.Sequential(*[_BiMamba(dm, ds, drop=drop) for _ in range(md)])
        self.s2m = nn.Sequential(*[_BiMamba(dm, ds, drop=drop) for _ in range(md)])
        self.tf = nn.ModuleList([_TFBlock(dm, nh, mr, drop) for _ in range(td)])
        self.norm = nn.LayerNorm(dm)
        self.head = nn.Sequential(nn.Linear(dm, dm//2), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(dm//2, nc))

    def forward(self, s1, s2, days, s1m, s2m):
        B, T, _ = s1.shape
        dev = s1.device
        pe = self.pos_tab[days.clamp(0, 130)]
        me = self.month_tab[(1 - (days//30).clamp(0, 11)) % 12]
        toks, masks = [], []
        for gi, (proj, mamba, f, m) in enumerate([
            (self.s1p, self.s1m, s1, s1m), (self.s2p, self.s2m, s2, s2m)]):
            ch = self.ch_emb(torch.tensor(gi, device=dev)).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            t = proj(f) + torch.cat([pe, ch, me], -1)
            toks.append(mamba(t))
            masks.append(m)
        x = torch.cat(toks, 1)
        kpm = torch.cat(masks, 1)
        for blk in self.tf:
            x = blk(x, kpm)
        x = self.norm(x)
        v = (~kpm).unsqueeze(-1).float()
        return self.head((x * v).sum(1) / v.sum(1).clamp(min=1))


# ═══════════════════════════════════════════════════════════════════
#  PROFILING
# ═══════════════════════════════════════════════════════════════════

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def fmt(n):
    return f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.1f}K" if n >= 1e3 else str(int(n)))

def latency(model, inputs):
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(*inputs)
        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            model(*inputs)
        return (time.perf_counter() - t0) / N_RUNS * 1000


def main():
    dummy_fused = (torch.randn(BATCH, MAX_T, FUSED_FEATS),)
    dummy_sep = (torch.randn(BATCH, MAX_T, S1_FEATS),
                 torch.randn(BATCH, MAX_T, S2_FEATS),
                 torch.randint(0, 90, (BATCH, MAX_T)),
                 torch.zeros(BATCH, MAX_T, dtype=torch.bool),
                 torch.zeros(BATCH, MAX_T, dtype=torch.bool))

    models = [
        ("BiLSTM",              BiLSTM(FUSED_FEATS),         dummy_fused),
        ("TempCNN",             TempCNN(FUSED_FEATS),        dummy_fused),
        ("TCN",                 TCN(FUSED_FEATS),            dummy_fused),
        ("Vanilla Transformer", VanillaTransformer(FUSED_FEATS), dummy_fused),
        ("PhenoCrop-Presto",    PhenoCropPresto(),           dummy_sep),
        ("PhenoCrop-BiMamba",   PhenoCropBiMamba(),          dummy_sep),
    ]

    print("=" * 65)
    print("  PhenoCrop — Computational Complexity Profile")
    print(f"  Device: {DEVICE} | Seq length: {MAX_T} | Batch: {BATCH}")
    print("=" * 65)

    results = []
    for name, model, inp in models:
        model.eval()
        # verify forward pass
        with torch.no_grad():
            out = model(*inp)
        assert out.shape == (BATCH, NUM_CLASSES), f"{name}: bad shape {out.shape}"

        p = count_params(model)
        ms = latency(model, inp)
        results.append({"model": name, "params": p, "params_fmt": fmt(p), "latency_ms": round(ms, 2)})
        print(f"  ✓ {name:<28} {fmt(p):>10} params   {ms:>8.2f} ms")

    # Markdown table
    print("\n\n## Computational Complexity\n")
    print("| Model | Parameters | Inference (ms) | Complexity |")
    print("|---|---|---|---|")
    for r in results:
        cplx = "O(T)" if "BiMamba" in r["model"] else ("O(T²)" if "Presto" in r["model"] or "Transformer" in r["model"] else "O(T)")
        print(f"| {r['model']} | {r['params_fmt']} | {r['latency_ms']} | {cplx} |")

    with open("scripts/complexity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved to scripts/complexity_results.json")


if __name__ == "__main__":
    main()
