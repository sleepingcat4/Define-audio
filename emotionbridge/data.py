import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import SR, load_audio, resample, slice_seconds, speech_crops
# Approximate mel frames per row (prompt + target) for length bucketing.
def frame_lens(rows, hop=256, sr=SR, gap_s=0.15):
    d = rows["duration_s"].astype(float)
    extra = rows["max_prompt_s"].fillna(0.0) + rows["prompt_split_t"].fillna(0.0)
    d = d + rows["split_t"].isna() * (extra + gap_s)
    return ((d * sr) / hop).astype(int).to_numpy()


class PairDataset(Dataset):
    # Returns prompt+target mel, text, exemplar crops from another clip of the same emotion, and labels.
    def __init__(self, rows, mel_fn, exemplar_k=2, exemplar_win_s=4.0, seed=0):
        self.rows = rows
        self.mel_fn = mel_fn
        self.k = exemplar_k
        self.win = exemplar_win_s
        self.rng = np.random.default_rng(seed)
        real = rows[rows["source"] != "aug"]
        pools = {g: idx.to_numpy() for g, idx in real.groupby("exemplar_group").groups.items()}
        for g, idx in rows.groupby("exemplar_group").groups.items():
            pools.setdefault(g, idx.to_numpy())
        self.by_group = pools
        self.by_speaker = {g: idx.to_numpy() for g, idx in rows.groupby("voice_id").groups.items()}
        self.gap = np.zeros(int(0.15 * SR), dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def pick_exemplar(self, i):
        r = self.rows.iloc[i]
        pool = self.by_group[r.exemplar_group]
        pool = pool[self.rows.iloc[pool]["voice_id"].to_numpy() != r.voice_id]
        if not len(pool):
            raise ValueError(f"Emotion {r.exemplar_group} needs at least two training speakers")
        return self.rows.iloc[int(self.rng.choice(pool))]

    def pick_prompt(self, i):
        r = self.rows.iloc[i]
        pool = self.by_speaker[r.voice_id]
        cand = self.rows.iloc[pool[pool != i]]
        neutral = cand[cand["exemplar_group"] == "neutral"]
        if len(neutral):
            cand = neutral
        short = cand[cand["duration_s"] <= r.max_prompt_s]
        cand = short if len(short) else cand.nsmallest(1, "duration_s")
        return cand.iloc[int(self.rng.integers(len(cand)))]

    def load_pair(self, i):
        r = self.rows.iloc[i]
        y = load_audio(r.wav_path, SR)
        if pd.notna(r.split_t):
            return slice_seconds(y, SR, 0, r.split_t), slice_seconds(y, SR, r.split_t), r.prompt_text
        if pd.notna(r.prompt_wav_path):
            p = load_audio(r.prompt_wav_path, SR)
            return np.concatenate([slice_seconds(p, SR, 0, r.prompt_split_t), self.gap]), y, r.prompt_text
        p = self.pick_prompt(i)
        return np.concatenate([load_audio(p.wav_path, SR), self.gap]), y, p.body_text

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        prompt, target, prompt_text = self.load_pair(i)
        mel = self.mel_fn(torch.from_numpy(np.concatenate([prompt, target]))[None])[0]
        ex_row = self.pick_exemplar(i)
        ex = resample(load_audio(ex_row.wav_path, SR), SR, 16000)
        crops = speech_crops(ex, 16000, win_s=self.win, k=self.k, rng=self.rng)
        return dict(mel=mel, n_prompt=int(round(len(prompt) / self.mel_fn.hop_length)),
                    text=f"{prompt_text} {r.body_text}".strip(), exemplar=torch.from_numpy(crops),
                    emotion_id=int(r.emotion_id), speaker_id=int(ex_row.speaker_id))


def collate(batch):
    lens = torch.tensor([b["mel"].shape[-1] for b in batch])
    T = int(lens.max())
    mel = torch.stack([torch.nn.functional.pad(b["mel"], (0, T - b["mel"].shape[-1])) for b in batch]).permute(0, 2, 1)
    span = torch.zeros(len(batch), T, dtype=torch.bool)
    for i, b in enumerate(batch):
        span[i, b["n_prompt"]:lens[i]] = True
    return dict(mel=mel, lens=lens, span_mask=span, text=[b["text"] for b in batch],
                exemplar=torch.stack([b["exemplar"] for b in batch]),
                emotion_id=torch.tensor([b["emotion_id"] for b in batch]),
                speaker_id=torch.tensor([b["speaker_id"] for b in batch]))


class FrameBatchSampler:
    # Length-sorted batches capped at max_frames padded mel frames.
    def __init__(self, frame_lens, max_frames, seed=0):
        self.rng = np.random.default_rng(seed)
        batches, cur, cur_max = [], [], 0
        for i in np.argsort(frame_lens):
            n = int(frame_lens[i])
            if cur and max(cur_max, n) * (len(cur) + 1) > max_frames:
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(int(i))
            cur_max = max(cur_max, n)
        if cur:
            batches.append(cur)
        self.batches = batches

    def __iter__(self):
        for i in self.rng.permutation(len(self.batches)):
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)
