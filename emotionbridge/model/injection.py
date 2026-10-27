import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionTimeEmbedding(nn.Module):
    # Wraps the DiT timestep embedding: t + gain * rms(t) * LN(W a); identity when emotion is None.
    def __init__(self, base, emotion_dim, dim, gain=0.3):
        super().__init__()
        self.base = base
        self.dim = dim
        self.proj = nn.Linear(emotion_dim, dim, bias=False)
        self.gain = nn.Parameter(torch.tensor(float(gain)), requires_grad=False)
        self.register_buffer("null", torch.zeros(emotion_dim))
        self.emotion = None

    def forward(self, timestep):
        t = self.base(timestep)
        a = self.emotion
        if a is None:
            return t
        if a.shape[0] != t.shape[0]:
            a = a.expand(t.shape[0], -1)
        h = F.layer_norm(self.proj(a.to(t.dtype)), (self.dim,))
        rms = t.detach().float().pow(2).mean(-1, keepdim=True).sqrt().to(t.dtype)
        return t + self.gain.to(t.dtype) * h * rms


class EmotionTextInject(nn.Module):
    # Wraps the DiT text embedding with the same emotion injection, broadcast over tokens.
    def __init__(self, base, holder, emotion_dim, text_dim, gain=0.3):
        super().__init__()
        self.base = base
        object.__setattr__(self, "holder", holder)
        self.text_dim = text_dim
        self.proj = nn.Linear(emotion_dim, text_dim, bias=False)
        self.gain = nn.Parameter(torch.tensor(float(gain)), requires_grad=False)

    def forward(self, text, seq_len, drop_text=False):
        out = self.base(text, seq_len=seq_len, drop_text=drop_text)
        a = self.holder.emotion
        if a is None:
            return out
        if a.shape[0] != out.shape[0]:
            a = a.expand(out.shape[0], -1)
        h = F.layer_norm(self.proj(a.to(out.dtype)), (self.text_dim,))
        rms = out.detach().float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt().to(out.dtype)
        return out + self.gain.to(out.dtype) * h[:, None, :] * rms


# Install both injection points on a DiT; returns the time-embedding module that holds the emotion.
def attach_emotion(dit, emotion_dim=128, time_gain=0.3, text_gain=0.5):
    dit.time_embed = EmotionTimeEmbedding(dit.time_embed, emotion_dim, dit.dim, gain=time_gain)
    dit.text_embed = EmotionTextInject(dit.text_embed, dit.time_embed, emotion_dim,
                                      dit.text_embed.text_embed.embedding_dim, gain=text_gain)
    return dit.time_embed


def injection_parameters(dit):
    return [dit.time_embed.proj.weight, dit.text_embed.proj.weight]
