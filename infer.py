import argparse
from pathlib import Path
import torch

from emotionbridge.audio import SR, load_audio, write_wav
from emotionbridge.config import Config
from emotionbridge.model.build import build_feat_fn, load_emotionbridge, load_vocoder
from emotionbridge.synth import embed_exemplars, lookup_emotion, synthesize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref-wav", required=True, help="voice prompt (a few seconds of speech)")
    ap.add_argument("--ref-text", required=True, help="transcript of the voice prompt")
    ap.add_argument("--text", required=True, help="text to synthesise")
    ap.add_argument("--emotion", default=None, help="emotion label (lookup mode)")
    ap.add_argument("--exemplars", nargs="+", default=None, help="emotion exemplar clips (exemplar mode)")
    ap.add_argument("--w", type=float, default=9.0, help="emotion guidance weight (0 = emotion off)")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if (a.emotion is None) == (a.exemplars is None):
        ap.error("give exactly one of --emotion or --exemplars")
    C = Config(a.config)
    cfm, enc, mel_fn, table, classes = load_emotionbridge(a.ckpt, C.f5_ckpt(), a.device)
    voc = load_vocoder(a.device, C.vocoder_dir(), C.hf_cache)
    if a.emotion is not None:
        emotion = lookup_emotion(table, classes, a.emotion)
    else:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        layer = ck.get("model_config", {}).get("ssl_layer", 15)
        feat_fn = build_feat_fn(C.ssl_model(), a.device, layer=layer, cache_dir=C.hf_cache)
        emotion = embed_exemplars(a.exemplars, feat_fn, enc, a.device, seed=a.seed)
    wav = synthesize(cfm, mel_fn, voc, load_audio(a.ref_wav, SR), a.ref_text, a.text, emotion, a.w, a.device,
                     nfe=a.nfe, cfg=a.cfg, speed=a.speed, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_wav(a.out, wav, SR)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
