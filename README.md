# DEFINE-Emotion

Research adaptation of [AMAAI-Lab/DEFINE](https://github.com/AMAAI-Lab/define),
“Prototype-Anchored Exemplar Conditioning for Accent Control in Zero-Shot TTS”.
This project transfers the released implementation's conditioning method to
**emotion-controlled speech synthesis**. It does not translate the paper into
another language and is not an emotion recognition application.

Status: implementation adapted; not trained or empirically validated. No emotion
checkpoint or corpus is included. The upstream accent checkpoint is incompatible.
The linked repository was the available method source; no full paper was supplied.

## Method adaptation

The frozen F5-TTS backbone, LoRA, conditioning injection, and two-stage objective
are retained. Stage 1 learns an emotion prototype table. Stage 2 anchors an XLS-R
exemplar encoder to those prototypes with cosine alignment and prototype
classification, alongside emotion contrastive learning and adversarial speaker
removal. Inference uses either an emotion label or emotional example recordings,
separately from the voice reference.

Emotion classes come from your corpus. Use consistent labels such as `neutral`,
`happy`, `sad`, `angry`, `fearful`, `surprised`; do not silently merge datasets'
different emotion definitions. Emotions absent from training may be supplied as
exemplars, but transfer quality is an experimental question.

Cross-speaker exemplars reduce voice leakage. Their actual speaker IDs supervise
the adversarial head. Voice prompts prefer another neutral clip from the same
speaker when available; otherwise another same-speaker clip is used. Neutral is
a learned condition; guidance weight zero removes explicit emotion conditioning
and does not guarantee neutral speech. Guidance strength is not calibrated
emotion intensity. Pitch and timing can also carry speaker identity, so verify
that speaker removal does not erase useful emotional cues.

## Setup

Use Python 3.11 and a CUDA training machine. Install PyTorch and torchaudio for
that machine, then:

```sh
pip install -r requirements.txt
```

`config.yaml` locates F5-TTS, XLS-R, and Vocos weights. Missing base weights are
fetched by the existing Hugging Face loading code. Training requires substantial
GPU resources; this adaptation has no measured hardware or convergence results.

## Data

Supply a CSV with `wav_path,text,speaker,emotion,split`. Audio paths resolve
relative to the CSV. Transcripts must match the complete recording. Choose
speaker-disjoint `train`, `val`, and `test` partitions before preparation. Never
reuse the same recording under different file names across partitions.

```csv
wav_path,text,speaker,emotion,split
clips/s01_001.wav,It is a beautiful day.,s01,neutral,train
clips/s01_002.wav,I am so glad you came.,s01,happy,train
clips/s02_001.wav,The train arrives soon.,s02,neutral,train
clips/s02_002.wav,That is wonderful news.,s02,happy,train
```

The four rows above illustrate the minimum training structure; real research
needs far more speakers and clips plus separate validation/test speakers.
Preparation checks files, durations, duplicate paths, split overlap, and minimum
speaker coverage. Every training emotion needs two speakers; every training
speaker needs two clips. Keep emotion coverage balanced across speakers and
recording conditions. Held-out emotions belong only in val/test.

```sh
python prepare_emotions.py --csv /path/to/corpus.csv
python train.py --mode lookup --steps 20000 --out ckpt/lookup
python train.py --mode proto --steps 30000 \
  --init-from ckpt/lookup/emotion_last.pt --out ckpt/proto
```

Step counts are inherited starting points, not validated emotion settings.
`--resume` restores an interrupted run from its output directory; keep the same
manifest, architecture, mode, and schedule. Only train rows are used by training;
there is no automatic validation or early stopping yet.

## Generate speech after training

```sh
python infer.py --ckpt ckpt/proto/emotion_last.pt \
  --ref-wav voice.wav --ref-text "Transcript of the voice reference." \
  --text "I am so glad you came." --emotion happy --w 3 --out out/happy.wav

python infer.py --ckpt ckpt/proto/emotion_last.pt \
  --ref-wav voice.wav --ref-text "Transcript of the voice reference." \
  --text "I am so glad you came." --exemplars happy1.wav happy2.wav \
  --w 3 --out out/exemplar.wav
```

Sweep guidance weights, including zero, and listen for intelligibility and voice
drift. The suggested weight is illustrative. Use correctly transcribed, clean
voice references and emotional exemplars from other speakers.

## Required experimental validation

Compare unadapted F5-TTS, lookup conditioning, prototype conditioning, and an
exemplar-only ablation. Evaluate on speakers excluded from training. Keep probe
training recordings separate from synthesis exemplars and evaluation recordings.
Report independent emotion classifier macro-F1/recall, listener emotion judgments,
speaker similarity, WER, naturalness, and guidance sweeps. Include neutral voice
references with different target emotions to test whether conditioning overrides
the reference's prosody. Separate seen-emotion and held-out-emotion results.

`emotionbridge/metrics.py` retains reusable speaker similarity, WER, naturalness,
and SSL logistic-regression probe utilities. Its emotion probe must be fitted
and independently validated on your corpus before its scores are meaningful.
There is currently no complete automatic emotion evaluation runner. Upstream
Common Voice preparation, accent evaluation, and voice-conversion augmentation
scripts were removed because their corpus assumptions do not establish emotion
labels or preservation.

## Provenance and checks

Source downloaded from upstream `main` on 2026-10-27. Retained core implementation
is credited to AMAAI-Lab. No upstream license file was present in the downloaded
source; this adaptation does not grant rights beyond those held by its authors.

Local check: Python syntax compilation. End-to-end model loading, training,
checkpoint round trips, and generated audio still require dependency installation,
a corpus, and GPU validation.
