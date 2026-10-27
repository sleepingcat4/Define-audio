"""Validate an emotion corpus CSV and create a speaker-disjoint training manifest."""
import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf

from emotionbridge.config import Config


def prepare(csv):
    csv = Path(csv).resolve()
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    required = {'wav_path', 'text', 'speaker', 'emotion', 'split'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    for c in required:
        df[c] = df[c].str.strip()
        if df[c].eq('').any():
            raise ValueError(f'Empty {c}')
    if not set(df.split) <= {'train', 'val', 'test'}:
        raise ValueError('split must be train, val, or test')
    if df.groupby('speaker')['split'].nunique().gt(1).any():
        raise ValueError('A speaker appears in multiple splits')
    paths, durations = [], []
    for raw in df.wav_path:
        path = Path(raw).expanduser()
        path = (csv.parent / path).resolve() if not path.is_absolute() else path.resolve()
        info = sf.info(str(path))
        if info.duration <= 0:
            raise ValueError(f'Empty audio: {path}')
        paths.append(str(path))
        durations.append(info.duration)
    df['wav_path'], df['duration_s'] = paths, durations
    if df.wav_path.duplicated().any():
        raise ValueError('Duplicate audio paths')
    train = df[df.split == 'train']
    if train.emotion.nunique() < 2:
        raise ValueError('At least two training emotions are required')
    if train.groupby('emotion').speaker.nunique().lt(2).any():
        raise ValueError('Each training emotion needs at least two speakers')
    if train.groupby('speaker').size().lt(2).any():
        raise ValueError('Each training speaker needs at least two clips for voice prompting')
    classes = sorted(train.emotion.unique())
    speakers = sorted(train.speaker.unique())
    df['emotion_id'] = df.emotion.map({v: i for i, v in enumerate(classes)}).fillna(-1).astype(int)
    df['speaker_id'] = df.speaker.map({v: i for i, v in enumerate(speakers)}).fillna(-1).astype(int)
    df['voice_id'], df['exemplar_group'] = df.speaker, df.emotion
    df['body_text'], df['prompt_text'] = df.text, ''
    df['source'], df['max_prompt_s'] = 'emotion', 7.0
    df['split_t'], df['prompt_split_t'], df['prompt_wav_path'] = float('nan'), float('nan'), None
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--config', default='config.yaml')
    a = ap.parse_args()
    rows = prepare(a.csv)
    out = Config(a.config).manifest / 'emotions.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out, index=False)
    print(rows.groupby(['split', 'emotion']).size().to_string())
    print('Wrote', out)


if __name__ == '__main__':
    main()
