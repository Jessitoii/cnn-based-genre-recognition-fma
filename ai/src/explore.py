import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/fma_metadata/tracks.csv", index_col=0, header=[0, 1])
subset = df[df[("set", "subset")] == "small"]
genre_col = ("track", "genre_top")

genres = subset[genre_col].unique()
fig, axes = plt.subplots(2, 4, figsize=(16, 6))

for ax, genre in zip(axes.flatten(), genres):
    sample = subset[subset[genre_col] == genre].index[0]
    tid = f"{sample:06d}"
    path = f"data/fma_small/{tid[:3]}/{tid}.mp3"

    y, sr = librosa.load(path, sr=22050, duration=30, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    librosa.display.specshow(
        mel_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax
    )
    ax.set_title(genre)
    ax.set_xlabel("")

plt.suptitle("Mel-Spectrogram per Genre (1 sample each)", y=1.02)
plt.tight_layout()
plt.savefig("mel_samples.png", dpi=100, bbox_inches="tight")
print("Saved: mel_samples.png")
