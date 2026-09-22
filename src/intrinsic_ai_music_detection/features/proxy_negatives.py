"""Synthesise generator-agnostic negatives from real music alone.

The idea, and why it is the strongest remaining lever
-----------------------------------------------------
Every text-to-music system in both benchmarks emits audio through a **neural
decoder**: MusicGen decodes EnCodec tokens, MusicLDM and AudioLDM2 invert a
mel-spectrogram with HiFi-GAN, Stable Audio Open runs its own autoencoder, and
Suno/Udio are commercial systems of the same family. "Has passed through a
neural decoder" is therefore the property the generators *share*, and unlike the
generators themselves it can be **manufactured from real music at will** — no
generator access, no fake labels, no per-generator tuning.

That is Afchar, Meseguer-Brocal & Hennequin's move (arXiv:2501.10111): train on
real audio versus a reconstruction *of that same audio*, and report 99.8%. Oh's
ArtifactNet (arXiv:2604.16254) exploits the same signal from the residual side.
Neither is zero-shot with respect to the detector's own training, but both are
zero-shot with respect to the **generator set**, which is the property that
actually has to hold when a new model ships.

What this buys us that a one-class flow cannot
----------------------------------------------
Our one-class flow only ever sees real music, so it has no way to learn *which*
directions away from the real manifold correspond to synthesis rather than to a
different corpus. That is exactly the (B) confound that has dominated every
result in this project: on SONICS the flow scores 0.9383 while a two-line
channel statistic scores 1.0000, and on FakeMusicCaps it scores 0.6204 against a
channel-alone 0.6077. Proxy negatives supply the missing direction while keeping
the detector free of generator labels.

Content control is inherent
---------------------------
A reconstruction shares its source's content, genre, key, tempo, loudness and
bandwidth exactly. Any decision boundary learned between the two therefore
cannot be a corpus, production or channel boundary — the confound that
invalidated FLAC complexity (AUC 0.52 on these same pairs) and that the channel
diagnostic found at 1.0000 on SONICS is *definitionally* absent here.

Ordering matters
----------------
Real generated audio is decoded first and band-limited by distribution second.
``synthesise_negative`` follows that order: reconstruct, then re-apply the lossy
channel, then bandwidth-match. Reversing it would teach the model an artifact
signature that no distributed fake actually carries.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

ENCODEC_SR = 24_000

# Decoder families available without extra downloads. Each is a different
# artifact signature, and training across several is what stops the detector
# from learning "this is EnCodec" instead of "this is a neural decoder".
RECON_TYPES = (
    "encodec_1.5kbps",
    "encodec_3kbps",
    "encodec_6kbps",
    "encodec_24kbps",
    "griffinlim_128mel",
    "griffinlim_256mel",
)

_ENCODEC_CACHE: dict = {}


def _encodec_model(bandwidth: float, device: str):
    """Load one EnCodec model per (bandwidth, device) and keep it.

    Reloading per track would dominate the runtime of a corpus-scale build.
    """
    key = (bandwidth, device)
    if key not in _ENCODEC_CACHE:
        from encodec import EncodecModel

        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(bandwidth)
        model.to(device).eval()
        _ENCODEC_CACHE[key] = model
        logger.info("Loaded EnCodec decoder at %.1f kbps on %s", bandwidth, device)
    return _ENCODEC_CACHE[key]


def reconstruct_encodec(
    audio: npt.NDArray[np.float32], sr: int, bitrate_kbps: float, device: str = "cpu"
) -> npt.NDArray[np.float32]:
    """Round-trip audio through the EnCodec codec at a given operating point.

    Unlike the *encoder-only* use elsewhere in this project, this runs the full
    quantise-and-decode path, so the returned waveform genuinely carries
    residual-vector-quantisation artifacts. Lower bitrates mean stronger
    artifacts, which is what makes a bitrate sweep a dose-response axis.
    """
    import torch
    from encodec.utils import convert_audio

    model = _encodec_model(float(bitrate_kbps), device)

    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)
    elif wav.ndim == 2:
        wav = wav.unsqueeze(0)
    if sr != ENCODEC_SR:
        wav = convert_audio(wav, sr, ENCODEC_SR, 1)

    with torch.no_grad():
        decoded = model.decode(model.encode(wav.to(device)))

    recon = decoded.squeeze().detach().cpu().numpy().astype(np.float32)

    if sr != ENCODEC_SR:
        import librosa

        recon = librosa.resample(recon, orig_sr=ENCODEC_SR, target_sr=sr, res_type="soxr_hq")
    return np.asarray(recon, dtype=np.float32)


def reconstruct_griffinlim(
    audio: npt.NDArray[np.float32],
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_iter: int = 32,
) -> npt.NDArray[np.float32]:
    """Round-trip through mel-spectrogram inversion.

    A NON-neural decoder, included deliberately as a contrast: it introduces
    phase-reconstruction artifacts without any learned vocoder signature. If a
    detector trained on neural reconstructions also fires on Griffin-Lim, it has
    learned "reconstructed" rather than "neurally decoded", which is a weaker and
    less transferable notion — so this doubles as a diagnostic.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    stft = librosa.feature.inverse.mel_to_stft(np.sqrt(np.maximum(mel, 1e-10)), sr=sr, n_fft=n_fft)
    recon = librosa.griffinlim(stft, n_iter=n_iter, hop_length=hop_length)
    return np.asarray(recon, dtype=np.float32)


def _match_length(recon: npt.NDArray[np.float32], n: int) -> npt.NDArray[np.float32]:
    """Trim or zero-pad a reconstruction to the source length.

    Codec framing changes the sample count by a few hundred samples. Left
    unmatched, that becomes a length difference correlated with the label — a
    channel cue of exactly the kind this module exists to avoid.
    """
    if len(recon) > n:
        return recon[:n]
    if len(recon) < n:
        return np.pad(recon, (0, n - len(recon)))
    return recon


def synthesise_negative(
    audio: npt.NDArray[np.float32],
    sr: int,
    recon_type: str,
    device: str = "cpu",
    post_mp3_kbps: int | None = 64,
    resample_hz: float | None = None,
    lowpass_hz: float | None = None,
) -> npt.NDArray[np.float32]:
    """Turn one real track into a generator-agnostic negative.

    Pipeline order mirrors what happens to a distributed fake:

    1. **decode** — the neural decoder imprints its artifact;
    2. **lossy channel** — the file is then encoded for delivery
       (``post_mp3_kbps``), so the artifact is observed *through* compression
       rather than in pristine form;
    3. **bandwidth match** — finally the shared ceiling is applied, identically
       to positives and negatives.

    Getting this order wrong produces a negative whose artifact has never been
    through a codec, which no real fake resembles.
    """
    from ..data.audio_preprocessing import bandwidth_match, mp3_roundtrip

    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)

    if recon_type.startswith("encodec_"):
        kbps = float(recon_type.split("_")[1].replace("kbps", ""))
        recon = reconstruct_encodec(audio, sr, kbps, device=device)
    elif recon_type.startswith("griffinlim_"):
        n_mels = int(recon_type.split("_")[1].replace("mel", ""))
        recon = reconstruct_griffinlim(audio, sr, n_mels=n_mels)
    else:
        raise ValueError(f"unknown recon_type {recon_type!r}; choose from {list(RECON_TYPES)}")

    recon = _match_length(recon, n)

    if post_mp3_kbps:
        recon = _match_length(mp3_roundtrip(recon, sr, bitrate_kbps=post_mp3_kbps), n)

    if resample_hz or lowpass_hz:
        recon = bandwidth_match(recon, sr, resample_hz=resample_hz, lowpass_hz=lowpass_hz)

    return np.asarray(recon, dtype=np.float32)
