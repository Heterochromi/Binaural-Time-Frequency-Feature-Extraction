from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torchaudio
import torchaudio.transforms as T


@dataclass
class BtffBatch:
    """
            PyTorch conventions 🙂☺.

    Attributes:
        features: Stacked tensor of shape [batch, num_features, n_mels, time_frames]
                 where num_features = 8 (itd, ild, mel_l, mel_r, v_l, v_r, sc_l, sc_r)
        feature_dict: Dictionary mapping feature names to tensors [batch, n_mels, time_frames]
        paths: List of file paths corresponding to each batch item
        batch_size: Number of samples in this batch

    You can access features via:
        - batch.features[i]           # Get all features for sample i
        - batch.feature_dict['itd']   # Get ITD for entire batch
        - batch['itd']                # Dictionary-style access
        - batch.to('cuda')            # Move all tensors to device
    """

    features: torch.Tensor  # [batch, 8, n_mels, time_frames]
    feature_dict: dict  # Individual feature tensors
    paths: List[str]
    batch_size: int

    def __getitem__(self, key):
        """Allow dict-style access: batch['itd']"""
        if isinstance(key, str):
            return self.feature_dict[key]
        elif isinstance(key, int):
            # Return all features for a specific sample
            return {k: v[key] for k, v in self.feature_dict.items()}
        else:
            raise KeyError(f"Invalid key type: {type(key)}")

    def to(self, device):
        """Move all tensors to specified device"""
        self.features = self.features.to(device)
        self.feature_dict = {k: v.to(device) for k, v in self.feature_dict.items()}
        return self

    def cpu(self):
        """Move all tensors to CPU"""
        return self.to("cpu")

    def cuda(self):
        """Move all tensors to CUDA"""
        return self.to("cuda")

    def pin_memory(self):
        """Pin memory for faster GPU transfer"""
        self.features = self.features.pin_memory()
        self.feature_dict = {k: v.pin_memory() for k, v in self.feature_dict.items()}
        return self

    def keys(self):
        """Return feature names"""
        return self.feature_dict.keys()

    def values(self):
        """Return feature tensors"""
        return self.feature_dict.values()

    def items(self):
        """Return (name, tensor) pairs"""
        return self.feature_dict.items()


class BtffBatchProcessor:
    """
    Efficient batch processor for BTFF transforms.
    Processes multiple audio files in parallel on GPU.
    """

    def __init__(
        self,
        n_fft=1024,
        n_mels=128,
        fmin=0,
        fmax=16000,
        device="cpu" or "cuda" or None,
        batch_size=32,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.batch_size = batch_size
        self.hop_size = round(n_fft / 4)
        self.eps = 1e-6

        # These will be initialized when we know the sample rate
        self.sr = None
        self.stft_transform = None
        self.mel_fb = None
        self.bin_width = None

    def _initialize_transforms(self, sr):
        """Initialize transforms once we know the sample rate"""
        if self.sr != sr:
            self.sr = sr
            self.bin_width = self.sr / self.n_fft

            self.stft_transform = T.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_size,
                win_length=self.n_fft,
                window_fn=torch.hann_window,
                power=None,
                normalized=False,
                center=True,
                pad_mode="reflect",
                onesided=True,
            ).to(self.device)

            self.mel_fb = T.MelScale(
                n_mels=self.n_mels,
                sample_rate=self.sr,
                f_min=self.fmin,
                f_max=self.fmax,
                n_stft=self.n_fft // 2 + 1,
                norm="slaney",
                mel_scale="slaney",
            ).to(self.device)

    def load_batch(
        self, audio_paths: List[str], max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load multiple audio files and create a batched tensor.

        Args:
            audio_paths: List of paths to audio files
            max_length: Maximum length to pad/trim to. If None, uses longest in batch.

        Returns:
            Tuple of (left_channels, right_channels) with shape [batch, samples]
        """
        waveforms = []
        sample_rate = None

        for path in audio_paths:
            waveform, sr = torchaudio.load_with_torchcodec(path)

            if sample_rate is None:
                sample_rate = sr
                self._initialize_transforms(sr)
            elif sr != sample_rate:
                raise ValueError(
                    f"All audio files must have same sample rate. Expected {sample_rate}, got {sr}"
                )

            # Handle channels
            if waveform.shape[0] == 1:
                ValueError(
                    f"Audio file {path} is mono, not stereo , left and right channels will be cloned for this file.(no real HRTF information)"
                )
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]

            waveforms.append(waveform)

        # Find max length if not specified
        if max_length is None:
            print(
                "no max length specified, using longest in batch for now, this could be problematic denpending on your use case, please specify max length or ensure all audio files are the same length"
            )
            max_length = max(w.shape[1] for w in waveforms)

        # Pad or trim to same length
        left_batch = []
        right_batch = []

        for waveform in waveforms:
            if waveform.shape[1] < max_length:
                # Pad
                padding = max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.shape[1] > max_length:
                # Trim
                waveform = waveform[:, :max_length]

            left_batch.append(waveform[0])
            right_batch.append(waveform[1])

        left_batch = torch.stack(left_batch).to(self.device)
        right_batch = torch.stack(right_batch).to(self.device)

        return left_batch, right_batch

    def _compute_spectrogram_data(
        self, left_batch: torch.Tensor, right_batch: torch.Tensor
    ):
        """
        Compute spectrograms for a batch of audio.

        Args:
            left_batch: [batch, samples]
            right_batch: [batch, samples]

        Returns:
            Dictionary with all intermediate representations
        """
        # Compute STFT for entire batch
        complex_spec_left = self.stft_transform(left_batch)  # [batch, freq, time]
        complex_spec_right = self.stft_transform(right_batch)

        # Magnitude and phase
        left_mag = torch.abs(complex_spec_left)
        left_phase = torch.angle(complex_spec_left)
        right_mag = torch.abs(complex_spec_right)
        right_phase = torch.angle(complex_spec_right)

        # Clamp magnitudes
        left_mag = torch.clamp_min(left_mag, min=self.eps)
        right_mag = torch.clamp_min(right_mag, min=self.eps)

        # Compute intensity and phase difference
        intensity = left_mag + right_mag
        phasediffs = left_phase - right_phase

        return {
            "left_mag": left_mag,
            "right_mag": right_mag,
            "left_phase": left_phase,
            "right_phase": right_phase,
            "intensity": intensity,
            "phasediffs": phasediffs,
        }

    def ITD_spect_batch(self, spectrograms: dict, start_freq=0, stop_freq=1500):
        """Compute ITD spectrogram for batch"""
        phasediffs = spectrograms["phasediffs"]
        intensity = spectrograms["intensity"]

        batch_size, numbins, numframes = phasediffs.shape
        itd = torch.zeros((batch_size, numbins, numframes), device=self.device)

        startbin = int(round(start_freq / self.bin_width))
        stopbin = int(round(stop_freq / self.bin_width))

        # Vectorized computation across batch
        phasediff = phasediffs[:, startbin:stopbin, :]
        wrapped_phase_diff = (
            torch.remainder(phasediff + torch.pi, 2 * torch.pi) - torch.pi
        )

        # Create bin indices [1, bins, 1] for broadcasting
        bin_indices = torch.arange(startbin, stopbin, device=self.device).view(1, -1, 1)

        if startbin == 0:
            bindelay = wrapped_phase_diff / (2 * torch.pi * self.bin_width)
            bindelay[:, 1:, :] = wrapped_phase_diff[:, 1:, :] / (
                2 * torch.pi * self.bin_width * bin_indices[:, 1:, :]
            )
        else:
            bindelay = wrapped_phase_diff / (
                2 * torch.pi * self.bin_width * bin_indices
            )

        # intensity masking
        intensity_mask = intensity[:, startbin:stopbin, :] >= self.eps
        itd[:, startbin:stopbin, :] = bindelay * intensity_mask

        # Apply mel filterbank to each item in batch
        return torch.stack([self.mel_fb(itd[i]) for i in range(batch_size)])

    def ILD_spect_batch(self, spectrograms: dict, start_freq=5000):
        """Compute ILD spectrogram for batch"""
        left_mag = spectrograms["left_mag"].clone()
        right_mag = spectrograms["right_mag"].clone()

        startbin = int(round(start_freq / self.bin_width))

        left_mag[:, :startbin, :] = self.eps
        right_mag[:, :startbin, :] = self.eps

        ild = (20 * torch.log10(left_mag)) - (20 * torch.log10(right_mag))

        return torch.stack([self.mel_fb(ild[i]) for i in range(ild.shape[0])])

    def mel_spect_batch(self, spectrograms: dict):
        """Compute Mel spectrogram for batch"""
        batch_length = spectrograms["left_mag"].shape[0]
        mel_left_spec = torch.stack(
            [self.mel_fb(spectrograms["left_mag"][i]) for i in range(batch_length)]
        )
        mel_right_spec = torch.stack(
            [self.mel_fb(spectrograms["right_mag"][i]) for i in range(batch_length)]
        )
        mel_left_spec = torch.log10(mel_left_spec)
        mel_right_spec = torch.log10(mel_right_spec)

        return mel_left_spec, mel_right_spec

    def process_files(
        self,
        audio_paths: List[str],
        itd_start_freq=0,
        itd_stop_freq=1500,
        ild_start_freq=5000,
        sc_map_start_freq=5000,
    ):
        """
        Args:
            audio_paths: List of paths to audio files
            itd_start_freq: Start frequency for ITD computation (Hz)
            itd_stop_freq: Stop frequency for ITD computation (Hz)
            ild_start_freq: Start frequency for ILD computation (Hz)
            sc_map_start_freq: Start frequency for SC map computation (Hz)
            return_dict: If True, returns dict. If False, returns BTFFBatch object (default)

        Yields:
            BTFFBatch object containing:
                - features: Stacked tensor [batch, 8, n_mels, time_frames]
                - feature_dict: Dictionary of individual features
                - paths: List of file paths
                - batch_size: Number of samples in batch

        Example:
            >>> processor = BTFFBatchProcessor(batch_size=16)
            >>> for batch in processor.process_files(audio_paths):
            >>>     # Access as tensor
            >>>     all_features = batch.features  # [16, 8, 128, time]
            >>>
            >>>     # Access individual features
            >>>     itd = batch['itd']  # [16, 128, time]
            >>>
            >>>     # Move to GPU
            >>>     batch = batch.to('cuda')
            >>>
            >>>     # Access by sample
            >>>     sample_0_features = batch[0]  # Dict of all features for first sample
        """
        num_files = len(audio_paths)

        for i in range(0, num_files, self.batch_size):
            batch_paths = audio_paths[i : i + self.batch_size]

            # Load batch
            left_batch, right_batch = self.load_batch(batch_paths)

            # Compute spectrograms
            spectrograms = self._compute_spectrogram_data(left_batch, right_batch)

            # Compute all features
            itd = self.ITD_spect_batch(spectrograms, itd_start_freq, itd_stop_freq)
            ild = self.ILD_spect_batch(spectrograms, ild_start_freq)

            mel_left_spec, mel_right_spec = self.mel_spect_batch(spectrograms)

            # V-maps
            v_map_left = self.v_map_batch(mel_left_spec)
            v_map_right = self.v_map_batch(mel_right_spec)

            # SC-maps
            sc_map_left, sc_map_right = self.sc_map_batch(
                spectrograms, sc_map_start_freq
            )

            # Create feature dictionary
            feature_dict = {
                "itd": itd,
                "ild": ild,
                "mel_left_spec": mel_left_spec,
                "mel_right_spec": mel_right_spec,
                "v_map_left": v_map_left,
                "v_map_right": v_map_right,
                "sc_map_left": sc_map_left,
                "sc_map_right": sc_map_right,
            }

            stacked_features = torch.stack(
                [
                    itd,
                    ild,
                    mel_left_spec,
                    mel_right_spec,
                    v_map_left,
                    v_map_right,
                    sc_map_left,
                    sc_map_right,
                ],
                dim=1,
            )  # [batch, 8, n_mels, time_frames]
            # Create BTFFBatch object
            yield BtffBatch(
                features=stacked_features,
                feature_dict=feature_dict,
                paths=batch_paths,
                batch_size=len(batch_paths),
            )

    def v_map_batch(self, mel_batch):
        """Compute velocity map for batch"""
        v_map = torch.zeros_like(mel_batch)
        v_map[:, 0, :] = mel_batch[:, 1, :] - mel_batch[:, 0, :]
        v_map[:, 1:-1, :] = (mel_batch[:, 2:, :] - mel_batch[:, :-2, :]) / 2
        v_map[:, -1, :] = mel_batch[:, -1, :] - mel_batch[:, -2, :]
        return v_map

    def sc_map_batch(self, spectrograms: dict, start_freq=5000):
        """Compute SC map for batch"""
        startbin = int(round(start_freq / self.bin_width))

        left_mag = spectrograms["left_mag"].clone()
        right_mag = spectrograms["right_mag"].clone()

        left_mag[:, :startbin, :] = self.eps
        right_mag[:, :startbin, :] = self.eps

        batch_size = left_mag.shape[0]
        sc_map_left = torch.stack([self.mel_fb(left_mag[i]) for i in range(batch_size)])
        sc_map_right = torch.stack(
            [self.mel_fb(right_mag[i]) for i in range(batch_size)]
        )

        sc_map_left = torch.log10(sc_map_left)
        sc_map_right = torch.log10(sc_map_right)

        return sc_map_left, sc_map_right
