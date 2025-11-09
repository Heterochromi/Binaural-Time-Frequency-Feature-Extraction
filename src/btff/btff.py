import torch
import torchaudio
import torchaudio.transforms as T


class BtffTransoform:
    def __init__(
        self,
        input_audio_path,
        n_fft=1024,
        n_mels=128,
        fmin=0,
        fmax=16000,
        device="cpu" or "cuda" or None,
    ):
        self.device = torch.device(device)
        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels
        self.n_fft = n_fft
        waveform, self.sr = torchaudio.load(input_audio_path)
        waveform = waveform.to(self.device)

        self.bin_width = self.sr / self.n_fft
        self.hop_size = round(n_fft / 4)

        self.eps = 1e-6

        if waveform.shape[0] == 1:
            ValueError("the audio file that was passed in mono not stereo")
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        stft_transform = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.n_fft,
            window_fn=torch.hann_window,
            power=None,  # None returns complex spectrogram
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

        self.complex_spectrogram_left = stft_transform(waveform[0])
        self.complex_spectrogram_right = stft_transform(waveform[1])

        self.left_mag = torch.abs(self.complex_spectrogram_left)
        self.left_phase = torch.angle(self.complex_spectrogram_left)

        self.right_mag = torch.abs(self.complex_spectrogram_right)
        self.right_phase = torch.angle(self.complex_spectrogram_right)

        self.left_mag = torch.clamp_min(self.left_mag, min=self.eps)
        self.right_mag = torch.clamp_min(self.right_mag, min=self.eps)

        [self.numbins, self.numframes] = self.left_mag.shape
        # self.spectra =torch.zeros((self.numbins , self.numframes)) for reference on  how to initilize a spec space
        self.intensity = self.left_mag + self.right_mag
        self.phasediffs = self.left_phase - self.right_phase

    def ITD_spect(self, start_freq=0, stop_freq=1500):
        if start_freq < 0 or stop_freq > self.sr / 2:
            raise ValueError(
                "Invalid frequency range. Valid range is [0, {}]".format(self.sr / 2)
            )
        if start_freq >= stop_freq:
            raise ValueError(
                "Invalid frequency range. Valid range is [0, {}]".format(self.sr / 2)
            )

        itd = torch.zeros((self.numbins, self.numframes), device=self.device)

        startbin = int(round(start_freq / self.bin_width))
        stopbin = int(round(stop_freq / self.bin_width))

        phasediff = self.phasediffs[startbin:stopbin, :]
        wrapped_phase_diff = (
            torch.remainder(phasediff + torch.pi, 2 * torch.pi) - torch.pi
        )
        bin_indices = torch.arange(startbin, stopbin, device=self.device).view(-1, 1)

        if startbin == 0:
            bindelay = wrapped_phase_diff / (2 * torch.pi * self.bin_width)
            bindelay[1:] = wrapped_phase_diff[1:] / (
                2 * torch.pi * self.bin_width * bin_indices[1:]
            )
        else:
            bindelay = wrapped_phase_diff / (
                2 * torch.pi * self.bin_width * bin_indices
            )

        intensity_mask = self.intensity[startbin:stopbin, :] >= self.eps
        itd[startbin:stopbin, :] = bindelay * intensity_mask
        return self.mel_fb(itd)

    def ILD_spect(self, start_freq=5000):
        startbin = int(round(start_freq / self.bin_width))
        left_mag = self.left_mag.clone()
        right_mag = self.right_mag.clone()

        left_mag[:startbin, :] = self.eps
        right_mag[:startbin, :] = self.eps

        ild = (20 * torch.log10(left_mag)) - (20 * torch.log10(right_mag))

        return self.mel_fb(ild)

    def v_map(self):
        mel_left_spec, mel_right_spec = self.mel_spect()

        v_map_left = torch.zeros_like(mel_left_spec, device=self.device)
        v_map_left[0] = mel_left_spec[1] - mel_left_spec[0]
        v_map_left[1:-1] = (mel_left_spec[2:] - mel_left_spec[:-2]) / 2
        v_map_left[-1] = mel_left_spec[-1] - mel_left_spec[-2]

        v_map_right = torch.zeros_like(mel_right_spec, device=self.device)
        v_map_right[0] = mel_right_spec[1] - mel_right_spec[0]
        v_map_right[1:-1] = (mel_right_spec[2:] - mel_right_spec[:-2]) / 2
        v_map_right[-1] = mel_right_spec[-1] - mel_right_spec[-2]
        return v_map_left, v_map_right

    def mel_spect(self):
        # mel_left_spec = self.left_mag.pow(2)
        # mel_right_spec = self.right_mag.pow(2)

        mel_left_spec = self.mel_fb(self.left_mag)
        mel_right_spec = self.mel_fb(self.right_mag)

        mel_left_spec = torch.log10(mel_left_spec)
        mel_right_spec = torch.log10(mel_right_spec)

        return mel_left_spec, mel_right_spec

    def sc_map(self, start_freq=5000):
        starttbin = int(round(start_freq / self.bin_width))

        left_mag = self.left_mag.clone()
        right_mag = self.right_mag.clone()

        left_mag[:starttbin, :] = self.eps
        right_mag[:starttbin, :] = self.eps

        # sc_map_left = left_mag.pow(2)
        # sc_map_right = right_mag.pow(2)

        sc_map_left = self.mel_fb(left_mag)
        sc_map_right = self.mel_fb(right_mag)

        sc_map_left = torch.log10(sc_map_left)
        sc_map_right = torch.log10(sc_map_right)

        return sc_map_left, sc_map_right

    def process_file(
        self,
        itd_start_freq=0,
        itd_stop_freq=1500,
        ild_start_freq=5000,
        sc_map_start_freq=5000,
    ):
        itd = self.ITD_spect(start_freq=itd_start_freq, stop_freq=itd_stop_freq)
        ild = self.ILD_spect(start_freq=ild_start_freq)
        mel_left_spec, mel_right_spec = self.mel_spect()
        v_map_left, v_map_right = self.v_map()
        sc_map_left, sc_map_right = self.sc_map(start_freq=sc_map_start_freq)

        return {
            "itd": itd,
            "ild": ild,
            "mel_left_spec": mel_left_spec,
            "mel_right_spec": mel_right_spec,
            "v_map_left": v_map_left,
            "v_map_right": v_map_right,
            "sc_map_left": sc_map_left,
            "sc_map_right": sc_map_right,
        }
