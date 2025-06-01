def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class AugmentWAV(object):

    def __init__(self, musan_path='/workspace/data/sjkim/argument/musan/', rir_path='/workspace/data/sjkim/argument/RIRS_NOISES/', max_frames=200):
        self.max_frames = max_frames
        self.max_audio = max_frames * 160
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise':[0, 15], 'speech':[13, 20], 'music':[0, 15]}
        self.numnoise = {'noise':[1, 1], 'speech':[3, 7], 'music':[1, 1]}
        self.noiselist = {}
        
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        for file in augment_files:
            key = file.split('/')[-3]
            if key not in self.noiselist:
                self.noiselist[key] = []
            self.noiselist[key].append(file)
            
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*/*.wav'))
        self.perturb_prob = 1.0
        self.speeds = [0.95, 1.05]  # 속도 비율 (예: 0.95, 1.05)
        self.sample_rate = 16000

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio**2) + 1e-4) 
        numnoise = self.numnoise[noisecat]
        
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        
        noises = []
        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0]**2) + 1e-4) 
            scale = np.sqrt(10**((clean_db - noise_db - noise_snr) / 10))
            noises.append(scale * noiseaudio)
        # noise들을 concatenate 후 합산하고 audio와 더함
        noise_sum = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise_sum + audio

    def additive_noise2(data):
        noise_amp = 0.035 * np.random.uniform() * np.amax(data)
        return data + noise_amp * np.random.normal(size=data.shape[0])

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, fs = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float32), 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        conv_audio = signal.convolve(audio, rir, mode='full')
        return conv_audio[:, :self.max_audio]

    def speed_perturb(self, audio):
        if torch.rand(1).item() > self.perturb_prob:
            return audio
        speed_rate = random.choice(self.speeds)
        # librosa를 이용한 시간 스트레칭 (채널 차원 유지)
        audio_np = audio.squeeze(0)
        perturbed = librosa.effects.time_stretch(audio_np, rate=speed_rate)
        # 길이가 달라질 수 있으므로 crop/pad
        if len(perturbed) > self.max_audio:
            perturbed = perturbed[:self.max_audio]
        else:
            perturbed = np.pad(perturbed, (0, self.max_audio - len(perturbed)), mode='constant')
        return np.expand_dims(perturbed, axis=0)

    def volume_control_augment(self, audio):
        return volume_control(audio)

    def time_strech(self, audio):
        rate = random.uniform(0.7, 1.3)
        audio_np = audio.squeeze(0)
        stretched = stretch(audio_np, rate)
        if len(stretched) > self.max_audio:
            stretched = stretched[:self.max_audio]
        else:
            stretched = np.pad(stretched, (0, self.max_audio - len(stretched)), mode='constant')
        return np.expand_dims(stretched, axis=0)

    # ===== 추가된 함수들 =====
    def pre_emphasis(self, audio, coeff=0.97):
        """Pre-emphasis: y[t] = x[t] - coeff * x[t-1]"""
        audio_np = audio.squeeze(0)
        emphasized = np.append(audio_np[0], audio_np[1:] - coeff * audio_np[:-1])
        return np.expand_dims(emphasized, axis=0)

    def pitch_shift(self, audio, n_steps=2):
        """Apply pitch shift using librosa"""
        audio_np = audio.squeeze(0)
        shifted = librosa.effects.pitch_shift(audio_np, self.sample_rate, n_steps=n_steps)
        return np.expand_dims(shifted, axis=0)

    def vad(self, audio, threshold=0.01):
        """Simple energy-based Voice Activity Detection (VAD)"""
        audio_np = audio.squeeze(0)
        energy = np.abs(audio_np)
        vad_audio = np.where(energy > threshold, audio_np, 0)
        return np.expand_dims(vad_audio, axis=0)

    def hpss(self, audio):
        """Extract harmonic and percussive components via HPSS"""
        audio_np = audio.squeeze(0)
        harmonic, percussive = librosa.effects.hpss(audio_np)
        # 여기서는 harmonic과 percussive를 평균내서 재구성하는 예시
        merged = (harmonic + percussive) / 2
        return np.expand_dims(merged, axis=0)

    def shift_audio(self, audio, max_shift_ms=5):
    """Random temporal shift"""
    shift_range = int(np.random.uniform(low=-max_shift_ms, high=max_shift_ms) * self.sample_rate / 1000)
    audio_np = audio.squeeze(0)
    shifted = np.roll(audio_np, shift_range)
    return np.expand_dims(shifted, axis=0)

    def time_strech(self, audio, rate=None):
        if rate is None:
            rate = random.uniform(0.7, 1.3)
        audio_np = audio.squeeze(0)
        stretched = librosa.effects.time_stretch(audio_np, rate)
        if len(stretched) > self.max_audio:
            stretched = stretched[:self.max_audio]
        else:
            stretched = np.pad(stretched, (0, self.max_audio - len(stretched)), mode='constant')
        return np.expand_dims(stretched, axis=0)

    def pitch_shift(self, audio, n_steps=None):
        if n_steps is None:
            n_steps = random.choice([-2, -1, 1, 2])
        audio_np = audio.squeeze(0)
        shifted = librosa.effects.pitch_shift(audio_np, self.sample_rate, n_steps=n_steps)
        return np.expand_dims(shifted, axis=0)

    # ===== 전체 파이프라인 예시 함수 =====
    def augment(self, audio, apply_preemphasis='None'):
      
        # Additive noise (예: noise 카테고리 사용)
        if random.random() < 0.5:
            audio = self.additive_noise('noise', audio)
        # Reverberation
        if random.random() < 0.3:
            audio = self.reverberate(audio)
        # Speed perturbation
#         if random.random() < 0.5:
#             audio = self.speed_perturb(audio)
        # Volume control
#         if random.random() < 0.5:
#             audio = self.volume_control_augment(audio)
        # Time stretching
#         if random.random() < 0.5:
#             audio = self.time_strech(audio)
        # Low pass filtering
#         if random.random() < 0.3:
#             audio = self.low_pass_filter(audio)
        # FIR bandpass filtering
        if random.random() < 0.3:
            audio = self.fir_bandpass_filter(audio)
        # Pitch shifting
#         if random.random() < 0.5:
#             n_steps = random.choice([-2, -1, 1, 2])
#             audio = self.pitch_shift(audio, n_steps=n_steps)
        # VAD
        if random.random() < 0.5:
            audio = self.vad(audio)
        # HPSS (harmonic/percussive)
        if random.random() < 0.5:
            audio = self.hpss(audio)
        # Pre-emphasis: 'always', 'augment' (랜덤 적용) 또는 'none'
        if apply_preemphasis == 'always':
            audio = self.pre_emphasis(audio)
        elif apply_preemphasis == 'augment' and random.random() < 0.5:
            audio = self.pre_emphasis(audio)
        
        return audio
