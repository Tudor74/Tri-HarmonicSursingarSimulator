"""
Tri-Harmonic Music — Sursringar Pitch
(Pulse origin fixed • Master Volume • Multi-strings • Click fix • Soft Attack • Slow Tempo • Wavering)

Pure sine (phase-continuous) • Rāga-constrained, Sa-anchored motifs • Qt UI
Stats: Rāga / Tonic / Tuning / Recent Notes.

Adds:
- CLICK FIX: no per-block attack; conditional phase reset on pluck
- Soft pluck attack: stateful envelope with attack→target and long decay across blocks
- Slower tempos: speed range 0.02× … 5×
- Wavering controls: Vibrato depth (cents) & rate (Hz), plus amplitude shimmer (%)

Requires: PyQt5, vispy, sounddevice (auto-installs).
"""

import sys
import time
import logging
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

# -----------------------------------------------------------------------------
# Dependency bootstrap
# -----------------------------------------------------------------------------
def _ensure_pkg(mod_name: str, pip_name: Optional[str] = None):
    try:
        __import__(mod_name)
    except ImportError:
        print(f"[setup] {mod_name} not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or mod_name])

_ensure_pkg("PyQt5", "PyQt5")
_ensure_pkg("vispy", "vispy")
_ensure_pkg("sounddevice", "sounddevice")

from PyQt5 import QtWidgets, QtCore
from vispy import app, scene
import sounddevice as sd

app.use_app("pyqt5")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------------------------------------------------------
# Core data
# -----------------------------------------------------------------------------
class SphereFunction(Enum):
    TEMPORAL = "temporal_memory"
    SEMANTIC = "semantic_drift"
    HARMONIC = "harmonic_feedback"

@dataclass
class DataPoint:
    value: float
    timestamp: float
    symbol: str
    phase: float
    spectrum: Optional[np.ndarray]
    energy: float
    duration: float
    metadata: Dict[str, Any] = None

    def clone(self):
        return DataPoint(
            value=self.value,
            timestamp=self.timestamp,
            symbol=self.symbol,
            phase=self.phase,
            spectrum=self.spectrum.copy() if self.spectrum is not None else None,
            energy=self.energy,
            duration=self.duration,
            metadata=self.metadata.copy() if self.metadata else {},
        )

# -----------------------------------------------------------------------------
# Tuning systems
# -----------------------------------------------------------------------------
class TuningPreset(Enum):
    JI_5_LIMIT = "5-limit JI"
    PYTHAGOREAN = "Pythagorean"
    SHRUTI_22_APPROX = "22-shruti (approx)"
    TET_12 = "12-TET"

SWARA_SEMITONES = {
    "S": 0, "r": 1, "R": 2, "g": 3, "G": 4, "m": 5, "M": 6,
    "P": 7, "d": 8, "D": 9, "n": 10, "N": 11, "S'": 12
}

JI_5_LIMIT_RATIOS = {
    "S": 1/1, "r": 16/15, "R": 9/8, "g": 6/5, "G": 5/4,
    "m": 4/3, "M": 45/32, "P": 3/2, "d": 8/5, "D": 5/3,
    "n": 9/5, "N": 15/8, "S'": 2/1
}

PYTHAGOREAN_RATIOS = {
    "S": 1/1,
    "r": 256/243, "R": 9/8,
    "g": 32/27,   "G": 81/64,
    "m": 4/3,     "M": 729/512,
    "P": 3/2,
    "d": 128/81,  "D": 27/16,
    "n": 16/9,    "N": 243/128,
    "S'": 2/1
}

SHRUTI22_APPROX_RATIOS = {
    "S": 1/1, "r": 16/15, "R": 10/9,
    "g": 6/5, "G": 5/4,
    "m": 4/3, "M": 45/32,
    "P": 3/2,
    "d": 8/5, "D": 5/3,
    "n": 9/5, "N": 15/8,
    "S'": 2/1
}

# -----------------------------------------------------------------------------
# Pitch system (with presets + 12-TET override)
# -----------------------------------------------------------------------------
class PitchSystem:
    RAGAS = {
        "Yaman":   ["S","R","G","M","P","D","N","S'"],
        "Bhairav": ["S","r","G","m","P","d","N","S'"],
        "Kafi":    ["S","R","g","m","P","D","n","S'"],
        "Bhairavi":["S","r","g","m","P","d","n","S'"],
        "Darbari": ["S","R","g","m","P","D","n","S'"],
    }
    def __init__(self, tonic_hz: float = 220.0, raga_name: str = "Yaman",
                 preset: TuningPreset = TuningPreset.JI_5_LIMIT,
                 force_12tet: bool = False):
        self.tonic_hz = float(tonic_hz)
        self.set_raga(raga_name)
        self.preset = preset
        self.force_12tet = force_12tet

    def set_tonic(self, hz: float):
        self.tonic_hz = float(max(40.0, min(1000.0, hz)))

    def set_raga(self, name: str):
        if name not in self.RAGAS:
            name = "Yaman"
        self.raga_name = name
        self.allowed_swaras = list(self.RAGAS[name])

    def set_preset(self, preset: TuningPreset):
        self.preset = preset

    def set_force_12tet(self, on: bool):
        self.force_12tet = bool(on)

    def swara_ratio(self, swara: str) -> float:
        if self.force_12tet or self.preset == TuningPreset.TET_12:
            semis = SWARA_SEMITONES[swara]
            return 2.0 ** (semis / 12.0)
        if self.preset == TuningPreset.JI_5_LIMIT:
            return JI_5_LIMIT_RATIOS[swara]
        if self.preset == TuningPreset.PYTHAGOREAN:
            return PYTHAGOREAN_RATIOS[swara]
        return SHRUTI22_APPROX_RATIOS[swara]

    def swara_to_hz(self, swara: str, octave_shift: int = 0) -> float:
        ratio = self.swara_ratio(swara)
        return self.tonic_hz * ratio * (2.0 ** octave_shift)

# -----------------------------------------------------------------------------
# Audio engine (phase-continuous; meend; drone; strings; vibrato/shimmer)
# -----------------------------------------------------------------------------
class ContinuousAudioPlayer:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.current_chord: List[Tuple[float, float]] = []
        self.target_chord:  List[Tuple[float, float]] = []
        self.voice_phases:  List[float] = []
        self.voice_gains:   List[float] = []
        self._lock = threading.Lock()

        # Master volume
        self.master_gain = 0.8  # 0..1

        # Meend
        self.meend_enabled = False
        self.glide_ms = 60.0
        self._alpha_gain_base = 0.08

        # Vibrato / shimmer (main voice)
        self.vibrato_depth_cents = 4.0  # UI adjustable
        self.vibrato_rate_hz = 5.5      # UI adjustable
        self.shimmer_pct = 2.0          # amplitude wander %, UI
        self._time_sec = 0.0
        self._voice_lfo_phase = []      # per-voice random LFO phase

        # Drone (Sa/Pa)
        self.drone_enabled = True
        self.drone_freqs = (0.0, 0.0)  # (Sa, Pa)
        self.drone_gain = 0.03
        self.drone_phases = [0.0, 0.0]

        # Sympathetic strings (plucked)
        self.strings_enabled = True
        self.string_decay_ms = 2400.0
        self.string_attack_ms = 18.0  # NEW: smooth pluck
        self.string_detune_ppm = 1200.0  # ±0.12% max
        # Each string: dict(freq, phase, env, env_target, base_gain)
        self.strings: List[Dict[str, float]] = []
        self._string_last_pluck_ts = 0.0

        self.last_pulse_info = None
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._audio_callback,
        )
        self.stream.start()

    # --- Controls ---
    def set_master_gain(self, g: float):
        with self._lock:
            self.master_gain = float(np.clip(g, 0.0, 1.0))

    def set_meend(self, enabled: bool, glide_ms: Optional[float] = None):
        with self._lock:
            self.meend_enabled = bool(enabled)
            if glide_ms is not None:
                self.glide_ms = float(max(1.0, min(5000.0, glide_ms)))

    def set_vibrato(self, depth_cents: float, rate_hz: float, shimmer_pct: float):
        with self._lock:
            self.vibrato_depth_cents = float(max(0.0, min(50.0, depth_cents)))
            self.vibrato_rate_hz = float(max(0.1, min(12.0, rate_hz)))
            self.shimmer_pct = float(max(0.0, min(20.0, shimmer_pct)))

    def set_drone_from_tonic(self, tonic_hz: float):
        pa = tonic_hz * 3.0 / 2.0
        with self._lock:
            self.drone_freqs = (float(tonic_hz), float(pa))

    def configure_strings(self, tonic_hz: float):
        rng = np.random.default_rng()
        base = [tonic_hz * 1.0, tonic_hz * 3.0/2.0, tonic_hz * 2.0]
        new_strings = []
        for i, f in enumerate(base):
            det_ppm = (rng.random()*2 - 1) * self.string_detune_ppm
            det = 1.0 + det_ppm * 1e-6
            new_strings.append({
                "freq": float(f * det),
                "phase": 0.0,
                "env": 0.0,            # instantaneous envelope
                "env_target": 0.0,     # target the env eases toward
                "base_gain": 0.05 if i == 0 else 0.035,
            })
        with self._lock:
            self.strings = new_strings

    def set_string_attack(self, attack_ms: float):
        with self._lock:
            self.string_attack_ms = float(max(1.0, min(200.0, attack_ms)))

    def pluck_strings(self, indices: List[int], velocity: float = 1.0):
        """Trigger plucked envelopes on selected strings, with smooth attack to env_target."""
        vel = float(np.clip(velocity, 0.05, 1.0))
        with self._lock:
            for idx in indices:
                if 0 <= idx < len(self.strings):
                    s = self.strings[idx]
                    was_silent = (s["env"] <= 1e-4)
                    # raise target; attack slews toward it inside the callback
                    s["env_target"] = max(s["env_target"], vel)
                    if was_silent:
                        s["phase"] = 0.0  # only if silent
            self._string_last_pluck_ts = time.time()

    # --- Internals ---
    def _glide_alpha(self) -> float:
        if not self.meend_enabled:
            return 1.0
        block_time = 1024 / self.sample_rate
        tau = self.glide_ms / 1000.0
        if tau <= 0.0:
            return 1.0
        return 1.0 - np.exp(-block_time / tau)

    def _ensure_voice_state(self, n: int):
        while len(self.voice_phases) < n:
            self.voice_phases.append(0.0)
            self.voice_gains.append(0.0)
            self._voice_lfo_phase.append(np.random.random() * 2*np.pi)
        if len(self.voice_phases) > n:
            self.voice_phases = self.voice_phases[:n]
            self.voice_gains = self.voice_gains[:n]
            self._voice_lfo_phase = self._voice_lfo_phase[:n]

    def _audio_callback(self, outdata, frames, time_info, status):
        with self._lock:
            tgt = list(self.target_chord)
            cur = list(self.current_chord)
            drone_freqs = self.drone_freqs
            drone_gain = self.drone_gain
            drone_enabled = self.drone_enabled
            strings = [s.copy() for s in self.strings]  # shallow copy
            strings_enabled = self.strings_enabled
            string_decay_ms = self.string_decay_ms
            string_attack_ms = self.string_attack_ms
            master_gain = self.master_gain
            vib_depth_c = self.vibrato_depth_cents
            vib_rate_hz = self.vibrato_rate_hz
            shimmer_pct = self.shimmer_pct

        if not tgt and not drone_enabled and not strings_enabled:
            outdata.fill(0.0)
            return

        n = max(len(cur), len(tgt))
        while len(cur) < n: cur.append((0.0, 0.0))
        while len(tgt) < n: tgt.append((0.0, 0.0))
        self._ensure_voice_state(n)

        alpha_gain = self._alpha_gain_base
        alpha_freq = self._glide_alpha()

        t = np.arange(frames, dtype=np.float32) / self.sample_rate
        time0 = self._time_sec
        mix = np.zeros(frames, dtype=np.float32)

        # Primary voices (with vibrato + shimmer)
        for i in range(n):
            cf, cg = cur[i]; tf, tg = tgt[i]
            if cf <= 0.0: cf = tf
            if cg < 0.0:  cg = 0.0

            nf = (1 - alpha_freq) * cf + alpha_freq * tf
            ng = (1 - alpha_gain) * self.voice_gains[i] + alpha_gain * tg
            self.voice_gains[i] = ng

            if nf > 0.0 and ng > 1e-6:
                # vibrato (frequency modulation in cents)
                if vib_depth_c > 0.0:
                    lfo_phase0 = self._voice_lfo_phase[i]
                    lfo = np.sin(2*np.pi*vib_rate_hz*(time0 + t) + lfo_phase0)
                    cents = (vib_depth_c / 1200.0) * lfo
                    nf_inst = nf * (2.0 ** cents)
                else:
                    nf_inst = nf

                # shimmer (slow amplitude wander)
                if shimmer_pct > 0.0:
                    sh = 1.0 + (shimmer_pct/100.0) * 0.5 * np.sin(2*np.pi*0.6*(time0 + t) + i*0.9)
                else:
                    sh = 1.0

                p0 = self.voice_phases[i]
                ph = p0 + 2 * np.pi * nf_inst * t
                mix += np.sin(ph, dtype=np.float32) * (ng * 0.22 * sh)
                # integrate phase at the *mean* freq for continuity
                self.voice_phases[i] = (p0 + 2*np.pi*nf*(frames/self.sample_rate)) % (2*np.pi)

            cur[i] = (nf, ng)

        # Drone (Sa + Pa)
        if drone_enabled and drone_freqs[0] > 0.0:
            for idx, f in enumerate(drone_freqs):
                if f <= 0.0: continue
                p0 = self.drone_phases[idx]
                ph = p0 + 2 * np.pi * f * t
                g = drone_gain * (0.9 if idx == 1 else 1.0)
                mix += np.sin(ph, dtype=np.float32) * g
                self.drone_phases[idx] = (p0 + 2*np.pi*f*(frames/self.sample_rate)) % (2*np.pi)

        # Sympathetic strings (plucked envelopes with smooth attack/decay)
        if strings_enabled and strings:
            decay_tau = max(0.05, string_decay_ms / 1000.0)
            attack_tau = max(0.002, string_attack_ms / 1000.0)
            # per-block smoothing factors
            blk_dt = frames / self.sample_rate
            att_alpha = 1.0 - np.exp(-blk_dt / attack_tau)
            dec_alpha = np.exp(-blk_dt / decay_tau)

            for k, s in enumerate(strings):
                f = s["freq"]; p0 = s["phase"]
                env = s["env"]; tgt_env = s["env_target"]

                # move env toward target (attack), then let target decay
                env = env + (tgt_env - env) * att_alpha
                tgt_env = tgt_env * dec_alpha

                if env > 1e-5:
                    ph = p0 + 2 * np.pi * f * t
                    # very light inherited vibrato for strings (¼ of main)
                    if vib_depth_c > 0.0:
                        cents = (0.25 * vib_depth_c / 1200.0) * np.sin(2*np.pi*(0.8*vib_rate_hz)*(time0 + t) + k*1.3)
                        f_inst = f * (2.0 ** cents)
                        ph = p0 + 2*np.pi * f_inst * t
                    mix += np.sin(ph, dtype=np.float32) * (s["base_gain"] * env)
                    p0 = (p0 + 2*np.pi*f*blk_dt) % (2*np.pi)

                # write back updated per-string state
                s["phase"] = p0
                s["env"] = float(env)
                s["env_target"] = float(tgt_env)

            # persist to shared state
            with self._lock:
                for i in range(min(len(self.strings), len(strings))):
                    self.strings[i]["phase"] = strings[i]["phase"]
                    self.strings[i]["env"] = strings[i]["env"]
                    self.strings[i]["env_target"] = strings[i]["env_target"]

        # Master gain + clip
        mix *= master_gain
        np.clip(mix, -0.98, 0.98, out=mix)
        with self._lock:
            self.current_chord = cur
        outdata[:, 0] = mix
        # advance time
        self._time_sec += frames / self.sample_rate

    # chord helper (CLEAN: single voice; sometimes octave doubler)
    major_radius = 10.0
    minor_radius = 3.0
    def generate_chord_from_note(self, root_note):
        if (root_note.metadata and root_note.metadata.get("voice") and root_note.symbol.startswith("CP")):
            return [(root_note.value, root_note.energy)], [[0, 0, 0]]
        f0 = float(root_note.value); g0 = float(root_note.energy)
        if np.random.random() < 0.25:
            chord = [(f0, g0), (f0*2.0, g0*0.35)]
            origins = [[0,0,0],[0.2,0.1,0.05]]
        else:
            chord = [(f0, g0)]
            origins = [[0,0,0]]
        return chord, origins

    def play_chord(self, chord_notes: List[Tuple[float, float]], origin_positions=None):
        with self._lock:
            changed = (self.target_chord != chord_notes)
            self.target_chord = chord_notes.copy()
            if changed and chord_notes:
                root_freq, root_gain = chord_notes[0]
                if origin_positions and len(origin_positions) >= 1 and origin_positions[0] is not None:
                    origin = origin_positions[0]
                else:
                    origin = [0.1, 0.1, 0.1]
                self.last_pulse_info = {
                    "freq": root_freq,
                    "energy": root_gain,
                    "origin": origin,
                    "timestamp": time.time(),
                    "chord": chord_notes,
                    "origins": origin_positions or [origin] * len(chord_notes),
                }

    def get_pulse_info(self):
        with self._lock:
            p = self.last_pulse_info
            self.last_pulse_info = None
            return p

    def stop(self):
        self.stream.stop()
        self.stream.close()

# -----------------------------------------------------------------------------
# Spheres / Core (routing & light transforms)
# -----------------------------------------------------------------------------
class SymbolicSphere:
    def __init__(self, radius, plane, function, capacity=100, decay_lambda=0.95, rotation_speed=0.5):
        self.radius = radius; self.plane = plane; self.function = function
        self.capacity = capacity; self.decay_lambda = decay_lambda; self.rotation_speed = rotation_speed
        self.memory_ring = deque(maxlen=capacity)
        self.phase_offset = np.random.random()*2*np.pi
        self.symbol_set = set()
        self.symbol_frequencies: Dict[str, int] = {}
        self.output_queue = deque()
        if function == SphereFunction.TEMPORAL:
            self.echo_delay = 5; self.phase_memory = deque(maxlen=20)
        elif function == SphereFunction.SEMANTIC:
            self.drift_rate = 0.1; self.symbol_embeddings = {}; self.decay_lambda = 0.98
        elif function == SphereFunction.HARMONIC:
            self.resonance_frequency = 0.2; self.feedback_strength = 0.5
            self.voice_1_history = deque(maxlen=10)
            self.voice_2_history = deque(maxlen=10)
            self.voice_motion_types = ["parallel", "contrary", "oblique", "similar"]
            self.current_motion = "contrary"
        self.transition_log = []
        self.total_residence_time = 0.0
        self.exit_count = 0

    def _pos(self, phase):
        a = phase + self.phase_offset
        if self.plane == "XY": return (self.radius*np.cos(a), self.radius*np.sin(a), 0.0)
        if self.plane == "YZ": return (0.0, self.radius*np.cos(a), self.radius*np.sin(a))
        if self.plane == "XZ": return (self.radius*np.cos(a), 0.0, self.radius*np.sin(a))
        return (0.0, 0.0, 0.0)

    def update_rotation(self):
        self.phase_offset = (self.phase_offset + self.rotation_speed) % (2*np.pi)
        for dp in self.memory_ring:
            dp.energy *= self.decay_lambda
        if self.function == SphereFunction.TEMPORAL and self.memory_ring:
            avgp = np.mean([dp.phase for dp in self.memory_ring])
            self.phase_memory.append(avgp)

    def can_accept(self, dp: DataPoint):
        s = 0.5
        if self.function == SphereFunction.TEMPORAL:
            if dp.symbol.startswith("T"): s += 0.3
            if self.memory_ring:
                d = abs(dp.phase - self.phase_memory[-1]) % (2*np.pi)
                s += 0.2 * (1 - d/np.pi)
        elif self.function == SphereFunction.SEMANTIC:
            if dp.symbol.startswith("S"): s += 0.3
            if dp.symbol in getattr(self, "symbol_embeddings", {}): s += 0.2
            elif len(self.symbol_set) < 15: s += 0.3
        else:
            if dp.spectrum is not None and len(dp.spectrum) > 0:
                peak = int(np.argmax(np.abs(dp.spectrum)))
                norm = peak / max(1, len(dp.spectrum))
                s += 0.4 * np.exp(-5 * abs(norm - self.resonance_frequency))
            else:
                s += 0.2
        if dp.symbol in self.symbol_set: s += 0.2
        elif len(self.symbol_set) < 20: s += 0.3
        return (s > 0.2, s)

    def inject(self, dp: DataPoint):
        d = dp.clone()
        d.metadata = d.metadata or {}
        d.metadata.update({
            "sphere_position": self._pos(d.phase),
            "entry_time": time.time(),
            "sphere_plane": self.plane,
            "sphere_function": self.function.value,
        })
        self.memory_ring.append(d)
        self.symbol_set.add(d.symbol)
        self.symbol_frequencies[d.symbol] = self.symbol_frequencies.get(d.symbol, 0) + 1

    def extract_ready(self):
        ready, rem = [], deque(); now = time.time()
        for d in self.memory_ring:
            exit_cond, reason = False, ""
            dt = now - d.metadata.get("entry_time", now)
            if self.function == SphereFunction.TEMPORAL and dt > 2.0:
                exit_cond, reason = True, "echo_complete"
            elif self.function == SphereFunction.SEMANTIC and (d.energy < 0.3 or d.metadata.get("drift_cycles", 0) > 5):
                exit_cond, reason = True, "drift_complete"
            elif self.function == SphereFunction.HARMONIC and dt > 1.0:
                exit_cond, reason = True, "harmonic_complete"
            if not exit_cond and d.energy < 0.05:
                exit_cond, reason = True, "energy_decay"
            if not exit_cond and len(self.memory_ring) > 10:
                exit_cond, reason = True, "sphere_full"
            if exit_cond:
                d.metadata["exit_reason"] = reason
                d.metadata["residence_time"] = dt
                self.total_residence_time += dt
                self.exit_count += 1
                ready.append(d)
            else:
                rem.append(d)
        self.memory_ring = rem
        return ready

    def apply_transformations(self):
        for d in self.memory_ring:
            d.phase = (d.phase + self.rotation_speed) % (2 * np.pi)
            if self.function == SphereFunction.HARMONIC and d.energy > 0.4 and np.random.random() < 0.6:
                for n in self._generate_counterpoint(d):
                    self.output_queue.append(n)
            elif self.function != SphereFunction.HARMONIC and d.energy > 0.5 and np.random.random() < 0.3:
                nd = d.clone()
                nd.value = max(50, d.value + np.random.uniform(-20, 20))
                nd.symbol = f"NOTE_{int(nd.value)}Hz"
                nd.energy = d.energy * np.random.uniform(0.8, 1.0)
                nd.timestamp = time.time()
                md = nd.metadata or {}; md["is_generated"] = True; md["parent_note"] = d.symbol; nd.metadata = md
                self.output_queue.append(nd)
            if self.function == SphereFunction.SEMANTIC:
                d.metadata["drift_cycles"] = d.metadata.get("drift_cycles", 0) + 1
            d.metadata["sphere_position"] = self._pos(d.phase)

    def _generate_counterpoint(self, base: DataPoint) -> List[DataPoint]:
        v1 = float(np.clip(base.value, 200, 800))
        intervals = {"perfect_4th": 4/3, "perfect_5th": 3/2, "octave": 2.0}
        k = np.random.choice(list(intervals.keys()))
        v2 = float(np.clip(v1 * intervals[k], 200, 800))
        n1 = base.clone(); n1.value=v1; n1.symbol=f"CP1-{int(v1)}Hz"; n1.energy=base.energy*0.65; n1.metadata["voice"]=1
        n2 = base.clone(); n2.value=v2; n2.symbol=f"CP2-{int(v2)}Hz"; n2.energy=base.energy*0.5; n2.phase=(n2.phase+np.pi/4)%(2*np.pi); n2.metadata["voice"]=2
        return [n1, n2]

    def get_resonant_output(self):
        out = list(self.output_queue)
        self.output_queue.clear()
        return out

class ResonanceGate:
    def __init__(self, phase_tolerance=0.5, energy_threshold=0.1):
        self.phase_tolerance = phase_tolerance; self.energy_threshold = energy_threshold
    def check_resonance(self, dp: DataPoint, sphere: SymbolicSphere):
        s = 0.0
        if dp.energy >= self.energy_threshold: s += 0.3*(dp.energy/1.0)
        else: return (False, 0.0)
        pd = abs(dp.phase - sphere.phase_offset) % (2*np.pi); al = min(pd, 2*np.pi-pd)
        s += 0.3*(1 - al/self.phase_tolerance) if al <= self.phase_tolerance else 0.1
        ok, sym = sphere.can_accept(dp)
        s += 0.4*sym if ok else 0.2
        return (s > 0.2, s)

class TriHarmonicCore:
    def __init__(self, major_radius=10.0, minor_radius=3.0, enable_s2=True, enable_s3=True):
        self.major_radius = major_radius; self.minor_radius = minor_radius
        r = minor_radius * 0.8
        self.sphere_s1 = SymbolicSphere(r, "XY", SphereFunction.TEMPORAL, rotation_speed=0.5)
        self.sphere_s2 = SymbolicSphere(r, "YZ", SphereFunction.SEMANTIC, rotation_speed=0.35, decay_lambda=0.98) if enable_s2 else None
        self.sphere_s3 = SymbolicSphere(r, "XZ", SphereFunction.HARMONIC, rotation_speed=0.75, decay_lambda=0.92) if enable_s3 else None
        self.spheres = [self.sphere_s1] + ([self.sphere_s2] if self.sphere_s2 else []) + ([self.sphere_s3] if self.sphere_s3 else [])
        self.gate = ResonanceGate()
        self.torus_buffer = deque(maxlen=1000)
        self.metrics = {"total_processed":0,"s1_entries":0,"s1_exits":0,"s2_entries":0,"s2_exits":0,"s3_entries":0,"s3_exits":0,
                        "torus_direct":0,"total_resonance_checks":0,"avg_resonance_score":0.0}
        self.resonance_scores: List[float] = []

    def process(self, dp: DataPoint):
        self.metrics["total_processed"] += 1
        best, best_sc = None, 0.0
        for s in self.spheres:
            ok, sc = self.gate.check_resonance(dp, s)
            self.metrics["total_resonance_checks"] += 1
            self.resonance_scores.append(sc)
            if ok and sc > best_sc:
                best, best_sc = s, sc
        if best:
            best.inject(dp)
            key = {self.sphere_s1:"s1_entries", self.sphere_s2:"s2_entries", self.sphere_s3:"s3_entries"}[best]
            self.metrics[key] += 1
        else:
            self.torus_buffer.append(dp); self.metrics["torus_direct"] += 1

        for s in self.spheres:
            s.update_rotation(); s.apply_transformations()
            for ed in s.extract_ready():
                ed.metadata["sphere_processed"] = True
                self.torus_buffer.append(ed)
                k = {self.sphere_s1:"s1_exits", self.sphere_s2:"s2_exits", self.sphere_s3:"s3_exits"}[s]
                self.metrics[k] += 1

        if self.resonance_scores:
            self.metrics["avg_resonance_score"] = float(np.mean(self.resonance_scores[-100:]))

    def get_state_summary(self) -> Dict[str, Any]:
        out = {"torus_size":len(self.torus_buffer),
               "s1_size":len(self.sphere_s1.memory_ring),
               "s1_phase":self.sphere_s1.phase_offset,
               "metrics":self.metrics}
        if self.sphere_s2:
            out["s2_size"] = len(self.sphere_s2.memory_ring); out["s2_phase"] = self.sphere_s2.phase_offset
        if self.sphere_s3:
            out["s3_size"] = len(self.sphere_s3.memory_ring); out["s3_phase"] = self.sphere_s3.phase_offset
        return out

class MusicCore(TriHarmonicCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio = ContinuousAudioPlayer()
        self.pulse_events = deque(maxlen=12)
        self.viz_pulse_callback = None
        self.last_notes = deque(maxlen=24)
        self._last_viz_pulse_ts = 0.0

    def set_pulse_callback(self, cb): self.viz_pulse_callback = cb

    def check_for_pulses(self):
        if not hasattr(self.audio, "get_pulse_info"):
            return
        p = self.audio.get_pulse_info()
        if not p:
            return
        if (time.time() - getattr(self, "_last_viz_pulse_ts", 0.0)) < 0.03:
            return
        if self.viz_pulse_callback:
            try: self.viz_pulse_callback(p["freq"], p["energy"], p["origin"])
            except Exception: pass
        self.pulse_events.append({"timestamp":p["timestamp"],"frequency":p["freq"],"energy":p["energy"],"intensity":1.0,"origin":p["origin"]})

    def get_active_pulses(self):
        now = time.time()
        out = []
        for e in self.pulse_events:
            age = now - e["timestamp"]
            if age < 2.0:
                d = e.copy()
                d["intensity"] = max(0.0, 1.0 - age/2.0)
                out.append(d)
        return out

    def get_recent_notes(self):
        return list(self.last_notes)

    # --- helper: always compute a concrete torus origin ---
    def _torus_pos(self, phase: float, energy: float):
        R, rr = self.major_radius, self.minor_radius * 0.9
        u = phase
        v = energy * 2 * np.pi
        x = (R + rr * np.cos(v)) * np.cos(u)
        y = (R + rr * np.cos(v)) * np.sin(u)
        z = rr * np.sin(v)
        return [float(x), float(y), float(z)]

    def process_and_play(self, note: DataPoint):
        self.process(note)
        played = 0
        for s in self.spheres:
            for g in s.get_resonant_output():
                self._play(g); played += 1
        for s in self.spheres:
            for e in s.extract_ready():
                self._play(e); played += 1
        if played == 0 and self.torus_buffer and np.random.random() < 0.3:
            self._play(list(self.torus_buffer)[-1])

    def _play(self, note: DataPoint):
        root_origin = None
        if note.metadata:
            root_origin = note.metadata.get("sphere_position") or note.metadata.get("current_position")
        if root_origin is None:
            root_origin = self._torus_pos(note.phase, note.energy)

        chord, origins = self.audio.generate_chord_from_note(note)
        if not origins or not origins[0] or (isinstance(origins[0], (list, tuple)) and np.allclose(origins[0], [0,0,0])):
            origins = origins or []
            if len(origins) < len(chord):
                origins += [[0,0,0]] * (len(chord) - len(origins))
            origins[0] = list(root_origin)

        if self.viz_pulse_callback and chord:
            try:
                self.viz_pulse_callback(chord[0][0], chord[0][1], origins[0])
                self._last_viz_pulse_ts = time.time()
            except Exception:
                pass

        self.audio.play_chord(chord, origins)

        if getattr(self, "_pluck_density", 0.0) > 0.0 and getattr(self, "_strings_enabled", True):
            if np.random.random() < self._pluck_density:
                choices = [0,1,2]; np.random.shuffle(choices)
                n = 1 + (np.random.random() < 0.4)
                idxs = choices[:int(n)]
                self.audio.pluck_strings(idxs, velocity=0.8 + 0.2*np.random.random())

        root_f = chord[0][0] if chord else note.value
        sw = (note.metadata or {}).get("swara")
        octv = (note.metadata or {}).get("oct")
        lab = (f"{sw}{'+' if octv>0 else '-' if octv<0 else ''}" if (sw is not None and isinstance(octv, int) and octv != 0)
               else (sw if sw is not None else note.symbol))
        self.last_notes.append({"freq": float(root_f), "label": lab})

    def stop(self): self.audio.stop()

# -----------------------------------------------------------------------------
# Sa-anchored generator with simple 8-beat cycle (no range clamping)
# -----------------------------------------------------------------------------
@dataclass
class AnchorSettings:
    anchor_bias: float = 0.7  # 0..1 probability to favor Sa
    cycle_len: int = 8

def generate_musical_stream(pitchsys: PitchSystem, anchor: AnchorSettings):
    t = 0.0
    rng = np.random.default_rng()
    motif_collect: List[DataPoint] = []
    phrase_queue: deque = deque()
    spacing = 0.35
    beat_idx = 0

    def mk_dp(freq: float, label: str, energy: float, dur: float, phase: float, sw=None, octv=0):
        spec = np.array([1.0, 0.3, 0.1, 0.05], dtype=float) * energy
        md = {"swara": sw, "oct": octv} if sw is not None else {}
        return DataPoint(value=freq, timestamp=t, symbol=label, phase=phase,
                         spectrum=spec, energy=energy, duration=dur, metadata=md)

    def pick_swara():
        nonlocal beat_idx
        swaras = pitchsys.allowed_swaras
        on_sam = (beat_idx % anchor.cycle_len == 0)
        on_half = (beat_idx % anchor.cycle_len == anchor.cycle_len // 2)
        bias = anchor.anchor_bias
        weights = []
        for sw in swaras:
            if sw == "S":
                w = 0.15 + 0.75 * bias + (0.10 if (on_sam or on_half) else 0.0)
            elif sw == "P":
                w = 0.12 + 0.18 * bias
            else:
                w = 1.0
            weights.append(w)
        weights = np.array(weights, dtype=float)
        weights = np.maximum(weights, 1e-6)
        weights /= weights.sum()
        return rng.choice(swaras, p=weights)

    while True:
        if phrase_queue:
            n = phrase_queue.popleft()
            yield n; t += n.duration * spacing
            beat_idx = (beat_idx + 1) % max(1, anchor.cycle_len)
            continue

        sw = pick_swara()
        octv = 0
        if rng.random() < 0.10 and sw != "S":
            octv = rng.choice([-1, 1])
        f = pitchsys.swara_to_hz(sw, octv)

        phase = rng.random() * 2*np.pi if rng.random() < 0.35 else (len(motif_collect) / 12.0) % (2*np.pi)
        energy = rng.uniform(0.55, 0.8)
        dur = rng.uniform(0.22, 0.45)

        raw = mk_dp(f, f"{sw}{'+' if octv>0 else '-' if octv<0 else ''}-{int(f)}Hz", energy, dur, phase, sw=sw, octv=octv)
        motif_collect.append(raw)

        if len(motif_collect) >= 2:
            last_two = motif_collect[-2:]
            phrase = []
            start_on_sa = (rng.random() < (0.6 * anchor.anchor_bias)) or (beat_idx % anchor.cycle_len == 0)
            if start_on_sa:
                fS = pitchsys.swara_to_hz("S", 0)
                phrase.append(mk_dp(fS, f"S-{int(fS)}Hz", energy*0.95, dur*0.95, phase, sw="S", octv=0))
            asc = rng.random() < 0.5
            seq = sorted(last_two, key=lambda d: d.value, reverse=not asc)
            phrase.extend([seq[0].clone(), seq[1].clone()])
            if all((n.metadata or {}).get("swara") != "S" for n in seq) and rng.random() < (0.7 * anchor.anchor_bias):
                fS = pitchsys.swara_to_hz("S", 0)
                phrase.insert(1, mk_dp(fS, f"S-{int(fS)}Hz", energy*0.9, dur*0.9, phase, sw="S", octv=0))
            if rng.random() < (0.75 * anchor.anchor_bias):
                fS = pitchsys.swara_to_hz("S", 0)
                phrase.append(mk_dp(fS, f"S-{int(fS)}Hz", energy*0.85, dur*0.85, phase, sw="S", octv=0))
            for k, n in enumerate(phrase):
                n.energy = float(np.clip(n.energy * (0.97 ** k), 0.05, 1.0))
                n.duration = float(np.clip(n.duration * (0.90 + 0.10*np.random.rand()), 0.12, 0.6))
                phrase_queue.append(n)
            n0 = phrase_queue.popleft()
            yield n0; t += n0.duration * spacing
            beat_idx = (beat_idx + 1) % max(1, anchor.cycle_len)
        else:
            yield raw; t += raw.duration * 0.22
            beat_idx = (beat_idx + 1) % max(1, anchor.cycle_len)

# -----------------------------------------------------------------------------
# Visualizer (Qt timer; Controls; Stats)
# -----------------------------------------------------------------------------
class TriHarmonicVispyVisualizer:
    def __init__(self, core: MusicCore):
        self.core = core
        self.sim_frame = 0
        self.vis_frame = 0
        # Wider range: very slow to fast
        self.speed_steps = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        self.speed_factor = 1.0

        # defaults for pluck behavior (used by MusicCore._play)
        self.core._pluck_density = 0.25
        self.core._strings_enabled = True

        self.qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("Tri-Harmonic (Sursringar Pitch System)")

        root = QtWidgets.QHBoxLayout(self.win)
        root.setContentsMargins(8,8,8,8)
        root.setSpacing(8)

        # LEFT: VisPy canvas
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)
        root.addLayout(left, 1)

        self.canvas = scene.SceneCanvas(title="TRI-HARMONIC MUSIC SYNTHESIS",
                                        size=(1400, 1000), bgcolor="black")
        left.addWidget(self.canvas.native, stretch=1)

        grid = self.canvas.central_widget.add_grid()
        self.view3d = grid.add_view(row=0, col=0, camera="turntable")
        self.view3d.camera.distance = 40
        self.view3d.camera.elevation = 30
        self.view3d.camera.azimuth = 45
        self.view3d.camera.fov = 60

        # RIGHT: Controls + Stats
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        root.addLayout(right, 0)

        # Speed
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Speed"))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(0, len(self.speed_steps)-1)
        self.speed_slider.setValue(self.speed_steps.index(1.0))
        self.speed_slider.valueChanged.connect(self._on_speed)
        self.speed_val_label = QtWidgets.QLabel("1.0×")
        row1.addWidget(self.speed_slider, 1); row1.addWidget(self.speed_val_label)
        right.addLayout(row1)

        # Volume (Master)
        row_vol = QtWidgets.QHBoxLayout()
        row_vol.addWidget(QtWidgets.QLabel("Master Volume"))
        self.vol_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.valueChanged.connect(self._on_volume)
        self.vol_label = QtWidgets.QLabel("80%")
        row_vol.addWidget(self.vol_slider, 1); row_vol.addWidget(self.vol_label)
        right.addLayout(row_vol)

        # Vibrato / shimmer
        vib_grid = QtWidgets.QGridLayout()
        vib_grid.addWidget(QtWidgets.QLabel("Vibrato Depth (cents)"), 0, 0)
        self.vib_depth = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.vib_depth.setRange(0, 50); self.vib_depth.setValue(4)
        vib_grid.addWidget(self.vib_depth, 0, 1)
        self.vib_depth_lbl = QtWidgets.QLabel("4")
        vib_grid.addWidget(self.vib_depth_lbl, 0, 2)

        vib_grid.addWidget(QtWidgets.QLabel("Vibrato Rate (Hz)"), 1, 0)
        self.vib_rate = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.vib_rate.setRange(1, 120)  # 0.1..12.0 scaled by 10
        self.vib_rate.setValue(55)  # 5.5 Hz
        vib_grid.addWidget(self.vib_rate, 1, 1)
        self.vib_rate_lbl = QtWidgets.QLabel("5.5")
        vib_grid.addWidget(self.vib_rate_lbl, 1, 2)

        vib_grid.addWidget(QtWidgets.QLabel("Shimmer (%)"), 2, 0)
        self.shimmer = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.shimmer.setRange(0, 200); self.shimmer.setValue(20)
        vib_grid.addWidget(self.shimmer, 2, 1)
        self.shimmer_lbl = QtWidgets.QLabel("2.0")
        vib_grid.addWidget(self.shimmer_lbl, 2, 2)

        right.addLayout(vib_grid)

        # Rāga / Tonic / Meend / Sa Anchor
        row3 = QtWidgets.QGridLayout()
        row3.addWidget(QtWidgets.QLabel("Rāga"), 0, 0)
        self.raga_combo = QtWidgets.QComboBox()
        for name in PitchSystem.RAGAS.keys():
            self.raga_combo.addItem(name)
        row3.addWidget(self.raga_combo, 0, 1, 1, 3)

        row3.addWidget(QtWidgets.QLabel("Tonic (Sa) Hz"), 1, 0)
        self.tonic_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.tonic_slider.setRange(100, 440)
        self.tonic_slider.setValue(220)
        self.tonic_label = QtWidgets.QLabel("220 Hz")
        row3.addWidget(self.tonic_slider, 1, 1, 1, 2)
        row3.addWidget(self.tonic_label, 1, 3)

        self.meend_check = QtWidgets.QCheckBox("Meend (glide)")
        self.meend_check.setChecked(False)
        row3.addWidget(self.meend_check, 2, 0)

        row3.addWidget(QtWidgets.QLabel("Glide (ms)"), 2, 1)
        self.glide_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.glide_slider.setRange(5, 600)
        self.glide_slider.setValue(60)
        self.glide_label = QtWidgets.QLabel("60 ms")
        row3.addWidget(self.glide_slider, 2, 2)
        row3.addWidget(self.glide_label, 2, 3)

        row3.addWidget(QtWidgets.QLabel("Sa Anchor (%)"), 3, 0)
        self.anchor_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.anchor_slider.setRange(0, 100)
        self.anchor_slider.setValue(70)
        self.anchor_label = QtWidgets.QLabel("70%")
        row3.addWidget(self.anchor_slider, 3, 1, 1, 2)
        row3.addWidget(self.anchor_label, 3, 3)

        # Tuning preset + 12-TET override
        row3.addWidget(QtWidgets.QLabel("Tuning Preset"), 4, 0)
        self.preset_combo = QtWidgets.QComboBox()
        for p in TuningPreset:
            self.preset_combo.addItem(p.value)
        row3.addWidget(self.preset_combo, 4, 1, 1, 3)

        self.tet_override_check = QtWidgets.QCheckBox("Force 12-TET (A/B)")
        row3.addWidget(self.tet_override_check, 5, 0, 1, 4)

        # Sympathetic strings
        row4 = QtWidgets.QGridLayout()
        self.strings_check = QtWidgets.QCheckBox("Sympathetic Strings")
        self.strings_check.setChecked(True)
        row4.addWidget(self.strings_check, 0, 0, 1, 2)

        row4.addWidget(QtWidgets.QLabel("Pluck Density"), 1, 0)
        self.pluck_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pluck_slider.setRange(0, 100)
        self.pluck_slider.setValue(25)
        self.pluck_label = QtWidgets.QLabel("25%")
        row4.addWidget(self.pluck_slider, 1, 1)

        row4.addWidget(QtWidgets.QLabel("String Decay (ms)"), 2, 0)
        self.decay_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.decay_slider.setRange(200, 4000)
        self.decay_slider.setValue(2400)
        self.decay_label = QtWidgets.QLabel("2400 ms")
        row4.addWidget(self.decay_slider, 2, 1)

        row4.addWidget(QtWidgets.QLabel("String Attack (ms)"), 3, 0)
        self.attack_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.attack_slider.setRange(1, 200)
        self.attack_slider.setValue(18)
        self.attack_label = QtWidgets.QLabel("18 ms")
        row4.addWidget(self.attack_slider, 3, 1)

        right.addLayout(row3)
        right.addLayout(row4)

        # Stats panel
        self.stats_box = QtWidgets.QPlainTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setMinimumWidth(360)
        self.stats_box.setMinimumHeight(460)
        fm = self.stats_box.font()
        fm.setFamily("Consolas" if sys.platform.startswith("win") else "Monospace")
        fm.setPointSize(9)
        self.stats_box.setFont(fm)
        right.addWidget(self.stats_box, 1)

        # Geometry
        self._create_torus_wireframe(); self._create_sphere_wireframes()

        # Particles
        self.torus_particles = scene.visuals.Markers(parent=self.view3d.scene)
        self.s1_particles = scene.visuals.Markers(parent=self.view3d.scene)
        self.s2_particles = scene.visuals.Markers(parent=self.view3d.scene)
        self.s3_particles = scene.visuals.Markers(parent=self.view3d.scene)
        for p in (self.torus_particles, self.s1_particles, self.s2_particles, self.s3_particles):
            p.set_gl_state("translucent", depth_test=True)

        self.origin_marker = scene.visuals.Markers(parent=self.view3d.scene)
        self.origin_marker.set_data(pos=np.array([[0,0,0]]), face_color="white", edge_color="yellow", edge_width=2, size=20, symbol="disc")
        self.x_axis = scene.visuals.XYZAxis(parent=self.view3d.scene, width=2)

        # Pulse visuals
        self.pulse_rings: List[Dict[str, Any]] = []; self.max_pulse_rings = 5

        # Pitch system + generator + anchor settings
        self.pitchsys = PitchSystem(
            tonic_hz=float(self.tonic_slider.value()),
            raga_name=self.raga_combo.currentText(),
            preset=TuningPreset.JI_5_LIMIT,
            force_12tet=False
        )
        self.anchor = AnchorSettings(anchor_bias=self.anchor_slider.value()/100.0, cycle_len=8)
        self.data_stream_gen = generate_musical_stream(self.pitchsys, self.anchor)

        # Hook pulse + UI signals
        self.core.set_pulse_callback(self.on_pulse)
        self.raga_combo.currentTextChanged.connect(self._on_raga_changed)
        self.tonic_slider.valueChanged.connect(self._on_tonic_changed)
        self.meend_check.stateChanged.connect(self._on_meend_changed)
        self.glide_slider.valueChanged.connect(self._on_glide_changed)
        self.anchor_slider.valueChanged.connect(self._on_anchor_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.tet_override_check.stateChanged.connect(self._on_tet_override)

        # Master volume & strings signals
        self.vol_slider.valueChanged.connect(self._on_volume)
        self.strings_check.stateChanged.connect(self._on_strings_toggle)
        self.pluck_slider.valueChanged.connect(self._on_pluck_changed)
        self.decay_slider.valueChanged.connect(self._on_decay_changed)
        self.attack_slider.valueChanged.connect(self._on_attack_changed)

        # Vibrato/shimmer signals
        self.vib_depth.valueChanged.connect(self._on_vib_changed)
        self.vib_rate.valueChanged.connect(self._on_vib_changed)
        self.shimmer.valueChanged.connect(self._on_vib_changed)

        self._apply_meend()
        self._apply_vibrato()

        # Prime drone & strings
        self.core.audio.set_drone_from_tonic(self.pitchsys.tonic_hz)
        self.core.audio.configure_strings(self.pitchsys.tonic_hz)

        # Qt timer
        self.qtimer = QtCore.QTimer()
        self.qtimer.setInterval(50)  # ~20 FPS
        self.qtimer.timeout.connect(lambda: self._tick())
        self.qtimer.start()

        # Kick one note immediately
        try:
            n0 = next(self.data_stream_gen)
            self.core.process_and_play(n0)
        except Exception as e:
            print(f"[engine prime] {e}")
        print("[engine] Qt timer started")

    # UI handlers
    def _on_speed(self, idx: int):
        idx = max(0, min(idx, len(self.speed_steps)-1))
        self.speed_factor = self.speed_steps[idx]
        self.speed_val_label.setText(f"{self.speed_factor:.2f}×")

    def _on_volume(self, v: int):
        self.vol_label.setText(f"{v}%")
        self.core.audio.set_master_gain(v/100.0)

    def _on_strings_toggle(self, _):
        on = self.strings_check.isChecked()
        self.core._strings_enabled = bool(on)
        self.core.audio.strings_enabled = bool(on)

    def _on_pluck_changed(self, v: int):
        self.pluck_label.setText(f"{v}%")
        self.core._pluck_density = float(v)/100.0

    def _on_decay_changed(self, v: int):
        self.decay_label.setText(f"{v} ms")
        self.core.audio.string_decay_ms = float(v)

    def _on_attack_changed(self, v: int):
        self.attack_label.setText(f"{v} ms")
        self.core.audio.set_string_attack(float(v))

    def _on_vib_changed(self, _):
        depth = float(self.vib_depth.value())
        rate = float(self.vib_rate.value())/10.0
        shim = float(self.shimmer.value())/10.0
        self.vib_depth_lbl.setText(f"{depth:.0f}")
        self.vib_rate_lbl.setText(f"{rate:.1f}")
        self.shimmer_lbl.setText(f"{shim:.1f}")
        self.core.audio.set_vibrato(depth, rate, shim)

    def _on_raga_changed(self, name: str):
        self.pitchsys.set_raga(name)

    def _on_tonic_changed(self, v: int):
        self.pitchsys.set_tonic(float(v))
        self.tonic_label.setText(f"{v} Hz")
        self.core.audio.set_drone_from_tonic(self.pitchsys.tonic_hz)
        self.core.audio.configure_strings(self.pitchsys.tonic_hz)

    def _apply_meend(self):
        self.core.audio.set_meend(self.meend_check.isChecked(), float(self.glide_slider.value()))
    # inside class TriHarmonicVispyVisualizer, e.g. near other _apply_* methods
    def _apply_vibrato(self):
        depth = float(self.vib_depth.value())
        rate  = float(self.vib_rate.value()) / 10.0   # slider is 1..120 -> 0.1..12.0 Hz
        shim  = float(self.shimmer.value()) / 10.0    # slider is 0..200 -> 0..20.0 %
        self.vib_depth_lbl.setText(f"{depth:.0f}")
        self.vib_rate_lbl.setText(f"{rate:.1f}")
        self.shimmer_lbl.setText(f"{shim:.1f}")
        self.core.audio.set_vibrato(depth, rate, shim)

    def _on_meend_changed(self, _):
        self._apply_meend()

    def _on_glide_changed(self, v: int):
        self.glide_label.setText(f"{v} ms")
        self._apply_meend()

    def _on_anchor_changed(self, v: int):
        self.anchor.anchor_bias = float(v)/100.0
        self.anchor_label.setText(f"{v}%")

    def _on_preset_changed(self, _):
        text = self.preset_combo.currentText()
        for p in TuningPreset:
            if p.value == text:
                self.pitchsys.set_preset(p); break

    def _on_tet_override(self, _):
        self.pitchsys.set_force_12tet(self.tet_override_check.isChecked())

    # Timer tick
    def _tick(self):
        try:
            self.core.check_for_pulses()
            # With small speed_factor, emit notes less often
            step = max(1, int(1 / max(1e-3, self.speed_factor)))
            if self.sim_frame % step == 0:
                self.core.process_and_play(next(self.data_stream_gen))
            self.sim_frame += 1; self.vis_frame += 1
            self._update_pulse_rings(); self._update_particles(); self._update_stats_panel(self.core.get_state_summary())
        except StopIteration:
            self.data_stream_gen = generate_musical_stream(self.pitchsys, self.anchor)
        except Exception as e:
            print(f"[tick error] {e}")

    # Pulse visuals
    def on_pulse(self, freq, energy, origin_position):
        try:
            if len(self.pulse_rings) >= self.max_pulse_rings:
                old = self.pulse_rings.pop(0)
                if old.get("visual") and old["visual"].parent: old["visual"].parent = None
            origin = list(origin_position)
            dist = float(np.linalg.norm(origin))
            if dist > 15.0:
                s = 5.0 / dist; origin = [origin[0]*s, origin[1]*s, origin[2]*s]
            theta = np.linspace(0, 2*np.pi, 30); r0 = 2.0
            f_norm = min(1.0, max(0.0, (freq - 300.0)/300.0))
            col = (1.0, 1.0-f_norm, f_norm, 1.0)
            rx = origin[0] + r0*np.cos(theta); ry = origin[1] + r0*np.sin(theta); rz = origin[2] + 0*theta
            vis = scene.visuals.Line(np.column_stack([rx, ry, rz]), color=col, parent=self.view3d.scene, width=5)
            self.pulse_rings.append({"visual":vis,"start_time":time.time(),"color":col,"origin":origin,"radius":r0})
        except Exception as e:
            print(f"[pulse create] {e}")

    # Geometry / HUD
    def _create_torus_wireframe(self):
        R = self.core.major_radius; r = self.core.minor_radius
        u = np.linspace(0, 2*np.pi, 30); v = np.linspace(0, 2*np.pi, 20)
        for i in range(len(u)):
            x = (R + r*np.cos(v))*np.cos(u[i]); y = (R + r*np.cos(v))*np.sin(u[i]); z = r*np.sin(v)
            scene.visuals.Line(np.column_stack([x,y,z]), color=(0.5,0.5,0.5,0.2), parent=self.view3d.scene, width=0.5)
        for j in range(0, len(v), 2):
            x = (R + r*np.cos(v[j]))*np.cos(u); y = (R + r*np.cos(v[j]))*np.sin(u); z = r*np.sin(v[j])*np.ones_like(u)
            scene.visuals.Line(np.column_stack([x,y,z]), color=(0.5,0.5,0.5,0.2), parent=self.view3d.scene, width=0.5)

    def _create_sphere_wireframes(self):
        th = np.linspace(0, 2*np.pi, 20); r = self.core.sphere_s1.radius
        scene.visuals.Line(np.column_stack([r*np.cos(th), r*np.sin(th), 0*th]), color="orange", parent=self.view3d.scene, width=2)
        if self.core.sphere_s2:
            scene.visuals.Line(np.column_stack([0*th, r*np.cos(th), r*np.sin(th)]), color="cyan", parent=self.view3d.scene, width=2)
        if self.core.sphere_s3:
            scene.visuals.Line(np.column_stack([r*np.cos(th), 0*th, r*np.sin(th)]), color="lime", parent=self.view3d.scene, width=2)

    def _update_pulse_rings(self):
        if not self.pulse_rings: return
        now = time.time(); alive=[]
        for ring in self.pulse_rings:
            try:
                age = now - ring["start_time"]; max_age = 3.0
                if age < max_age:
                    prog = age/max_age; radius = ring["radius"] + prog*20.0; alpha = (1.0-prog)*0.9
                    th = np.linspace(0, 2*np.pi, 30); o = ring["origin"]
                    rx = o[0] + radius*np.cos(th); ry = o[1] + radius*np.sin(th); rz = o[2] + 0*th
                    ring["visual"].set_data(np.column_stack([rx,ry,rz]), color=(*ring["color"][:3], alpha))
                    alive.append(ring)
                else:
                    if ring.get("visual") and ring["visual"].parent: ring["visual"].parent = None
            except Exception:
                if ring.get("visual") and ring["visual"].parent: ring["visual"].parent = None
        self.pulse_rings = alive

    def _update_particles(self):
        if self.core.torus_buffer:
            n = min(30, len(self.core.torus_buffer)); samples = list(self.core.torus_buffer)[-n:]
            pos=[]
            for d in samples:
                u = d.phase + self.vis_frame*0.02*self.speed_factor; v = d.energy*2*np.pi
                R, rr = self.core.major_radius, self.core.minor_radius*0.9
                x=(R+rr*np.cos(v))*np.cos(u); y=(R+rr*np.cos(v))*np.sin(u); z=rr*np.sin(v)
                pos.append([x,y,z])
            self.torus_particles.set_data(np.array(pos), face_color="yellow", edge_color="white", edge_width=0.5, size=10)
        else:
            self.torus_particles.set_data(pos=np.zeros((0,3)))

        self._update_sphere_pts(self.core.sphere_s1, self.s1_particles, "orange")
        if self.core.sphere_s2: self._update_sphere_pts(self.core.sphere_s2, self.s2_particles, "cyan")
        if self.core.sphere_s3: self._update_sphere_pts(self.core.sphere_s3, self.s3_particles, "lime")

    def _update_sphere_pts(self, sphere, particles, face_col: str):
        pts=[]
        for d in sphere.memory_ring:
            if "sphere_position" in (d.metadata or {}): pts.append(d.metadata["sphere_position"])
        if pts:
            particles.set_data(np.array(pts), face_color=face_col, edge_color="white", edge_width=0.5, size=15, symbol="star")
        else:
            particles.set_data(pos=np.zeros((0,3)))

    def _update_stats_panel(self, state: Dict[str, Any]):
        s = "SYSTEM METRICS\n\n"
        s += "━━━ BUFFER STATE ━━━\n"
        s += f"Torus: {state['torus_size']} items\n"
        s += f"S₁ (XY): {state['s1_size']} items\n"
        if "s2_size" in state: s += f"S₂ (YZ): {state['s2_size']} items\n"
        if "s3_size" in state: s += f"S₃ (XZ): {state['s3_size']} items\n"

        pulses = self.core.get_active_pulses() if hasattr(self.core, "get_active_pulses") else []
        s += "\n━━━ AUDIO PULSES ━━━\n"
        s += f"Active pulses: {len(pulses)}\n"
        if pulses:
            lp = max(pulses, key=lambda p: p["timestamp"])
            s += f"Latest: {lp['frequency']:.1f}Hz\n"
            s += f"Intensity: {lp['intensity']:.2f}\n"

        s += "\n━━━ ROUTING ━━━\n"
        total = state["metrics"]["total_processed"]
        if total > 0:
            s1p = state["metrics"]["s1_entries"]/total*100
            s += f"S₁ entries: {state['metrics']['s1_entries']} ({s1p:.1f}%)\n"
            if "s2_size" in state:
                s2p = state["metrics"]["s2_entries"]/total*100; s += f"S₂ entries: {state['metrics']['s2_entries']} ({s2p:.1f}%)\n"
            if "s3_size" in state:
                s3p = state["metrics"]["s3_entries"]/total*100; s += f"S₃ entries: {state['metrics']['s3_entries']} ({s3p:.1f}%)\n"
            dpct = state["metrics"]["torus_direct"]/total*100; s += f"Direct: {state['metrics']['torus_direct']} ({dpct:.1f}%)\n"

        s += "\n━━━ PERFORMANCE ━━━\n"
        eff = 100 - (state["metrics"]["torus_direct"]/total*100) if total>0 else 0
        s += f"Routing Efficiency: {eff:.1f}%\n"
        s += f"Resonance Score: {state['metrics']['avg_resonance_score']:.3f}\n"

        # Rāga / tuning header
        s += "\n━━━ RĀGA & TUNING ━━━\n"
        s += f"Rāga:  {self.pitchsys.raga_name}\n"
        s += f"Tonic: {self.pitchsys.tonic_hz:.1f} Hz\n"
        s += f"Tuning: {self.preset_combo.currentText()}\n"
        s += f"12-TET Override: {'ON' if self.tet_override_check.isChecked() else 'OFF'}\n"
        s += f"Sa Anchor: {int(self.anchor.anchor_bias*100)}%  | Cycle: {self.anchor.cycle_len}\n"

        # Strings status
        s += "\n━━━ STRINGS ━━━\n"
        s += f"Enabled: {'ON' if self.strings_check.isChecked() else 'OFF'}\n"
        s += f"Pluck Density: {self.pluck_slider.value()}% | Decay: {self.decay_slider.value()} ms | Attack: {self.attack_slider.value()} ms\n"
        s += f"Master Vol: {self.vol_slider.value()}%\n"

        # Vibrato readout
        s += "\n━━━ WAVERING ━━━\n"
        s += f"Vibrato: {self.vib_depth_lbl.text()} cents @ {self.vib_rate_lbl.text()} Hz | Shimmer: {self.shimmer_lbl.text()} %\n"

        # Recent notes
        if hasattr(self.core, "get_recent_notes"):
            recent = self.core.get_recent_notes()[-12:]
            if recent:
                s += "\n━━━ RECENT NOTES ━━━\n"
                for i, n in enumerate(recent, 1):
                    s += f"{i:>2}. {n['label']:<4}  {n['freq']:.1f} Hz\n"

        s += f"\nSim Frame: {self.sim_frame} | Vis Frame: {self.vis_frame}"
        self.stats_box.setPlainText(s)

    def run(self):
        self.canvas.show(); self.win.resize(1680, 1080); self.win.show()
        try:
            self.qt_app.exec_()
        finally:
            self.core.stop()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("\n" + "="*76)
    print("  TRI-HARMONIC MUSIC (Sursringar • Pulse Fix • Master Vol • Multi-Strings • Soft Attack • Wavering)")
    print("="*76)
    core = MusicCore(enable_s2=True, enable_s3=True)
    viz = TriHarmonicVispyVisualizer(core)
    viz.run()

if __name__ == "__main__":
    main()
