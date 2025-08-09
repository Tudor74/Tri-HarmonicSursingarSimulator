Tri-Harmonic Music — Sursringar Pitch System
A real-time generative music engine and visualizer that fuses North Indian classical rāga structures with a Tri-Harmonic toroidal routing core and a phase-continuous synthesis engine.
Implements multi-sphere symbolic processing for note routing, Sa-anchored motif generation, and immersive 3D visualization with PyQt5 and VisPy.

Features
🎼 Musical System
Rāga-constrained generation — Sa-anchored motifs with configurable anchor bias & cycle length

Multiple tuning systems — 5-limit Just Intonation, Pythagorean, 22-Shruti (approx.), 12-TET override

Pulse-origin fix — stable spatial rendering of audio pulses in 3D scene

Multi-string sympathetic resonance — independent attack/decay envelopes per string with detuning

Soft pluck attack — stateful per-string envelope smoothing to remove clicks

Drone support — Sa & Pa drones with adjustable gain

Wavering controls — vibrato depth/rate and amplitude shimmer

🔊 Audio Engine
Phase-continuous pure sine synthesis with vibrato & shimmer

Meend (glide) with adjustable time constants

Configurable master volume

Real-time multi-voice chord synthesis

Click suppression with conditional phase resets

Sympathetic strings with random detune and smooth envelopes

🌀 Tri-Harmonic Core
Three symbolic spheres:

S₁ (Temporal Memory) — phase-coherent echo memory

S₂ (Semantic Drift) — symbol-driven transformation

S₃ (Harmonic Feedback) — counterpoint generation

Resonance gate for note-to-sphere routing based on phase, energy, and symbolic match

Toroidal buffer for direct-routed notes

Detailed per-sphere metrics and routing efficiency tracking

🎨 Visualization
3D torus wireframe and sphere overlays in VisPy

Real-time particle trails for active notes in torus and spheres

Expanding pulse rings for triggered notes

Live HUD with:

Buffer sizes, routing counts, efficiency

Rāga, tonic, tuning preset

Vibrato/shimmer settings

Sympathetic string parameters

Recent notes list

🎛 Interactive UI (PyQt5)
Speed control (0.02× to 5×)

Master volume slider

Rāga selection, tonic frequency slider

Meend enable & glide time

Sa anchor bias slider

Tuning preset selection + 12-TET override

Vibrato depth/rate and shimmer amount

Sympathetic strings toggle, pluck density, attack/decay sliders

Live metrics panel

Requirements
Python 3.8+

PyQt5

VisPy

sounddevice

All required packages auto-install on first run.

Running
bash
Copy
Edit
python tri_harmonic_sursringar.py
Then use the PyQt5 UI to adjust musical and synthesis parameters in real time.

License
MIT License — see LICENSE file.
