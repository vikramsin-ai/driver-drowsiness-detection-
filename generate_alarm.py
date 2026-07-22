"""
generate_alarm.py
-----------------
Run this ONCE to create the alarm.wav file used by the detector.
No external downloads needed — generates a beep tone using numpy + wave.

Usage:
    python generate_alarm.py
"""

import wave
import struct
import math
import os

OUTPUT = "alarm.wav"
SAMPLE_RATE = 44100
DURATION    = 1.0      # seconds per beep
FREQUENCY   = 1000     # Hz (beep pitch)
AMPLITUDE   = 28000    # loudness (max 32767)
REPEATS     = 3        # number of beep-silence cycles

def generate_beep(freq, duration, sample_rate, amplitude):
    samples = []
    n = int(sample_rate * duration)
    for i in range(n):
        t = i / sample_rate
        val = int(amplitude * math.sin(2 * math.pi * freq * t))
        samples.append(val)
    return samples

def generate_silence(duration, sample_rate):
    return [0] * int(sample_rate * duration)

all_samples = []
for _ in range(REPEATS):
    all_samples += generate_beep(FREQUENCY, DURATION, SAMPLE_RATE, AMPLITUDE)
    all_samples += generate_silence(0.3, SAMPLE_RATE)

with wave.open(OUTPUT, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    packed = struct.pack(f"<{len(all_samples)}h", *all_samples)
    wf.writeframes(packed)

print(f"[OK] Created {OUTPUT} ({os.path.getsize(OUTPUT)//1024} KB)")
