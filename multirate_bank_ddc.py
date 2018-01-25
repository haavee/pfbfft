#!/usr/bin/python
#
# DDC experiments which attempt to improve filter efficiency and performance.
#
# Ludwig Schwardt
# 26 October 2010
#

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

#### PARAMETERS

# Input sample rate, fs
sample_rate = 800e6
no_samples = 8192
# Input tone frequency
tone_freq = 254e6
# LO frequency used in mixing
lo_freq = 199e6
# Interpolate by factor L and decimate by factor M
# The settings below give 800e6 * L / M = 128e6
L = 4
M = 25
# Integer factors for the z^(-1) trick, determined via Euclid's theorem by solving -n0 * L + n1 * M = 1
n0 = 31
n1 = 5

# Filter parameters (operating at a sample rate of L * fs)
taps = 128
cutoff = 64e6

# Number of FFT samples in plots
no_freqs = 8 * no_samples

#### DESIGN FILTER

# Filter operates at highest sample rate (refer to brute-force block diagram)
nyq = L * sample_rate / 2.
# Transition bandwidth for Hamming-windowed design
trans_bw = nyq * 8 / taps
# Maximum allowed ripple (in linear terms)
pass_ripple = stop_ripple = 1 / 256.
# More refined filter specification of passband and stopband edge frequencies
# The Remez design reduces the transition BW to 0.64 of that of the Hamming design
# This keeps the maximum passband and stopband ripple below 1/256, which is good for 8-bit data
pass_edge = cutoff - .64 * trans_bw / 2.
stop_edge = cutoff + .64 * trans_bw / 2.
# Relative weight given to matching the stopband
stop_weight = 0.5
# Normalise all frequencies to Nyquist (i.e. f = 1 means pi rads / sample)
wc, wp, ws = cutoff / nyq, pass_edge / nyq, stop_edge / nyq

## Use windowing method
h_window = scipy.signal.firwin(taps, wc)

## Use Parks-Mclellan aka Remez exchange method
# Increased grid density a bit, as filter bandwidth is quite low (1 / M)
h_remez = scipy.signal.remez(taps, [0, wp, ws, 1], [1, 0], weight=[1 - stop_weight, stop_weight],
                             Hz=2, grid_density=64)

## Use eigenfilter method
# This method designs a FIR filter h[n] of even order N, having N+1 tap weights h[0] to h[N].
# Since the filter has even symmetry to ensure linear phase, it can be represented by
# the (N // 2 + 1) Fourier cosine coefficients b[m] of its real and even amplitude response.
# In this case, N is taken to be taps - 2, to ensure that this filter is not longer than one above.
coefs = (taps - 2) // 2 + 1
mm, nn = np.meshgrid(np.arange(coefs), np.arange(coefs))
Pstop = 0.5 * (np.sinc(nn + mm) - ws*np.sinc((nn + mm)*ws) + np.sinc(nn - mm) - ws*np.sinc((nn - mm)*ws))
Ppass = wp * (1 - np.sinc(nn*wp) - np.sinc(mm*wp) + 0.5*np.sinc((nn + mm)*wp) + 0.5*np.sinc((nn - mm)*wp))
Poverall = stop_weight * Pstop + (1 - stop_weight) * Ppass
# Pick eigenvector associated with smallest eigenvalue (minimising Rayleigh quotient)
u, s, vh = np.linalg.svd(Poverall)
b = u[:, -1]
# Recreate filter tap weights h[n] from cosine coefficients b[m]
h_eigen = np.hstack((0.5 * np.flipud(b[1:]), (b[0],), 0.5 * b[1:]))
h_eigen /= h_eigen.sum()
h = np.hstack((h_eigen, (0,)))

print "Effective transition regions:"
for name, filt in zip(['window', 'remez', 'eigen'], [h_window, h_remez, h_eigen]):
    # Amplitude response with MHz bins
    nfreqs = int(2 * nyq / 1e6)
    amp_resp = np.abs(np.fft.fft(filt, nfreqs))
    eff_pass_edge = (amp_resp[:nfreqs // 2] > 1 - pass_ripple).tolist().index(False)
    eff_stop_edge = (amp_resp[:nfreqs // 2] < stop_ripple).tolist().index(True)
    print "%s = %d to %d MHz" % (name, eff_pass_edge, eff_stop_edge)

# Choose filter
h = h_window

#### CREATE SIGNALS

# Discrete time indices
n = np.arange(no_samples)
# LO signal
cos_lo_n = 2 * np.cos(2 * np.pi * lo_freq / sample_rate * n)
# Input signal: single sinusoid at given frequency
x = np.cos(2 * np.pi * tone_freq / sample_rate * n)
# Input signal: bandlimited white noise
x = scipy.signal.lfilter(h, 1, np.random.randn(no_samples)) * \
    np.cos(2 * np.pi * (lo_freq + cutoff / 4) / sample_rate * n)

#### PERFORM DDC THE BRUTE-FORCE WAY (I channel only)

# Mix signal
i_mix = x * cos_lo_n
# Upsample
i_up = np.zeros(L * no_samples)
i_up[::L] = i_mix
# Filter
i_fir = scipy.signal.lfilter(h, 1, i_up)
# Downsample
i_out = i_fir[::M]

#### PERFORM MORE EFFICIENT DDC

# Mix signal
i_mix = x * cos_lo_n
# Polyphase components (type 1: E_0 to E_(L-1) or type 2: R_(L-1) to R_0)
poly_h = h.reshape(-1, L)
poly_taps = poly_h.shape[0]
# The delays might not be correct for all L and M...
poly_delay = np.arange(0, -L * n0, -n0) + M * (np.arange(0, L * n1, n1) // L)
poly_delay = np.flipud(poly_delay) - poly_delay.min()
# Effective filter memory
memory = poly_delay.max() + poly_taps
# Prepend zeroes to signal to serve as initial filter memory
# (this makes the comparison with the brute-force approach exact)
i_mix_plus_memory = np.hstack((np.zeros(poly_taps - 1), i_mix))
# Number of output samples when first interpolating and then decimating
no_outputs_up_down = (L * no_samples) // M + 1
y = np.zeros(no_outputs_up_down)
for l in range(L):
    tap_weights = np.flipud(poly_h[:, l])
    for k, start in enumerate(range(poly_delay[l], len(i_mix_plus_memory) - poly_taps, M)):
        y[l + k * L] = np.dot(i_mix_plus_memory[start:start + poly_taps], tap_weights)
assert np.all(y == i_out), 'Efficient version not identical to brute-force one'

print "Dumping output to disk\n"
y.astype(np.int8).tofile("py_ddc_out.dump") #dump the output as 8bit signed ints

#### DEBUGGING

# This compares the step-wise transformation of the brute-force filter to
# the efficient one, as illustrated in Figure 18 of P.P. Vaidyanathan's tut
# (Proc. IEEE, vol. 78, no. 1, Jan 1990).

# Step (a): Move interpolator past filter (which becomes polyphase), compare against brute-force UP+FIR
poly_x_a = np.zeros((no_samples, L))
for l in range(L):
    poly_x_a[:, l] = scipy.signal.lfilter(poly_h[:, l], 1, i_mix)
i_fir_check = poly_x_a.ravel()
assert np.all(i_fir_check == i_fir), 'Step (a) not identical'

# Step (c): Interchange interpolator and decimator
# Number of output samples when first decimating and then interpolating
no_outputs_down_up = L * (no_samples // M + 1)
y_c = np.zeros(no_outputs_down_up + L*n1)
for l in range(L):
    advanced_x = np.hstack((i_mix[l*n0:], np.zeros(l*n0)))
    y_c[l*n1:l*n1 + no_outputs_down_up:L] = scipy.signal.lfilter(poly_h[:, l], 1, advanced_x)[::M]
i_out_check = y_c[:len(i_out)]
# Attempt to figure out the discrepancy at start of filter output (not well understood yet)
discrepancy = max((memory + poly_taps) // M * L, M)
assert np.all(i_out_check[discrepancy:] == i_out[discrepancy:]), 'Step (c) not identical'

# Step (d): Move decimator before filter (which splits into polyphase components yet again)
y_d = np.zeros(no_outputs_down_up + L*n1)
for l in range(L):
    advanced_x = np.hstack((i_mix_plus_memory[l*n0:], np.zeros(l*n0)))
    tap_weights = np.flipud(poly_h[:, l])
    for start in range(0, len(advanced_x) - poly_taps, M):
        y_d[l*n1 + start // M * L] = np.dot(advanced_x[start:start + poly_taps], tap_weights)
i_out_check2 = y_d[:len(i_out)]
assert np.all(i_out_check2[discrepancy:] == i_out[discrepancy:]), 'Step (d) not identical'

# Plot it up :)
freq = np.arange(0, sample_rate, sample_rate/no_freqs) / 1e6

plt.figure(1)
plt.subplot(311)
plt.title("Scaled FFT of input signal")
plt.plot(freq, abs(np.fft.fft(x, no_freqs)) / no_samples)
plt.xlim(0, sample_rate / 1e6)
plt.xticks([])
plt.subplot(312)
plt.title("Scaled FFT of LO signal")
plt.plot(freq, abs(np.fft.fft(cos_lo_n, no_freqs)) / no_samples)
plt.xlim(0, sample_rate / 1e6)
plt.xticks([])
plt.subplot(313)
plt.title("Scaled FFT of mixer output")
plt.plot(freq, abs(np.fft.fft(i_mix, no_freqs)) / no_samples)
plt.xlim(0, sample_rate / 1e6)
plt.xlabel('Frequency (MHz)')

plt.figure(2)
plt.subplot(311)
plt.title("Scaled FFT of upsampler output")
plt.plot(L * freq, abs(np.fft.fft(i_up, no_freqs)) / no_samples)
plt.xlim(0, L * sample_rate / 1e6)
plt.xticks([])
plt.subplot(312)
plt.title("Frequency response of FIR filter")
plt.plot(L * freq, abs(np.fft.fft(h, no_freqs)))
plt.xlim(0, L * sample_rate / 1e6)
plt.xticks([])
plt.subplot(313)
plt.title("Scaled FFT of filter output")
plt.plot(L * freq, abs(np.fft.fft(i_fir, no_freqs)) / no_samples)
plt.xlim(0, L * sample_rate / 1e6)
plt.xlabel('Frequency (MHz)')

plt.figure(3)
plt.subplot(211)
plt.title("Final output")
plt.plot(i_out, label='brute force')
plt.plot(y, label='efficient')
plt.legend()
plt.xlabel('Samples')
plt.subplot(212)
plt.title("Scaled FFT of final output")
plt.plot(freq * L / M, abs(np.fft.fft(i_out, no_freqs)) * M / no_samples, label='brute force')
plt.plot(freq * L / M, abs(np.fft.fft(y, no_freqs)) * M / no_samples, label='efficient')
plt.xlim(0, sample_rate * L / M / 1e6)
plt.xlabel('Frequency (MHz)')

plt.figure(4)
plt.title("Filter impulse response")
plt.plot(h_window[1:] if taps % 2 == 0 else h_window, label='Windowing')
plt.plot(h_eigen, label='Eigenfilter')
plt.plot(h_remez, label='Remez')
plt.legend()
plt.xlabel('Samples')

plt.figure(5)
plt.title("Filter frequency response")
plt.plot(L * freq, 20 * np.log10(abs(np.fft.fft(h_window, no_freqs))), label='Windowing')
plt.plot(L * freq, 20 * np.log10(abs(np.fft.fft(h_eigen, no_freqs))), label='Eigenfilter')
plt.plot(L * freq, 20 * np.log10(abs(np.fft.fft(h_remez, no_freqs))), label='Remez')
plt.xlim(0, L * sample_rate / 1e6)
plt.legend()
plt.xlabel('Frequency (MHz)')

plt.show()

