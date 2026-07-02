# ---------------------------------------------------------------------------
# module_6_filter_structures.py -- Discrete Filter Structure Architectures
#
# Implements the Direct Form II Transposed (DF2T) filter structure as an
# explicit sample-by-sample processing loop.  This is the core engine
# used by Modules 7 and 8 for all IIR and FIR filtering.
#
# Key DSP concepts:
#   - Difference equations for causal LTI systems
#   - Direct Form II Transposed realisation (minimises delay elements)
#   - Internal state (delay) registers and their update equations
#   - Canonical filter implementations
#
# IMPORTANT: No calls to scipy.signal.lfilter or filtfilt are made here.
#            Every multiply-accumulate step is computed explicitly.
#
# DF2T Algorithm
# ~~~~~~~~~~~~~~
# Given the general IIR transfer function:
#
#     H(z) = B(z) / A(z) = (b0 + b1*z^-1 + ... + bM*z^-M)
#                         / (1  + a1*z^-1 + ... + aN*z^-N)
#
# The transposed form uses K-1 state variables (where K = max(M+1, N+1))
# and processes each input sample as follows:
#
#     y[n]     = b[0]*x[n]               + w_0[n-1]
#     w_0[n]   = b[1]*x[n] - a[1]*y[n]  + w_1[n-1]
#     w_1[n]   = b[2]*x[n] - a[2]*y[n]  + w_2[n-1]
#     ...
#     w_{K-2}[n] = b[K-1]*x[n] - a[K-1]*y[n]
# ---------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt


def direct_form_2_transposed(b, a, x):
    """Filter a signal using the Direct Form II Transposed structure.

    This function implements the standard DF2T algorithm with explicit
    state registers and a sample-by-sample processing loop.  No external
    filtering libraries are called.

    Parameters
    ----------
    b : array-like
        Numerator (feedforward) coefficients.  Length M+1.
    a : array-like
        Denominator (feedback) coefficients.  Length N+1.
        a[0] is expected to be non-zero; the function normalises so
        that a[0] = 1.
    x : np.ndarray
        Input signal array.

    Returns
    -------
    y : np.ndarray
        Filtered output signal, same length as x.
    """
    b = np.asarray(b, dtype=float).copy()
    a = np.asarray(a, dtype=float).copy()

    # Normalise so that a[0] = 1
    b = b / a[0]
    a = a / a[0]

    # Pad b and a to the same length K
    K = max(len(b), len(a))
    bp = np.zeros(K)
    ap = np.zeros(K)
    bp[:len(b)] = b
    ap[:len(a)] = a

    N = len(x)
    y = np.zeros(N, dtype=float)

    # State registers (delay elements).  K-1 internal states are needed.
    w = np.zeros(K - 1, dtype=float)

    # ---- Main sample-by-sample processing loop ----
    for n in range(N):
        # Output equation: combine first feedforward tap with first state
        y[n] = bp[0] * x[n] + w[0]

        # State update: shift each state forward and apply the tapped
        # delay line equations.  Each state combines the next feedforward
        # contribution, the feedback contribution, and the next state.
        for i in range(K - 2):
            w[i] = bp[i + 1] * x[n] - ap[i + 1] * y[n] + w[i + 1]

        # The last state has no successor to draw from
        if K >= 2:
            w[K - 2] = bp[K - 1] * x[n] - ap[K - 1] * y[n]

    return y


# ============================= PLOTTING ====================================

def plot_impulse_response_test(b_test, a_test, y_impulse, output_dir):
    """Save a stem plot of the DF2T impulse response for verification."""
    n = np.arange(len(y_impulse))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stem(n[:60], y_impulse[:60], linefmt="C0-", markerfmt="C0o",
            basefmt="k-")
    ax.set_xlabel("Sample index  n")
    ax.set_ylabel("h[n]")
    ax.set_title("Module 6 -- DF2T Impulse Response Verification")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "df2t_impulse_response.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Module 6] Saved impulse response   -> {path}")
    return path


# ============================= MODULE RUNNER ===============================

def run(output_dir):
    """Verify the DF2T implementation with a known test case.

    Test: apply a simple 2nd-order IIR filter to an impulse and
    compare the output against a manually computed reference.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("MODULE 6 -- Discrete Filter Structure Architectures")
    print("=" * 70)

    # Test filter:  H(z) = (1 + 0.5*z^-1) / (1 - 0.8*z^-1 + 0.64*z^-2)
    # This is a stable IIR filter (poles inside the unit circle).
    b_test = np.array([1.0, 0.5])
    a_test = np.array([1.0, -0.8, 0.64])

    # Create an impulse signal
    impulse_len = 100
    impulse = np.zeros(impulse_len)
    impulse[0] = 1.0

    # Run through DF2T
    y_df2t = direct_form_2_transposed(b_test, a_test, impulse)

    # Manually compute the first few samples using the difference equation:
    #   y[n] = x[n] + 0.5*x[n-1] + 0.8*y[n-1] - 0.64*y[n-2]
    y_manual = np.zeros(5)
    x = impulse
    y_manual[0] = x[0]                                           # = 1.0
    y_manual[1] = x[1] + 0.5 * x[0] + 0.8 * y_manual[0]         # = 0 + 0.5 + 0.8 = 1.3
    y_manual[2] = x[2] + 0.5 * x[1] + 0.8 * y_manual[1] - 0.64 * y_manual[0]
    y_manual[3] = x[3] + 0.5 * x[2] + 0.8 * y_manual[2] - 0.64 * y_manual[1]
    y_manual[4] = x[4] + 0.5 * x[3] + 0.8 * y_manual[3] - 0.64 * y_manual[2]

    max_err = np.max(np.abs(y_df2t[:5] - y_manual))

    print(f"  Test filter: H(z) = (1 + 0.5z^-1) / (1 - 0.8z^-1 + 0.64z^-2)")
    print(f"  Impulse response (DF2T):   {y_df2t[:5].round(6)}")
    print(f"  Impulse response (manual): {y_manual.round(6)}")
    print(f"  Max error (first 5 samples): {max_err:.2e}")
    print(f"  Verification: {'PASSED' if max_err < 1e-10 else 'FAILED'}")

    plot_impulse_response_test(b_test, a_test, y_df2t, output_dir)

    return {"b_test": b_test, "a_test": a_test, "impulse_response": y_df2t}
