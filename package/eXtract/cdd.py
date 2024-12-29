import numpy as np
import math
from collections import defaultdict

def compute_cdd(cell, bin_size=1_000_000, exponent_steps=0.33):
    """
    Oblicza rozkład prawdopodobieństwa kontaktów (CDD) w zależności od odległości genomowej,
    z opcją dostosowania rozmiaru kroków w skali logarytmicznej poprzez parametr exponent_steps.

    cell: 2D numpy.array (N x N) – macierz kontaktów
    bin_size: rozdzielczość w bp (domyślnie 1 Mb)
    exponent_steps: wartość określająca krok w skali logarytmicznej.
                    Np. exponent_steps = 0.1 oznacza, że podział będzie 10-krotnie gęstszy.
    """

    N = cell.shape[0]
    # Indeksy górnego trójkąta (bez diagonali)
    triu_i, triu_j = np.triu_indices(N, k=1)
    # Pobieramy odpowiadające im wartości kontaktów
    contact_vals = cell[triu_i, triu_j]

    # Filtrujemy wartości zerowe (lub ujemne - jeśli by się zdarzały)
    mask_nonzero = contact_vals > 0
    triu_i = triu_i[mask_nonzero]
    triu_j = triu_j[mask_nonzero]
    contact_vals = contact_vals[mask_nonzero]

    # Obliczamy odległość genomową
    dists = (triu_j - triu_i) * bin_size

    # Liczymy log2(dist)
    log_vals = np.log2(dists)

    # Wyznaczamy bin_start i bin_end, czyli przedziały log-binów
    bin_starts = np.floor(log_vals / exponent_steps) * exponent_steps
    bin_ends = bin_starts + exponent_steps

    # Zliczamy sumarycznie liczbę (waga) kontaktów w każdej "szufladce" log-bin
    dist_counts = defaultdict(float)
    dist_sums = float(np.sum(contact_vals))  # całkowita liczba kontaktów (do przeliczenia na prawdopodobieństwa)

    for bs, be, cval in zip(bin_starts, bin_ends, contact_vals):
        dist_counts[(bs, be)] += cval

    # Przeliczamy na prawdopodobieństwa
    bins_array = sorted(dist_counts.keys())  # klucze posortowane po bin_start
    probs_array = [dist_counts[b] / dist_sums for b in bins_array]

    return {
        "bins_array": bins_array,
        "probs_array": probs_array
    }