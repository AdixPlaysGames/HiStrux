import numpy as np

def compute_insulation_scores(cell: np.ndarray, scale: int) -> np.ndarray:
    """
    Oblicza insulation score dla każdego bina w macierzy kontaktów Hi-C.

    Parametry:
    -----------
    cell : np.ndarray
        Kwadratowa macierz kontaktów Hi-C (N x N), już zbinowana do 1 Mb.
    scale : int
        Liczba binów po każdej stronie aktualnego bina, które bierzemy do sumowania.
        Jeśli scale = 1, to rozpatrujemy otoczenie [b-1, b+1] zarówno w wierszach jak i kolumnach.

    Zwraca:
    --------
    ins_scores : np.ndarray
        Wektor długości N z obliczonymi wartościami insulation score dla każdego bina.
    """

    N = cell.shape[0]
    ins_scores = np.zeros(N, dtype=float)

    # Zakładamy, że kontakt matrix jest symetryczna.
    # Ins(b) = suma kontaktów w oknie (b-scale : b+scale, b-scale : b+scale)
    # z uwzględnieniem granic macierzy.

    for b in range(N):
        start = max(b - scale, 0)
        end = min(b + scale, N - 1)

        # Wycinamy podmatrycę z cell. Jest to kwadrat o wymiarach [2*scale + 1]
        # (lub mniejszy przy krawędziach).
        submat = cell[start:end+1, start:end+1]

        # Suma wszystkich kontaktów w tym podobszarze:
        ins_scores[b] = np.nansum(submat)  # używamy nansum na wypadek wartości NaN

    return ins_scores


def compute_insulation_features(cell: np.ndarray, scale: int=100) -> dict:
    """
    Zwraca słownik z kilkoma statystykami insulation,
    przydatnymi do klasyfikacji scHi-C w różne etapy cyklu komórkowego.
    """
    N = cell.shape[0]
    ins_scores = np.zeros(N, dtype=float)

    for b in range(N):
        left_start = max(b - scale, 0)
        left_end   = b
        right_start= b + 1
        right_end  = min(b + scale, N - 1)

        if left_end < left_start or right_end < right_start:
            ins_scores[b] = 0.0
            continue

        submat = cell[left_start:left_end+1, right_start:right_end+1]
        ins_scores[b] = np.nansum(submat)

    # Teraz mamy wektor ins_scores
    avg = np.mean(ins_scores)
    med = np.median(ins_scores)
    std = np.std(ins_scores)
    
    # np.percentile: dobry do zbadania "minima" czy "maksima"
    p10 = np.percentile(ins_scores, 10)
    p90 = np.percentile(ins_scores, 90)

    # Zwracamy zbiór interesujących liczb
    return {
        "mean_ins": float(avg),
        "median_ins": float(med),
        "std_ins": float(std),
        "p10_ins": float(p10),
        "p90_ins": p90,
        "vector": ins_scores  # można tez zachować cały wektor
    }