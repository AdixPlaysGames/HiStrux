import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .imputation import imputation


def compute_directionality_index(contact_matrix, window=5):
    """
    Uproszczone wyliczanie directionality index (DI).
    """
    N = contact_matrix.shape[0]
    di_values = np.zeros(N, dtype=float)

    for i in range(N):
        left_start = max(i - window, 0)
        left_end   = i
        right_start = i + 1
        right_end   = min(i + window + 1, N)

        left_sum = contact_matrix[left_start:left_end, i].sum()
        right_sum = contact_matrix[right_start:right_end, i].sum()

        numerator = (right_sum - left_sum)
        denominator = abs(left_sum + right_sum)
        if denominator == 0:
            di = 0
        else:
            di = numerator / denominator
        di_values[i] = di

    return di_values

def detect_tad_boundaries(di_values, threshold=0.8):
    """
    Bardzo uproszczona detekcja granic TAD na podstawie directionality index.
    """
    boundaries = []
    prev_sign = np.sign(di_values[0])

    for i in range(1, len(di_values)):
        cur_sign = np.sign(di_values[i])
        if cur_sign != prev_sign and abs(di_values[i]) > threshold:
            boundaries.append(i)
        prev_sign = cur_sign

    return boundaries

def build_tad_df(chrom, boundaries, n_bins, bin_size):
    """
    Z listy granic buduje ramkę danych z kolumnami:
      [chromosome, tad_id, start_bin, end_bin, start_bp, end_bp, size_in_bins]
    """
    tads = []
    start_bin = 0
    tad_id = 0

    for b in boundaries:
        end_bin = b - 1
        if end_bin < start_bin:
            continue
        tads.append((chrom, tad_id, start_bin, end_bin))
        start_bin = b
        tad_id += 1

    # Ostatni TAD
    if start_bin < n_bins:
        tads.append((chrom, tad_id, start_bin, n_bins - 1))

    data_out = []
    for (chr_, tid, sb, eb) in tads:
        sb_bp = sb * bin_size
        eb_bp = (eb + 1) * bin_size - 1
        size_in_bins = eb - sb + 1
        data_out.append(
            (chr_, tid, sb, eb, sb_bp, eb_bp, size_in_bins)
        )

    tad_df = pd.DataFrame(data_out, columns=[
        "chromosome", "tad_id",
        "start_bin", "end_bin",
        "start_bp", "end_bp",
        "size_in_bins"
    ])
    return tad_df

def compute_tad_stats(tad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Liczy wybrane statystyki TAD dla każdej pary (chromosom).
    """
    if tad_df.empty:
        return pd.DataFrame(columns=["chromosome", "n_tads", "mean_size_in_bins", 
                                     "min_size_in_bins", "max_size_in_bins"])

    stats_list = []
    for chrom, group in tad_df.groupby("chromosome"):
        n_tads = len(group)
        mean_size = group["size_in_bins"].mean()
        min_size = group["size_in_bins"].min()
        max_size = group["size_in_bins"].max()

        stats_list.append({
            "chromosome": chrom,
            "n_tads": n_tads,
            "mean_size_in_bins": mean_size,
            "min_size_in_bins": min_size,
            "max_size_in_bins": max_size
        })
    stats_df = pd.DataFrame(stats_list)
    return stats_df


def plot_tads_for_chrom(
    contact_matrix: np.ndarray,
    boundaries: list[int],
    chrom: str,
    out_prefix: str = None,
    show_plot: bool = True
):
    """
    Rysuje macierz kontaktów (np. heatmapę) dla danego chromosomu
    oraz rysuje pionowe i poziome linie w miejscach granic TAD.
    
    Parametry:
    -----------
      contact_matrix: 2D np.ndarray (N x N)
      boundaries: lista indeksów binów, np. [3, 48, 195] itd.
      chrom: str
      out_prefix: opcjonalny prefix do zapisu wykresu do pliku
      show_plot: czy pokazywać plt.show() (domyślnie True)
                 jeśli generujesz wiele chromosomów, można ustawić False
                 i robić plt.close() / plt.savefig() w pętli
    """
    N = contact_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    # Dla lepszej czytelności można użyć log1p:
    cax = ax.imshow(np.log1p(contact_matrix), 
                    cmap='Reds', 
                    origin='upper',
                    interpolation='nearest')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Dodajemy linie dla boundary:
    for b in boundaries:
        if 0 < b < N:
            ax.axhline(y=b - 0.5, color='blue', linewidth=0.7)
            ax.axvline(x=b - 0.5, color='blue', linewidth=0.7)

    ax.set_title(f"Chrom: {chrom} - TAD boundaries")
    ax.set_xlabel("bin index")
    ax.set_ylabel("bin index")

    # Opcjonalne zapisywanie
    if out_prefix is not None:
        plt.savefig(f"{out_prefix}_{chrom}_tads.png", dpi=150, bbox_inches='tight')
    # Pokazanie/ukrycie
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def calculate_cis_tads(
    contacts_df: pd.DataFrame,
    bin_size: int = 1_000_000,
    w: int = 5,
    p: float = 0.85,
    threshold_percentile: int = 90,
    imputation_involved: bool = False,
    boundary_threshold: float = 0.8,
    out_prefix: str = None,
    show_plot: bool = False,
    substring: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rozszerzona wersja, która (po wykryciu TAD-ów) tworzy 
    wykresy macierzy kontaktów wraz z granicami TAD dla każdego chromosomu.

    Parametry (jak poprzednio), plus:
      - out_prefix: prefix do zapisu wykresów do plików PNG (jeśli None, nie zapisujemy).
      - show_plot: czy wyświetlać plt.show() (True) czy tylko zapisać do pliku/pozamykać.

    Zwraca:
      (stats_df, tad_df)
    """
    # Filtrowanie cis
    contacts_df['chromosome_1'] = contacts_df['chromosome_1'].str[:-substring]
    contacts_df['chromosome_2'] = contacts_df['chromosome_2'].str[:-substring]
    
    cis_df = contacts_df[contacts_df["chromosome_1"] == contacts_df["chromosome_2"]].copy()
    if cis_df.empty:
        raise ValueError("Brak cis-kontaktów w podanej ramce danych.")

    # Biny
    cis_df["bin_1"] = cis_df["start_1"] // bin_size
    cis_df["bin_2"] = cis_df["start_2"] // bin_size

    chrom_list = cis_df["chromosome_1"].unique()
    all_tads = []

    for chrom in sorted(chrom_list):
        chrom_data = cis_df[cis_df["chromosome_1"] == chrom]
        bin_contacts = (
            chrom_data
            .groupby(["bin_1", "bin_2"])
            .size()
            .reset_index(name="contact_count")
        )

        max_bin_id = bin_contacts[["bin_1", "bin_2"]].max().max()
        N = max_bin_id + 1
        contact_matrix = np.zeros((N, N), dtype=float)

        for row in bin_contacts.itertuples(index=False):
            i = row.bin_1
            j = row.bin_2
            c = row.contact_count
            contact_matrix[i, j] += c
            contact_matrix[j, i] += c

        # (opcjonalnie) imputacja
        if imputation_involved:
            contact_matrix = imputation(
                contact_matrix,
                w=w,
                p=p,
                threshold_percentile=threshold_percentile
            )

        # Detekcja TAD
        di_vals = compute_directionality_index(contact_matrix, window=w)
        boundaries = detect_tad_boundaries(di_vals, threshold=boundary_threshold)
        tad_df_chrom = build_tad_df(chrom, boundaries, N, bin_size)
        all_tads.append(tad_df_chrom)

        # RYSOWANIE WYKRESU
        plot_tads_for_chrom(
            contact_matrix=contact_matrix,
            boundaries=boundaries,
            chrom=chrom,
            out_prefix=out_prefix,
            show_plot=show_plot
        )

    # Sklejamy TAD-y
    tad_df = pd.concat(all_tads, ignore_index=True)
    # Statystyki
    stats_df = compute_tad_stats(tad_df)

    return {"stats_df": stats_df,
            "tad_df": tad_df}