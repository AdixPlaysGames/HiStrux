import pandas as pd
import numpy as np
from sklearn.decomposition import PCA # type: ignore
from .imputation import imputation

def compute_ab_compartments(
    contacts_df: pd.DataFrame,
    bin_size: int = 1_000_000, w=5, p=0.85, threshold_percentile=90,
    imputation_involved=False
) -> pd.DataFrame:
    """
    Oblicza wartości PC1 (główny komponent) dla każdego bina w genomie,
    aby wyznaczyć A/B compartments metodą PCA.

    Parametry:
    ----------
    contacts_df : pd.DataFrame
        Ramka danych z kolumnami:
        - chromosome_1
        - start_1
        - end_1
        - chromosome_2
        - start_2
        - end_2
        - (opcjonalnie) mapping_quality, cell_id, itp.

    bin_size : int
        Wielkość bina (w bp); domyślnie 1 Mb = 1,000,000 bp.

    Zwraca:
    --------
    result_df : pd.DataFrame
        Ramka danych z kolumnami:
        - chromosome
        - bin_start
        - bin_end
        - PC1
        - compartment_label  (np. 'A' lub 'B')

        W kolejności rosnącej po chromosomie i binach.
    """

    # --- 1. Filtrowanie wyłącznie cis (kontakty w obrębie tego samego chromosomu) ---
    cis_df = contacts_df[contacts_df["chromosome_1"] == contacts_df["chromosome_2"]].copy()
    if len(cis_df) == 0:
        raise ValueError("Nie znaleziono cis-kontaktów w przekazanej ramce danych.")

    # --- 2. Wyznaczenie zakresu (min, max) dla każdego chromosomu, żeby wiedzieć ile binów potrzebujemy ---
    # Przy okazji ustalamy ID binu dla 'start_1', 'start_2' itp.
    def get_bin_id(start, bin_size):
        return start // bin_size

    cis_df["bin_1"] = cis_df["start_1"].apply(lambda x: get_bin_id(x, bin_size))
    cis_df["bin_2"] = cis_df["start_2"].apply(lambda x: get_bin_id(x, bin_size))

    # --- 3. Dla wygody grupujemy osobno każdy chromosom ---
    chrom_list = cis_df["chromosome_1"].unique()

    # Lista wyników do scalania
    results = []

    for chrom in sorted(chrom_list):
        chrom_data = cis_df[cis_df["chromosome_1"] == chrom]

        # Jeśli mamy sumę mapping_quality na bin – można to zsumować.
        # Albo, jeżeli mamy liczbę kontaktów – też można to zliczyć.
        # Tu załóżmy, że liczymy po prostu liczbę kontaktów (count):
        bin_contacts = (
            chrom_data
            .groupby(["bin_1", "bin_2"])
            .size()  # -> liczba kontaktów w danym bin_1, bin_2
            .reset_index(name="contact_count")
        )

        # Ustalamy maksymalny ID binu (żeby wiedzieć, jak duża będzie macierz)
        max_bin_id = bin_contacts[["bin_1", "bin_2"]].max().max()

        # Budujemy macierz (N x N) dla danego chromosomu
        N = max_bin_id + 1
        contact_matrix = np.zeros((N, N), dtype=float)

        for row in bin_contacts.itertuples(index=False):
            i = row.bin_1
            j = row.bin_2
            c = row.contact_count
            # Macierz symetryczna (zakładamy scHi-C w stylu -> i, j == j, i)
            contact_matrix[i, j] += c
            contact_matrix[j, i] += c

        if imputation_involved is True:
          contact_matrix = imputation(contact_matrix, w=w, p=p, threshold_percentile=threshold_percentile)


        # --- 4. Zamieniamy macierz na macierz korelacji (lub O/E, etc.) ---
        # Dla prostoty: weźmy korelację Pearsona między kolumnami (binami),
        # co często się robi w pakietach do compartments.
        # Trzeba uważać na wiersze z zerami (niskie coverage).
        # Często poprzedza się to normalizacją - np. liczymy sumę w wierszu, sumę w kolumnie.
        # Tu dla uproszczenia: liczymy po prostu korelację kolumn.

        # Jeżeli kolumna to bin, musimy wziąć contact_matrix.T, bo np. np.corrcoef liczy korelację wzdłuż wierszy.
        # Bierzemy np. corrcoef(contact_matrix) lub corrcoef(contact_matrix.T) zależnie od konwencji.

        with np.errstate(invalid='ignore'):
            corr_matrix = np.corrcoef(contact_matrix)

        # Zdarza się, że jakieś biny są same 0 => pojawi się NaN w korelacji.
        # Możemy je później wykluczyć z PCA albo zastąpić 0.
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # --- 5. PCA po macierzy korelacji ---
        # PCA użyjemy z sklearn.decomposition
        pca = PCA(n_components=1)
        # Uwaga: aby puścić PCA na macierzy korelacji NxN, musimy mieć NxN obserwacji,
        # a standardowo PCA oczekuje X w postaci (n_samples, n_features).
        # Rozwiązanie: bierzemy wektory wierszy z corr_matrix lub kolumn.
        # Z reguły compartments wyznacza się: PC1 w kolumnach, więc X = corr_matrix.
        # Ale pamiętajmy, że musimy mieć w (n_samples, n_features).
        # Weźmy corr_matrix jako dane NxN i puśćmy PCA wprost, przy czym interpretacja:
        # "sample" = bin, "features" = korelacje z innymi binami.
        # Ewentualnie:
        X = corr_matrix

        pca.fit(X)
        # pc1 = pca.components_[0]  # To jest wektor 1 x N, da nam wagi cechy, ewentualnie
        # ALE w sklearn, pca.components_[0] to "vector of features" – w tym wypadku "binów".
        # Zwykle do compartments bierze się współrzędne w przestrzeni PC1 = pca.transform(X)[:, 0].
        # Obie metody mają sens, bo PC1 bywa definiowany różnie w zależności od konwencji.

        pc1_coords = pca.transform(X)[:, 0]
        # pc1_coords to tablica wielkości N, czyli "score" dla każdego bina.

        # Normalnie może się zdarzyć, że PC1 w niektórych pracach jest odwrócony
        # (tj. '+' = A, '-' = B), a w innych na odwrót.
        # Często ustala się znak tak, by A odpowiadało regionom bogatym w geny/GC.
        # Tutaj zostawiamy surowy sygnał.

        # --- 6. Budujemy ramkę wynikową dla danego chromosomu ---
        data_out = []
        for bin_id in range(N):
            start_bp = bin_id * bin_size
            end_bp = (bin_id + 1) * bin_size - 1
            val = pc1_coords[bin_id]
            label = "A" if val >= 0 else "B"
            data_out.append((chrom, bin_id, start_bp, end_bp, val, label))

        chrom_df = pd.DataFrame(
            data_out,
            columns=["chromosome", "bin_id", "bin_start", "bin_end", "PC1", "compartment_label"]
        )
        results.append(chrom_df)

    # Scalanie w jedną ramkę
    result_df = pd.concat(results, ignore_index=True)
    # Ewentualnie sortujemy
    result_df.sort_values(by=["chromosome", "bin_start"], inplace=True)

    return result_df


def compute_ab_stats(result_df: pd.DataFrame, contacts_df: pd.DataFrame, bin_size: int = 1_000_000):
    """
    Prosty przykład funkcji liczącej np. 'cis AB fraction',
    tj. udział kontaktów A-A, B-B oraz A-B w kontaktach cis (dla całego genomu lub per chromosom).

    Parametry:
    ----------
    result_df: pd.DataFrame
        Wynik funkcji `compute_ab_compartments`, tj. kolumny:
        [chromosome, bin_id, bin_start, bin_end, PC1, compartment_label]

    contacts_df: pd.DataFrame
        Oryginalna ramka z kontaktami (chromosome_1 == chromosome_2).

    bin_size: int
        Wielkość bina.

    Zwraca:
    --------
    ab_stats: pd.DataFrame
        Tabela zliczająca liczbę kontaktów A-A, B-B oraz A-B (i ich proporcje).
    """
    # Filtrowanie tylko cis:
    cis_df = contacts_df[contacts_df["chromosome_1"] == contacts_df["chromosome_2"]].copy()
    # Dodajemy bin_id dla start_1, start_2
    def get_bin_id(start):
        return start // bin_size

    cis_df["bin_1"] = cis_df["start_1"].apply(get_bin_id)
    cis_df["bin_2"] = cis_df["start_2"].apply(get_bin_id)

    # Słownik: (chrom, bin_id) -> 'A' lub 'B'
    compartment_map = {}
    for row in result_df.itertuples():
        compartment_map[(row.chromosome, row.bin_id)] = row.compartment_label

    # Zliczamy, ile jest kontaktów A-A, B-B, A-B
    # Zakładamy, że liczymy "liczbę kontaktów",
    # jeśli mamy mapping_quality i chcemy sumę, to lekko modyfikujemy kod.
    # Poniżej liczymy 'count' każdy kontakt = 1.

    stats = {"AA": 0, "BB": 0, "AB": 0}
    for row in cis_df.itertuples():
        chrom = row.chromosome_1
        b1 = row.bin_1
        b2 = row.bin_2
        c1 = compartment_map.get((chrom, b1), None)
        c2 = compartment_map.get((chrom, b2), None)
        if c1 is not None and c2 is not None:
            if c1 == "A" and c2 == "A":
                stats["AA"] += 1
            elif c1 == "B" and c2 == "B":
                stats["BB"] += 1
            else:
                stats["AB"] += 1

    total = stats["AA"] + stats["BB"] + stats["AB"]
    if total == 0:
        return pd.DataFrame({"contact_type": [], "count": [], "fraction": []})

    # Tworzymy dataframe z wynikami
    contact_types = []
    for k in ["AA", "BB", "AB"]:
        contact_types.append(
            {
                "contact_type": k,
                "count": stats[k],
                "fraction": stats[k] / total
            }
        )
    ab_stats = pd.DataFrame(contact_types)

    return ab_stats

def calculate_cis_ab_comp(
    contacts_df: pd.DataFrame,
    bin_size: int = 1_000_000,
    w: int = 5,
    p: float = 0.85,
    threshold_percentile: int = 90,
    imputation_involved=False
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # 1) Obliczamy A/B compartments
    compartments_df = compute_ab_compartments(
        contacts_df=contacts_df,
        bin_size=bin_size,
        w=w,
        p=p,
        threshold_percentile=threshold_percentile,
        imputation_involved=imputation_involved
    )

    # 2) Obliczamy statystyki A-B
    ab_stats_df = compute_ab_stats(
        result_df=compartments_df,
        contacts_df=contacts_df,
        bin_size=bin_size
    )

    return ab_stats_df