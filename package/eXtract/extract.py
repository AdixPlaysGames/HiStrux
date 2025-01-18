import pandas as pd
import tkinter as tk
import numpy as np
from pandastable import Table, config # type: ignore
from .process import process
from .primary import compute_basic_metrics
from .compartments import calculate_cis_ab_comp
from .cdd import compute_cdd
from .ins import compute_ins_features_for_each_chr
from .mcm import compute_mcm
from .pofs import compute_contact_scaling_exponent
from .tad import compute_tad_features


# process
def eXtract(cell_dataframe: pd.DataFrame,
            cell_id: str = None,
            chromosome_lengths: list[tuple[str, int]] = None,
            bin_size: int = 500_000,
            selected_chromosomes: list[str] = None,
            trans_interactions: bool = True,
            mapping_quality_involved: bool = False,
            substring = 2,

            # compartments
            compartments_w: int = 4,
            compartments_p: float = 0.85,
            compartments_bin_size: int = 400_000,

            # ins
            ins_bin_size: int = 400_000,
            scale: int = 15,
            ins_p: float=0.85,
            ins_w: int = 3,

            # mcm | tutaj wywalone
            near_threshold: float = 2.0,
            mid_threshold: float=6.0,

            # pofs
            min_distance: int = 1,
            max_distance: int = 100,
            pofs_bin_size: int= 500_000,
            num_log_bins: int = 40,

            # tad
            tad_bin_size: int = 300_000,
            tad_w: int = 3,
            tad_boundry_threshold: float = 0.05,
            out_prefix: str = None,

            # main
            vectorize: bool = False
            ) -> dict:
    
    cell_df = cell_dataframe.copy()

    if cell_id is None:
        cell_id = cell_df['cell_id'][0]
    
    cell_df = cell_df[cell_df['cell_id'] == cell_id]

    if substring is not None:
        cell_df['chromosome_1'] = cell_df['chromosome_1'].str[:-substring]
        cell_df['chromosome_2'] = cell_df['chromosome_2'].str[:-substring]

    cell_matrix = process(cell_df, cell_id=cell_id, chromosome_lengths=chromosome_lengths, bin_size=bin_size,
                          selected_chromosomes=selected_chromosomes, trans_interactions=True, substring=None)

    # Compartments ---------------------------------
    compartments = calculate_cis_ab_comp(
        cell_df,
        bin_size = compartments_bin_size,
        w = compartments_w,
        p = compartments_p,
        imputation_involved=True,
        plot = False
    )

    # MCM ------------------------------------------
    mcm = compute_mcm(cell_matrix, near_threshold=near_threshold, mid_threshold=mid_threshold)

    # P(s) -----------------------------------------
    cell_matrix_cis = process(cell_df, cell_id=cell_id, chromosome_lengths=chromosome_lengths, bin_size=pofs_bin_size,
                              selected_chromosomes=selected_chromosomes, trans_interactions=False, substring=None)
    pofs = compute_contact_scaling_exponent(cell_matrix_cis, min_distance=min_distance, max_distance=max_distance,
                                            plot=False, num_log_bins=num_log_bins)
    
    # Primary --------------------------------------
    primary = compute_basic_metrics(cell_df)

    # TAD ------------------------------------------
    if vectorize == False:
        tad = compute_tad_features(cell_df, bin_size=tad_bin_size, w=tad_w, p=0.85, imputation_involved=True,
                                boundary_threshold=tad_boundry_threshold, out_prefix=out_prefix, show_plot=False)
        

    # Ins ------------------------------------------
    ins = compute_ins_features_for_each_chr(cell_df, bin_size=ins_bin_size, plot=False,
                                            plot_insulation=False, w=ins_w, p=ins_p, scale=scale)
    

    ###############################################################################################################


    if vectorize:
        vector = []
        values = []
        # Primary
        vector += [value for key, value in primary.items()]
        values += [key for key, value in primary.items()]
        # Compartments
        vector += compartments['fraction'].tolist()
        values += compartments['contact_type'].tolist()
        # MCM
        vector += [value for key, value in mcm.items()]
        values += [key for key, value in mcm.items()]
        # P(s)
        vector += [v for k, v in pofs.items() if k != "pofs_distances" for v in (v if isinstance(v, np.ndarray) else [v])]
        values += [
            f"pofs{i}" if k == "p_of_s" and isinstance(v, np.ndarray) else k
            for k, v in pofs.items() if k != "pofs_distances"
            for i in range(len(v) if isinstance(v, np.ndarray) and k == "p_of_s" else 1)
        ]
        # Ins
        vector += [item[1][1] for item in ins]
        values += [f"{item[0]}_value" for item in ins]

        return values
    
    else:
        row_data = {
        "contact_type_AA": compartments.loc[0, 'fraction'],
        "contact_type_BB": compartments.loc[1, 'fraction'],
        "contact_type_AB": compartments.loc[2, 'fraction'],
        **mcm,
        **{key: value for key, value in pofs.items() if key not in {"pofs_distances", "p_of_s"}},
        **primary,
        **tad,
        "Bin size in bp": bin_size
        }

        extract_df = pd.DataFrame([row_data])
        def show_dataframe(df, icon_path):
            """
            Displays a DataFrame in a tkinter window with customized settings.
            - df: pandas DataFrame
            - icon_path: Path to the icon (e.g., icon.png)
            """

            column_renaming_map = {
                "contact_type_AA": "Contact Type AA",
                "contact_type_BB": "Contact Type BB",
                "contact_type_AB": "Contact Type AB",
                "mcm_near_ratio": "MCM Near Ratio",
                "mcm_mid_ratio": "MCM Mid Ratio",
                "mcm_far_ratio": "MCM Far Ratio",
                "pofs_slope": "P(s) Slope",
                "pofs_intercept": "P(s) Intercept",
                "pofs_r_value": "P(s) R Value",
                "pofs_p_value": "P(s) P Value",
                "pofs_std_err": "P(s) Std Err",
                "f_trans": "F Trans",
                "mean_contact_length": "Mean Contact Length",
                "std_contact_length": "STD Contact Length",
                "tad_n_tads_mean": "TAD N Tads Mean",
                "tad_mean_bin_size": "TAD Mean Bin Size",
                "tad_density_mean": "TAD Density Mean"
            }

            def rename_column(name):
                """Renames columns based on the map, defaults to original if not mapped."""
                return column_renaming_map.get(name, name)

            df = df.rename(columns=rename_column)
            df_t = df.T
            df_t.index = [rename_column(col) for col in df_t.index]

            n = len(df_t)
            mid = (n + 1) // 2
            df_first = df_t.iloc[:mid]
            df_second = df_t.iloc[mid:]
            max_len = max(len(df_first), len(df_second))

            colname1_list = []
            value1_list = []
            colname2_list = []
            value2_list = []

            for i in range(max_len):
                if i < len(df_first):
                    colname1_list.append(df_first.index[i])
                    val = df_first.iloc[i, 0]
                    if pd.notnull(val) and colname1_list[-1] == "P(s) P Value":
                        value1_list.append(f"{val:.4e}")  # for p(s) value
                    else:
                        value1_list.append(round(val, 4) if pd.notnull(val) else None)
                else:
                    colname1_list.append(None)
                    value1_list.append(None)

                if i < len(df_second):
                    colname2_list.append(df_second.index[i])
                    val = df_second.iloc[i, 0]
                    if pd.notnull(val) and colname2_list[-1] == "P(s) P Value":
                        value2_list.append(f"{val:.4e}")
                    else:
                        value2_list.append(round(val, 4) if pd.notnull(val) else None)
                else:
                    colname2_list.append(None)
                    value2_list.append(None)

            formatted_df = pd.DataFrame({
                'Metrics 1': colname1_list,
                'Values 1': value1_list,
                'Metrics 2': colname2_list,
                'Values 2': value2_list
            })

            root = tk.Tk()
            root.title("eXtract")
            window_width = 915
            window_height = 450
            root.geometry(f"{window_width}x{window_height}")

            try:
                root.iconphoto(True, tk.PhotoImage(file=icon_path))
            except Exception as e:
                print(f"Icon loading failed: {e}")

            frame = tk.Frame(root, width=window_width, height=window_height)
            frame.pack(fill='both', expand=True, padx=10, pady=10)

            pt = Table(frame, dataframe=formatted_df, editable=False)

            table_options = {
                'cellwidth': 150,
                'rowheight': 30,
                'align': 'center',
                'fontsize': 12,
                'floatprecision': 4,
                'showstatusbar': False,
                'colheadercolor': 'white',
                'rowselectedcolor': None
            }
            config.apply_options(table_options, pt)

            frame.update_idletasks()
            pt.redraw()
            pt.show()

            root.mainloop()

        show_dataframe(extract_df, '')
        return extract_df