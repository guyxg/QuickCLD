import pandas as pd
import numpy as np
import sys
import os
from prettytable import PrettyTable
from math import comb

def get_categories_and_treatments(df: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    """
    Returns a list of categories and a list of lists of treatments,
    each list corresponds to a category and ordered by means in each
    category.
    """
    categories = [category.replace(u'\xa0', u' ').strip() for category in df.index]
    treatments_list = []
    columns = [col for col in df.columns if col[1] == 'mean']
    names = [col[0].replace(u'\xa0', u' ').strip() for col in columns]

    def apply_func(row):
        means = [row[col] for col in columns]
        treatments_list.append([column for _, column in sorted(zip(means, names), reverse=True)])

    df.apply(apply_func, axis=1)
    return categories, treatments_list

def split_discoveries(df: pd.DataFrame) -> list[pd.DataFrame]:
    df_list = []
    category_header_indexes = np.where(df.isna().any(axis=1) & df.notna().any(axis=1))[0]
    for i, j in zip(category_header_indexes, category_header_indexes[1:]):
        cur_df = df[i:j]
        df_list.append(cur_df.dropna(how="any"))
    df_list.append(df[category_header_indexes[-1]:].dropna(how="any"))
    return df_list

def get_sd_pairs(df: pd.DataFrame, treatments: list[str]):
    df.rename(lambda col: col.replace('\xa0', ' ').strip(), axis='columns', inplace=True)
    df['Original FDR method of Benjamini and Hochberg'] = df['Original FDR method of Benjamini and Hochberg'].apply(lambda x: str(x).replace(u'\xa0', u' '))
    result = []

    def apply_func(row):
        if row['Discovery?'] == 'Yes':
            row_val = row['Original FDR method of Benjamini and Hochberg']
            row_val = row_val.replace('\xa0', u' ')
            t1, t2 = row_val.split(" vs. ")
            t1, t2 = t1.strip(), t2.strip()
            result.append((treatments.index(t1), treatments.index(t2)))
    df.apply(apply_func, axis=1)

    return result

def insert_absorb(n_treatments, sd_pairs):
    table = [np.full(n_treatments, True)]
    for i, j in sd_pairs:
        indexes_to_delete = []
        for col_idx, col in enumerate(table):
            if col[i] and col[j]:
                indexes_to_delete.append(col_idx)
                for k in [i, j]:
                    # Insert
                    new_col = col.copy()
                    new_col[k] = False
                    to_insert = True
                    # Absorb
                    for old_col_idx, old_col in enumerate(table):
                        if old_col_idx not in indexes_to_delete and np.array_equal(np.bitwise_and(old_col, new_col), new_col):
                            to_insert = False
                            break
                    if to_insert:
                        table.append(new_col)
        for idx in reversed(indexes_to_delete):
            del table[idx]


    def sort_key(col: np.ndarray):
        mask = np.where(col == True)[0]
        if len(mask) == 0:
            return len(col)
        else:
            return mask[0]

    return sorted(table, key=sort_key)

def get_cld_table(treatments: list[str], insert_absorb_table: list[np.ndarray]):
    print_table = PrettyTable()
    print_table.field_names = ["Name", "CLD"]
    print_table.align["Name"] = "l"
    for i, treatment in enumerate(treatments):
        treatment_cld = ''
        for j, col in enumerate(insert_absorb_table):
            if col[i]:
                treatment_cld += chr(ord('A') + j)
        print_table.add_row([treatment, treatment_cld])
    return print_table

def generate_cld_from_excel(filename):
    df = pd.read_excel(filename, header=[0, 1], sheet_name=0, index_col=0)
    categories, treatments_list = get_categories_and_treatments(df)

    df = pd.read_excel(filename, sheet_name=1)
    discoveries_dfs = split_discoveries(df)

    # make sure no row was dropped in discoveries
    n_treatments = len(treatments_list[0])
    for df in discoveries_dfs:
        assert len(df) == comb(n_treatments, 2)

    table = PrettyTable()
    for i, category in enumerate(categories):
        treatments = treatments_list[i]
        discoveries_df = discoveries_dfs[i]
        letters_table = insert_absorb(len(treatments), get_sd_pairs(discoveries_df, treatments))
        cld_table = get_cld_table(treatments, letters_table)
        cld_table.title = category
        print(cld_table)

def main():
    usage = f"Usage: {os.path.basename(__file__)} [source_file.xlsx]"

    argv = sys.argv
    if len(argv) != 2:
        print(usage)
        sys.exit(1)

    source_file = argv[1]
    if not source_file.lower().endswith('.xlsx'):
        print(usage)
        sys.exit(1)

    generate_cld_from_excel(source_file)

if __name__ == '__main__':
    main()