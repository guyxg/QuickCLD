import numpy as np
import pandas as pd
import sys
import os
from prettytable import PrettyTable

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

def get_treatments(means_filename):
    df = pd.read_excel(means_filename)
    columns = df.columns
    values = df.iloc[-1]
    return [column.replace(u'\xa0', u' ').strip() for _, column in sorted(zip(values, columns), reverse=True)]

def get_sd_pairs(discoveries_filename, treatments):
    df = pd.read_excel(discoveries_filename)
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

def print_cld(treatments: list[str], insert_absorb_table: list[np.ndarray]):
    print_table = PrettyTable()
    print_table.field_names = ["Treatment", "CLD"]
    print_table.align["Treatment"] = "l"
    for i, treatment in enumerate(treatments):
        treatment_cld = ''
        for j, col in enumerate(insert_absorb_table):
            if col[i]:
                treatment_cld += chr(ord('A') + j)
        print_table.add_row([treatment, treatment_cld])
    print(print_table)

def main():
    usage = f"Usage: {os.path.basename(__file__)} [treatments.xlsx] [discoveries.xlsx]"

    argv = sys.argv
    if len(argv) != 3:
        print(usage)
        sys.exit(1)

    _, treatments_file, discoveries_file = argv
    if not (treatments_file.lower().endswith('.xlsx') and discoveries_file.lower().endswith('.xlsx')):
        print(usage)
        sys.exit(1)

    treatments = get_treatments(treatments_file)
    print(treatments)
    sd_pairs = get_sd_pairs(discoveries_file, treatments)
    insert_absorb_table = insert_absorb(len(treatments), sd_pairs)

    if len(insert_absorb_table) > 26:
        raise ValueError("Too many letters for compact letters display.")

    print_cld(treatments, insert_absorb_table)

if __name__ == '__main__':
    main()