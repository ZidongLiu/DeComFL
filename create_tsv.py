import pandas as pd
import glob

file_paths = glob.glob("data/gen_TSV_logs/train/*.tsv")

df_list = [pd.read_csv(file, sep="\t") for file in file_paths]
combined_df = pd.concat(df_list, ignore_index=True)

combined_df.to_csv("train.tsv", sep="\t", index=False)
