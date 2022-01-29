from app_post_htseq2_parsing import main
from pathlib import Path
import yaml

feature_types_to_keep=[
'CDS',
'rRNA',
'tRNA',
'sRNA'
# 'RNase_P_RNA',
# 'pseudogene',
# 'region',
# 'riboswitch',
# 'tmRNA',
]

with open('../../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

results_dir = Path(config["results"])
htseq2dir = Path(config["output"]["feature_count"])
raw_gff_dir = Path(config["input"]["gff_refs"])
condition_table_path = Path('../../config/samples.tsv')
raw_reads_dir = Path(config["input"]["raw_reads"])

print("results_dir: {}".format(results_dir))
print("htseq2dir: {}".format(htseq2dir))
print("raw_gff_dir: {}".format(raw_gff_dir))
print("condition_table_path: {}".format(condition_table_path))

main(results_dir, htseq2dir, raw_gff_dir, condition_table_path, raw_reads_dir, feature_types_to_keep)