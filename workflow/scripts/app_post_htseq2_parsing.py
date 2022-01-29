from pathlib import Path
import shutil
from matplotlib.colors import same_color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
import gffpandas.gffpandas as gffpd
import math

# rpy2 imports
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.conversion import localconverter
base = importr('base')
utils = importr('utils')
deseq2 = importr('DESeq2')

import logging as log

log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

def makeOutDir(outputDir, folderName):
    """
    Makes an out directory if there is not one. Returns file path as Path object
    Inputs:
    outputDir - Pathlib directory object that directory will be created within
    folderName - Name of directory to be made/returned
    """
    outdir = outputDir / folderName
    if not outdir.exists():
        outdir.mkdir()
    return outdir

def get_htseq2_and_metadata_df(htseq2dir, organism_type_ID_df, feature_types_to_keep=None):
    raw_counts_df_list = []
    raw_metadata_df_list = []
    first_unique_ids = []

    for i, path in enumerate(sorted(list(htseq2dir.iterdir()))):
        if path.suffix == '.tsv':
            # get sample ID from path
            sample_name = path.stem

            # read in HTseq TSV
            temp_df = pd.read_csv(path, sep='\t', names=['long_ID', sample_name])

            # check that long_IDs match
            if len(first_unique_ids) == 0:
                first_unique_ids = temp_df['long_ID'].unique()
            else:
                temp_unique_ids = temp_df['long_ID'].unique()
                assert first_unique_ids.all() == temp_unique_ids.all()

            temp_df = temp_df.set_index('long_ID')

            temp_metadata_df = temp_df[temp_df.index.str.contains('__')]

            temp_counts_df = temp_df[~temp_df.index.str.contains('__')]

            # append df to raw_counts_df_list
            raw_counts_df_list.append(temp_counts_df)
            raw_metadata_df_list.append(temp_metadata_df)

    counts_df = pd.concat(raw_counts_df_list, axis=1)
    counts_df = counts_df.add_prefix('sample_')

    metadata_df = pd.concat(raw_metadata_df_list, axis=1)
    metadata_df = metadata_df.add_prefix('sample_')
    metadata_df.index = metadata_df.index.str.replace('__', '')

    counts_df = counts_df.join(organism_type_ID_df)
    counts_df = counts_df.set_index(['organism', 'type'], append=True)
    counts_df = counts_df.reorder_levels(['organism', 'type', 'long_ID'])

    if feature_types_to_keep:
        counts_df = counts_df[counts_df.index.get_level_values('type').isin(feature_types_to_keep)]
    
    feature_df = counts_df.groupby(['type']).sum() 
    metadata_df = pd.concat([feature_df, metadata_df])

    return metadata_df, counts_df

def get_dge_table(results_table):

        def symlog10(x):
            if x > 0:
                return math.log10(x+1)
            elif x < 0:
                return -math.log10(-x+1)
            else:
                return 0
            
        results_table['symlog10baseMean'] = results_table['baseMean'].apply(lambda x: symlog10(x))

        cols = list(results_table.columns.values)
        
        index_of_baseMean = cols.index('baseMean')
        cols.remove('symlog10baseMean')
        cols.insert(index_of_baseMean+1, 'symlog10baseMean')

        results_table = results_table[cols]

        return results_table

def main(results_dir, htseq2dir, gff_dir, condition_table_path, raw_reads_dir, feature_types_to_keep=None):

    results_dir.mkdir(exist_ok=True)
    app_dfs_dir = makeOutDir(results_dir, 'app_dfs')
    shutil.copy(condition_table_path, app_dfs_dir / condition_table_path.name)
    
    # attributes and annotations

    gffs = []
    for gff_path in gff_dir.iterdir():
        organism_name = str(gff_path.stem)
        annotation = gffpd.read_gff3(gff_path)
        attributes_df = annotation.attributes_to_columns()
        attributes_df['organism'] = organism_name
        gffs.append(attributes_df)

    attributes_df = pd.concat(gffs)

    attributes_df = attributes_df.rename(columns={'ID':'long_ID'})

    attributes_df = attributes_df.set_index('long_ID')

    organism_type_ID_df = attributes_df[['organism', 'type']]

    metadata_df, counts_df = get_htseq2_and_metadata_df(htseq2dir, organism_type_ID_df, feature_types_to_keep)
    
    counts_df.to_csv(app_dfs_dir / "counts.tsv", sep="\t")
    metadata_df.to_csv(app_dfs_dir / 'metadata.tsv', sep='\t')

    # # pct of genes with >1 read for each organism
    # organism_sample_gene_mapping_pct_df = pd.DataFrame(columns=counts_df.columns)
    # count_df_gene_sampled_bool = counts_df >= 1
    # count_df_gene_sampled_all = counts_df >= 0

    # organism_sample_gene_mapping_pct_df = count_df_gene_sampled_bool.groupby('organism').sum() / count_df_gene_sampled_all.groupby('organism').sum()

    # comparisons and DEseq2
    conditions_df =  pd.read_csv(condition_table_path, sep='\t', index_col='sample_name')

    # extracts columns that include the word "comparison"
    comparisons_include_df = conditions_df[conditions_df.columns[np.array(conditions_df.columns.str.contains('comparison'))]]
    
    # creates new dataframe from comparisons_include_df that replaces control/treatment with True in order to subset future datasets
    conditions_df_sans_comparisons = conditions_df[conditions_df.columns[~np.array(conditions_df.columns.str.contains('comparison'))]]

    attributes_df = attributes_df.set_index(['organism', 'type'], append=True)
    attributes_df = attributes_df.reorder_levels(['organism', 'type', 'long_ID'])
    attributes_df.columns = pd.MultiIndex.from_product([['metadata'], attributes_df.columns])

    if feature_types_to_keep:
        attributes_df = attributes_df[attributes_df.index.get_level_values('type').isin(feature_types_to_keep)]
    attributes_df = attributes_df.reset_index(level='type', drop=True)
    
    attributes_df = attributes_df.loc[~attributes_df.index.duplicated(keep='first')]
    counts_df = counts_df.loc[~counts_df.index.duplicated(keep='first')]

    log.debug(f"Attributes df:\n{attributes_df}")

    results_dfs = [attributes_df]
    rlog_dfs = []

    
    for comparison in comparisons_include_df.columns:
        
        log.info(comparison)

        included_samples_series = comparisons_include_df[comparisons_include_df[comparison].notna()][comparison]

        conditions = list(included_samples_series.unique())
        
        return_results_df = len(conditions) == 2 and 'control' in conditions and 'treatment' in conditions

        log.info(f"conditions {return_results_df} for {conditions}")

        comparison_condition_df = conditions_df_sans_comparisons.loc[included_samples_series.index]
        comparison_condition_df[comparison] = included_samples_series

        comparison_samples = included_samples_series.index.values 

        comparison_counts_df = counts_df[[f"sample_{s:02d}" for s in comparison_samples]]

        comparison_counts_df = comparison_counts_df.reset_index(level='type', drop=True)

        for organism in comparison_counts_df.index.unique(0):
            log.info(f'{organism}\n')

            comparison_organism_count_df = comparison_counts_df.loc[organism, :]

            # organism must be present in all samples. Good test to tell if it is from experience...
            # log.info('{} - organism lowest sample read mapping in comparison: {}'.format(organism, min(comparison_organism_count_df.mean())))
            if min(comparison_organism_count_df.mean()) > 1:

                # DEseq process
                # 1a. transfer count df into r df
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    r_comparison_organism_count_df = robjects.conversion.py2rpy(comparison_organism_count_df)

                robjects.globalenv['r_comparison_organism_count_df'] = r_comparison_organism_count_df
                
                # 1b. transfer condition df into r df
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    r_comparison_condition_df = robjects.conversion.py2rpy(comparison_condition_df)
                robjects.globalenv['r_comparison_condition_df'] = r_comparison_condition_df
                
                # 2. create DESeqDataSet object
                # exp design
                robjects.r(f"""dds <- DESeqDataSetFromMatrix(countData = r_comparison_organism_count_df, 
                                                            colData = r_comparison_condition_df, 
                                                            design = ~ {comparison})""")
                
                # 3. run DEseq command
                dds_processed = robjects.r("DESeq(dds)")
                robjects.globalenv['dds_processed'] = dds_processed

                # get normalized counts df
                r_normalized_counts_df = robjects.r("counts(dds_processed, normalized=TRUE)")
                
                if return_results_df:
                    # 4a. set up comparison controls and treatments
                    contrast_string_list = robjects.StrVector([comparison, 'control', 'treatment'])

                    # 4b. get results df
                    r_results = deseq2.results(dds_processed, contrast = contrast_string_list)
                    robjects.globalenv['r_results'] = r_results
                    r_results_df = robjects.r("as.data.frame(r_results)")

                # 5. get rlog and vsd dfs
                rlog_output = deseq2.rlog(dds_processed, blind=False)
                robjects.globalenv['rlog_output'] = rlog_output
                r_rlog_df = robjects.r("assay(rlog_output)")

                robjects.r("vst_output <- varianceStabilizingTransformation(dds, blind=FALSE)")
                r_vst_df = robjects.r("assay(vst_output)")
                
                # 6. transfer normalized counts, rlog, and vst df to pandas
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    rlog_array = robjects.conversion.rpy2py(r_rlog_df)
                
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    vst_array = robjects.conversion.rpy2py(r_vst_df)
                
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    normalized_counts_array = robjects.conversion.rpy2py(r_normalized_counts_df)

                if return_results_df:
                    # 7. transfer results df to pandas
                    with localconverter(robjects.default_converter + pandas2ri.converter):
                        results_table = robjects.conversion.rpy2py(r_results_df)
                        
                    results_table = results_table.rename({'seq_id':'long_ID'})
                    results_table['long_ID'] = comparison_organism_count_df.index.values
                    results_table = results_table.set_index('long_ID')

                normalized_counts_df = pd.DataFrame(normalized_counts_array, index=comparison_organism_count_df.index, columns=included_samples_series.index.values)
                rlog_df = pd.DataFrame(rlog_array, index=comparison_organism_count_df.index, columns=included_samples_series.index.values)
                vst_df = pd.DataFrame(vst_array, index=comparison_organism_count_df.index, columns=included_samples_series.index.values)
                

                # post-DEseq2 analysis

                if return_results_df:

                    all_dge_table = get_dge_table(results_table)
                    
                    #results df
                    multiindex_results_df = results_table
                    multiindex_results_df.columns = pd.MultiIndex.from_product([[comparison], results_table.columns])
                    multiindex_results_df.index = pd.MultiIndex.from_product([[organism], results_table.index])
                    log.debug(f"multiindex_results_df:\n{multiindex_results_df}")
                    # log.debug(f"multiindex_results_df.shape before filter:\n{multiindex_results_df.shape}")
                    # multiindex_results_df = multiindex_results_df.loc[~multiindex_results_df.index.duplicated(keep='first')]
                    # log.debug(f"multiindex_results_df.shape after filter:\n{multiindex_results_df.shape}")
                    results_dfs.append(multiindex_results_df)

                    #rlog df
                    multiindex_rlog_df = rlog_df
                    multiindex_rlog_df.columns = pd.MultiIndex.from_product([[organism], [comparison], rlog_df.columns])
                    multiindex_rlog_df.index = pd.MultiIndex.from_product([[organism], rlog_df.index])
                    log.debug(f"multiindex_rlog_df:\n{multiindex_rlog_df}")
                    rlog_dfs.append(multiindex_rlog_df)

    app_results_df = pd.concat(results_dfs, axis=1)
    app_rlog_df = pd.concat(rlog_dfs, axis=1)

    app_results_df.to_csv(app_dfs_dir / 'results_df.tsv', sep='\t')
    app_rlog_df.to_csv(app_dfs_dir / 'rlog_df.tsv', sep='\t')
