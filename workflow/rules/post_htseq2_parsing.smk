rule make_results_dir:
    output:
        directory(config["results"])
    shell:
        "mkdir -p {output}; sleep 1"

# run post-HTseq script
rule post_htseq2_parsing:
    input:
        sample_counts=expand(Path(config["output"]["feature_count"]) / "{sample}.tsv", sample=SAMPLES),
        raw_gff_dir=Path(config["input"]["gff_refs"]),
        condition_table_path=Path(config["samples"]),
        r_dir = Path(config["results"]),
        raw_reads=Path(config["input"]["raw_reads"]),
        raw_reads_counts=Path(config["output"]["library_count"])
    output:
        done_flag = touch(Path(config["output"]["done_files"]) / "post_htseq2_parsing.done"),
    conda:
        "../envs/post_htseq2_parsing.yaml"
    script:
        "../scripts/post_htseq2_parsing_snakemake.py"
