rule unzip_gz:
    input:
        output_path_dict["trimmed_reads"] / "{anything}.fastq.gz",
    output:
        output_path_dict["trimmed_reads"] / "{anything}.fastq",
    benchmark:
        benchmark_dir / 'unzip' / 'unzip_gz' / '{anything}.benchmark'
    resources:
        partition = 'sched_mit_chisholm',
        mem = '10G',
        ntasks = 1,
        time = '0-12', 
        output = str(log_dir / 'unzip' / 'unzip_gz' / '{anything}.out'),
        error = str(log_dir / 'unzip' / 'unzip_gz' / '{anything}.err'),
    conda:
        "../envs/gzip.yaml"
    shell:
        "gzip -d {input}"

# rule unzip_bz2:
#     input:
#         output_path_dict["trimmed_reads"]) / "{anything}.fastq.bz2",
#     output:
#         output_path_dict["trimmed_reads"]) / "{anything}.fastq",
#     resources:
#         partition = 'sched_mit_chisholm',
#         mem = '250G',
#         ntasks = 20,
#         time = '0-12', 
#         mem_mb=100000,
#     conda:
#         "../envs/gzip.yaml"
#     shell:
#         "bzip2 -d {input}"
        