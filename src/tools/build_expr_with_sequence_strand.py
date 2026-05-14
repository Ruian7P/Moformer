#!/usr/bin/env python
"""Add strand-aware hg38 promoter sequences to the Xpresso expression table.

The output reproduces the project table:
  data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv

For each gene, promoter_2k is extracted as a transcription-oriented 2 kb
window from 1500 bp upstream to 500 bp downstream of the TSS. For minus-strand
genes, the genomic interval is reverse-complemented so the sequence is always
oriented 5' to 3' relative to the transcript.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyfaidx
from tqdm import tqdm

_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(_RC)[::-1].upper()


class FastaExtractor:
    def __init__(self, fasta_path: str):
        self.fasta = pyfaidx.Fasta(fasta_path)
        self.chrom_sizes = {str(k): len(v) for k, v in self.fasta.items()}

    def resolve_chrom(self, chrom) -> str:
        c = str(chrom)
        candidates = [c]
        if c.startswith("chr"):
            candidates.append(c[3:])
        else:
            candidates.append("chr" + c)
        for x in candidates:
            if x in self.chrom_sizes:
                return x
        raise KeyError(f"Chromosome {chrom!r} not found in FASTA. Tried {candidates}")

    def extract(self, chrom, start0: int, end0: int) -> str:
        """Extract a 0-based half-open interval, padding out-of-bound bases with N."""
        chrom = self.resolve_chrom(chrom)
        chrom_len = self.chrom_sizes[chrom]
        left_pad = "N" * max(0, -start0)
        right_pad = "N" * max(0, end0 - chrom_len)
        s = max(0, start0)
        e = min(chrom_len, end0)
        if e <= s:
            seq = ""
        else:
            # pyfaidx uses 1-based inclusive coordinates.
            seq = str(self.fasta.get_seq(chrom, s + 1, e).seq).upper()
        return left_pad + seq + right_pad

    def close(self):
        self.fasta.close()


def parse_args():
    p = argparse.ArgumentParser(description="Build expression table with strand-aware promoter_2k sequences.")
    p.add_argument("--input", default="data/GM12878_K562_18377_gene_expr_fromXpresso.csv")
    p.add_argument("--fasta", default="/home/ruian7p/Projects/puffin/resources/hg38.fa")
    p.add_argument("--output", default="data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv")
    p.add_argument("--upstream", type=int, default=1500)
    p.add_argument("--downstream", type=int, default=500)
    p.add_argument("--gene-id-col", default="gene_id")
    p.add_argument("--chrom-col", default="chrom")
    p.add_argument("--tss-col", default="TSS_xpresso")
    p.add_argument("--strand-col", default="strand")
    p.add_argument(
        "--hk-list",
        default="",
        help="Optional one-column text/CSV file of housekeeping gene IDs. If omitted, is_hk is set to False.",
    )
    return p.parse_args()


def load_housekeeping_ids(path: str) -> set[str]:
    if not path:
        return set()
    df = pd.read_csv(path, sep=None, engine="python", header=None)
    return set(df.iloc[:, 0].astype(str))


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    hk_ids = load_housekeeping_ids(args.hk_list)
    fasta = FastaExtractor(args.fasta)

    seqs = []
    starts = []
    ends = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract promoter_2k"):
        tss = int(row[args.tss_col])
        strand = str(row[args.strand_col])
        chrom = row[args.chrom_col]

        if strand == "-":
            start0 = tss - args.downstream
            end0 = tss + args.upstream
            seq = revcomp(fasta.extract(chrom, start0, end0))
        else:
            start0 = tss - args.upstream
            end0 = tss + args.downstream
            seq = fasta.extract(chrom, start0, end0).upper()

        seqs.append(seq)
        # Keep the same 20 kb coordinate convention present in the base table if
        # start/end already exist; these are just useful sanity fields otherwise.
        starts.append(int(row["start"]) if "start" in df.columns else start0)
        ends.append(int(row["end"]) if "end" in df.columns else end0)

    fasta.close()

    if "is_hk" not in df.columns:
        if hk_ids:
            df["is_hk"] = df[args.gene_id_col].astype(str).isin(hk_ids)
        else:
            df["is_hk"] = False
    df["promoter_2k"] = seqs

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"saved: {args.output}")
    print(f"n_genes: {len(df)}")
    print(f"promoter length min/max: {min(map(len, seqs))}/{max(map(len, seqs))}")


if __name__ == "__main__":
    main()
