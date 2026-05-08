import logging, os, re, tempfile
from pathlib import Path
import MDAnalysis as mda
from biopandas.pdb import PandasPdb
import pandas as pd
import numpy as np
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

class PDBChainMapper:
    """
    Handles the reversible transformation of multi-chain PDBs to single-chain PDBs.
    Maintains a mapping of (Merged Residue ID) <-> (Original Chain, Original Residue ID).
    """
    def __init__(self):
        self.mapping_df = None # Columns: [merged_res_id, orig_chain, orig_res_id, res_name]

    def generate_single_chain(self, input_pdb_path, output_pdb_path, save_map_path=None):
        """
        Reads a multi-chain PDB, filters for protein, merges to single chain 'A',
        renumbers sequentially, and saves the file.
        """
        try:
            ppdb = PandasPdb().read_pdb(str(input_pdb_path))
        except Exception as err:
            logging.error(f"Cannot read {input_pdb_path}: {err}")
            return

        df = ppdb.df["ATOM"]

        # 1. Filter Selection (Protein only, remove waters/ions)
        protein_residues = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
        ]
        df = df[df["residue_name"].isin(protein_residues)]

        # 2. Create Mapping Logic
        # We identify unique residues by (chain_id, residue_number, insertion)
        # We sort them to ensure consistent ordering
        residue_groups = df.groupby(["chain_id", "residue_number", "residue_name"], sort=False)

        mapping_data = []
        new_res_id_counter = 1

        # Helper dict to speed up updating the big dataframe later
        # key: (orig_chain, orig_res_num) -> value: new_res_num
        transform_map = {}

        # Iterate over residues in the order they appear (or sorted by chain/resnum)
        # Note: We sort groups to ensure Chain A comes before Chain B
        sorted_groups = sorted(residue_groups.groups.keys(), key=lambda x: (x[0], x[1]))

        for (chain, res_num, res_name) in sorted_groups:
            mapping_data.append({
                "merged_res_id": new_res_id_counter,
                "orig_chain": chain,
                "orig_res_id": res_num,
                "res_name": res_name
            })
            transform_map[(chain, res_num)] = new_res_id_counter
            new_res_id_counter += 1

        self.mapping_df = pd.DataFrame(mapping_data)

        if save_map_path:
            self.mapping_df.to_csv(save_map_path, index=False)

        # 3. Apply Transformation to DataFrame
        # We use a vectorized apply or map approach for speed
        def get_new_res_id(row):
            return transform_map.get((row["chain_id"], row["residue_number"]), -1)

        # Create new columns
        df["residue_number"] = df.apply(get_new_res_id, axis=1)
        df["chain_id"] = "A"

        # Clean up and sort
        df = df.sort_values(["residue_number", "atom_number"])
        ppdb.df["ATOM"] = df

        ppdb.to_pdb(path=str(output_pdb_path), records=["ATOM"], gz=False, append_newline=True)
        logging.info(f"Generated single-chain: {output_pdb_path} (Max ResID: {new_res_id_counter-1})")

    def reverse_transformation(self, merged_pdb_path, recovered_pdb_path):
        """
        Takes a single-chain PDB (e.g. from AI model), uses the stored mapping,
        and regenerates the original multi-chain structure.
        """
        if self.mapping_df is None:
            logging.error("No mapping loaded. Cannot reverse transformation.")
            return

        ppdb = PandasPdb().read_pdb(str(merged_pdb_path))
        df = ppdb.df["ATOM"]

        # Map merged_res_id back to orig_chain and orig_res_id
        # We create a dictionary for O(1) lookups
        # Key: merged_res_id -> Val: (orig_chain, orig_res_id)
        reverse_map = self.mapping_df.set_index("merged_res_id")[["orig_chain", "orig_res_id"]].to_dict('index')

        def restore_row(row):
            res_id = row["residue_number"]
            if res_id in reverse_map:
                data = reverse_map[res_id]
                return data["orig_chain"], data["orig_res_id"]
            return "A", res_id # Fallback

        # Apply restoration
        # Note: iterating rows is slow for huge files, but safe for logic.
        # For optimization, we can merge dataframes.

        # Merge approach (faster):
        # Rename residue_number to merged_res_id for joining
        df["merged_res_id"] = df["residue_number"]

        # Merge with mapping df
        df_restored = df.merge(self.mapping_df, on="merged_res_id", how="left")

        # Update columns
        df_restored["chain_id"] = df_restored["orig_chain"]
        df_restored["residue_number"] = df_restored["orig_res_id"]

        # Drop helper columns
        columns_to_keep = ppdb.df["ATOM"].columns
        df_restored = df_restored[columns_to_keep]

        ppdb.df["ATOM"] = df_restored
        ppdb.to_pdb(path=str(recovered_pdb_path), records=["ATOM"], gz=False, append_newline=True)
        logging.info(f"Reversed structure saved to: {recovered_pdb_path}")

    def get_original_info(self, merged_residue_id):
        """
        Query: Given '1223' in merged structure, return {Chain: 'B', Res: 223}
        """
        if self.mapping_df is None: return None

        row = self.mapping_df[self.mapping_df["merged_res_id"] == int(merged_residue_id)]
        if row.empty:
            return "Residue not found in map"
        return row.iloc[0].to_dict()

    def map_mutations(self, mutations_input):
        """
        Accepts a list of strings (e.g., ["A123C", "150"]) or a DataFrame.
        Returns a DataFrame with original chain/residue info.
        """
        if self.mapping_df is None: return None

        # Standardize input to a list of dicts
        parsed_muts = []

        input_iterable = []
        if isinstance(mutations_input, list):
            input_iterable = mutations_input
        elif isinstance(mutations_input, pd.DataFrame):
            # Assuming column 'mutation' or index exists
            if 'mutation' in mutations_input.columns:
                input_iterable = mutations_input['mutation'].tolist()
            else:
                input_iterable = mutations_input.index.tolist()

        for item in input_iterable:
            # Regex to find integer residue number in string like "A123C" or "123"
            match = re.search(r'(\d+)', str(item))
            if match:
                merged_id = int(match.group(1))
                # Lookup
                orig_info = self.get_original_info(merged_id)
                if isinstance(orig_info, dict):
                    parsed_muts.append({
                        "input_mutation": item,
                        "merged_res_id": merged_id,
                        "original_chain": orig_info['orig_chain'],
                        "original_res_id": orig_info['orig_res_id'],
                        "residue_name": orig_info['res_name']
                    })

        return pd.DataFrame(parsed_muts)


def pdb_single_chain_section(uploaded_file, input_format):
    st.header("🔗 PDB Single Chain Processing")
    st.info("Merge multi-chain PDBs into a single chain, or reverse the process using a saved mapping.")

    if not uploaded_file:
        st.warning(f"Please upload a {input_format} file in the sidebar first.")
        return

    mode = st.radio("Select Operation:", ["Merge to Single Chain", "Reverse to Multi-Chain"])

    if mode == "Merge to Single Chain":
        if st.button("Generate Single Chain PDB & Map"):
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_in:
                    tmp_in.write(uploaded_file.getvalue())
                    tmp_in_path = tmp_in.name

                out_pdb_path = tmp_in_path + "_single.pdb"
                out_map_path = tmp_in_path + "_map.csv"

                mapper = PDBChainMapper()
                mapper.generate_single_chain(tmp_in_path, out_pdb_path, save_map_path=out_map_path)

                with open(out_pdb_path, "rb") as f:
                    pdb_data = f.read()

                with open(out_map_path, "rb") as f:
                    map_data = f.read()

                st.success("Successfully generated single-chain PDB and mapping CSV!")

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Single Chain PDB",
                        data=pdb_data,
                        file_name=f"{uploaded_file.name.split('.')[0]}_single_chain.pdb",
                        mime="chemical/x-pdb"
                    )
                with col2:
                    st.download_button(
                        label="📥 Download Mapping CSV",
                        data=map_data,
                        file_name=f"{uploaded_file.name.split('.')[0]}_mapping.csv",
                        mime="text/csv"
                    )

                os.remove(tmp_in_path)
                if os.path.exists(out_pdb_path): os.remove(out_pdb_path)
                if os.path.exists(out_map_path): os.remove(out_map_path)


    elif mode == "Reverse to Multi-Chain":
        map_file = st.file_uploader("Upload Mapping CSV File", type=["csv"])
        if map_file:
            if st.button("Reverse to Multi-Chain PDB"):
                with st.spinner("Reversing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_in:
                        tmp_in.write(uploaded_file.getvalue())
                        tmp_in_path = tmp_in.name

                    mapper = PDBChainMapper()
                    mapper.mapping_df = pd.read_csv(map_file)

                    out_pdb_path = tmp_in_path + "_multi.pdb"
                    mapper.reverse_transformation(tmp_in_path, out_pdb_path)

                    with open(out_pdb_path, "rb") as f:
                        pdb_data = f.read()

                    st.success("Successfully reversed back to multi-chain PDB!")
                    st.download_button(
                        label="📥 Download Multi-Chain PDB",
                        data=pdb_data,
                        file_name=f"{uploaded_file.name.split('.')[0]}_multi_chain.pdb",
                        mime="chemical/x-pdb"
                    )

                    os.remove(tmp_in_path)
                    if os.path.exists(out_pdb_path): os.remove(out_pdb_path)
