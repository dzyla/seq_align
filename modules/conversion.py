import streamlit as st
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from io import StringIO

from modules.msa import MSAError, run_msa
from modules.utils import have_params_changed, stale_notice

OUTPUT_FORMATS = ("fasta", "clustal", "phylip-relaxed", "stockholm", "nexus", "embl", "genbank")
ALIGNMENT_FORMATS = {"clustal", "phylip-relaxed", "stockholm", "nexus"}
FILE_EXTENSIONS = {"fasta": "fasta", "clustal": "aln", "phylip-relaxed": "phy", "stockholm": "sto",
                   "nexus": "nex", "embl": "embl", "genbank": "gb"}


def convert_format(sequences, output_format, seq_type="Protein"):
    """Write sequences in `output_format`. Returns (data, None) or (None, error)."""
    molecule_type = "DNA" if seq_type == "DNA" else "protein"
    records = []
    for record in sequences:
        annotations = dict(record.annotations)
        annotations.setdefault("molecule_type", molecule_type)
        name = record.id if record.name in ("", "<unknown name>") else record.name
        records.append(SeqRecord(record.seq, id=record.id, name=name,
                                 description=record.description, annotations=annotations))
    if output_format in ALIGNMENT_FORMATS and len({len(r.seq) for r in records}) > 1:
        return None, (f"{output_format} requires aligned sequences of equal length. "
                      "Enable 'Align with FAMSA before converting' or export from the MSA tool.")
    try:
        output = StringIO()
        SeqIO.write(records, output, output_format)
        return output.getvalue(), None
    except Exception as e:
        return None, f"Error during format conversion: {e}"


def format_conversion_section(sequences, input_format, seq_type="Protein"):
    """Handles the sequence format conversion workflow."""
    st.header("🔄 Sequence Format Conversion")
    st.info("Convert your sequences between different file formats.")

    output_format = st.selectbox("Select Output Format", OUTPUT_FORMATS,
                                 help="Choose the desired format for the converted sequences")
    align_first = False
    if output_format in ALIGNMENT_FORMATS:
        align_first = st.checkbox("Align with FAMSA before converting", value=True,
                                  help="Alignment formats need sequences of equal length.")

    current_params = {
        'sequences': tuple((s.id, str(s.seq)) for s in sequences),
        'input_format': input_format,
        'output_format': output_format,
        'seq_type': seq_type,
        'align_first': align_first,
    }
    params_changed = have_params_changed(current_params, 'last_conversion_params')

    if st.button("Convert Format"):
        with st.spinner("Converting format..."):
            records = sequences
            error = None
            if align_first:
                try:
                    aligned = run_msa([(s.id, str(s.seq)) for s in sequences], seq_type)
                    records = [SeqRecord(Seq(seq), id=seq_id, description="") for seq_id, seq in aligned]
                except MSAError as e:
                    error = str(e)
            if error:
                st.session_state.converted_data, st.session_state.conversion_error = None, error
            else:
                st.session_state.converted_data, st.session_state.conversion_error = convert_format(
                    records, output_format, seq_type)
            st.session_state.last_conversion_params = current_params
        params_changed = False

    if params_changed:
        if st.session_state.get('converted_data') or st.session_state.get('conversion_error'):
            stale_notice()
        return
    if st.session_state.get('converted_data'):
        st.success("Format conversion successful.")
        st.text(st.session_state.converted_data)
        st.download_button(
            label="📥 Download Converted File",
            data=st.session_state.converted_data,
            file_name=f"converted_sequences.{FILE_EXTENSIONS[output_format]}",
            mime="text/plain", on_click="ignore")
    elif st.session_state.get('conversion_error'):
        st.error(f"Format conversion failed: {st.session_state.conversion_error}")
