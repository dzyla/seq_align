import streamlit as st
from Bio import SeqIO
from io import StringIO
from modules.utils import have_params_changed

def format_conversion_section(sequences, input_format):
    """
    Handles the sequence format conversion workflow.

    Parameters:
        sequences: List of sequence records to convert
        input_format (str): Original format of the sequences
    """
    st.header("🔄 Sequence Format Conversion")
    st.info(
        "Convert your sequences between different file formats. "
        "This is useful for compatibility with different analysis abilities."
    )

    conversion_output_format = st.selectbox(
        "Select Output Format",
        ("fasta", "clustal", "phylip", "embl", "genbank"),
        help="Choose the desired format for the converted sequences"
    )

    # Create a hash of the current parameters
    current_params = {
        'sequences_ids': tuple(seq.id for seq in sequences),
        'input_format': input_format,
        'output_format': conversion_output_format
    }

    # Check if parameters have changed
    params_changed = have_params_changed(current_params, 'last_conversion_params')

    convert_button = st.button("Convert Format")

    # Use session state to store conversion results
    if 'converted_data' not in st.session_state:
        st.session_state.converted_data = None
        st.session_state.conversion_error = None

    # Only run conversion if button is clicked or we have results and parameters haven't changed
    if convert_button or (st.session_state.converted_data is not None and not params_changed):
        # Only recalculate if parameters changed or explicitly requested
        if convert_button or params_changed:
            if input_format.lower() == "newick" and conversion_output_format.lower() != "newick":
                st.warning("Newick format is a tree format and cannot be converted to sequence formats directly.")
                st.session_state.converted_data = None
                st.session_state.conversion_error = "Incompatible formats"
            else:
                with st.spinner("Converting format..."):
                    converted_data, error = convert_format(sequences, conversion_output_format)
                    st.session_state.converted_data = converted_data
                    st.session_state.conversion_error = error
                    st.session_state.last_conversion_params = current_params

        # Display results if available
        if st.session_state.converted_data:
            st.success("Format conversion successful.")
            st.text(st.session_state.converted_data)
            st.download_button(
                label="📥 Download Converted File",
                data=st.session_state.converted_data,
                file_name=f"converted_sequences.{conversion_output_format}",
                mime=f"text/{conversion_output_format}"
            )
        elif st.session_state.conversion_error:
            st.error(f"Format conversion failed: {st.session_state.conversion_error}")

def convert_format(sequences, output_format):
    """
    Convert sequences from the current format to the desired output format.

    Parameters:
        sequences: List of sequence records to convert
        output_format (str): Target format for conversion

    Returns:
        tuple: (converted_data, error) - Converted data as string and error message or None
    """
    try:
        output = StringIO()
        SeqIO.write(sequences, output, output_format)
        converted_data = output.getvalue()
        return converted_data, None
    except Exception as e:
        return None, f"Error during format conversion: {e}"
