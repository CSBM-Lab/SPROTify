from contextlib import contextmanager
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def read_fasta(input_path):
    """
    Read a FASTA file and yield (sequence_id, sequence).
    """
    with open(input_path, 'r') as input_file:
        seq_id = None
        sequence = []
        
        for line in input_file:
            line = line.strip()
            if line.startswith('>'):
                if seq_id:
                    yield (seq_id, ''.join(sequence))
                seq_id = line[1:]
                sequence = []
            else:
                sequence.append(line)
        
        if seq_id:
            yield (seq_id, ''.join(sequence))


def write_fasta(sequences, output_path):
    """
    Write (sequence_id, sequence) pairs to a FASTA file.
    """
    with open(output_path, 'w') as output_file:
        for seq_id, seq in sequences:
            output_file.write(f'>{seq_id}\n{seq}\n')



@contextmanager
def open_input_output(input_path, output_path):
    """
    Open input and output files as a context manager.
    """
    input_file = open(input_path, 'r')
    output_file = open(output_path, 'w')
    try:
        yield input_file, output_file 
    finally:
        input_file.close()
        output_file.close()

def get_file_path(relative_path):
    """
    Convert a path relative to the project root into an absolute path.
    """
    return os.path.abspath(os.path.join(project_root, relative_path))
