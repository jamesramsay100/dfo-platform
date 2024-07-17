import h5py
import numpy as np


def create_synthetic_das_dataset(input_file, output_file):
    # Open the input file
    with h5py.File(input_file, 'r') as f:
        # Read the original data
        original_data = f['DAS'][:]

        # Extract metadata
        pulse_rate = f.get('DAS').attrs['Rate'][0]

    # Extract the 3000-4000 index section
    section_data = original_data[:, 3000:4000]

    # Repeat the section 4 times to create a 2-minute duration
    repeated_data = np.tile(section_data, (4, 1))

    # Extract the quiet section (3900-4000 index)
    quiet_section = original_data[:, 3900:4000]

    # Repeat the quiet section 10 times in the depth dimension to match the 1000 channels
    repeated_quiet_section = np.tile(quiet_section, (1, 10))

    # Repeat the expanded quiet section 4 times in time to match the 2-minute duration
    repeated_quiet_section = np.tile(repeated_quiet_section, (4, 1))

    # Create a sinusoidal fading function that completes a full cycle (out and in) over 2 minutes
    fade_length = repeated_data.shape[0]
    fade_function = 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, 1, fade_length)))

    # Apply fading to both the original section and the quiet section
    faded_original_section = repeated_data * fade_function[:, np.newaxis]
    faded_quiet_section = repeated_quiet_section * (1 - fade_function[:, np.newaxis])

    # Combine the faded original data with the faded quiet section
    synthetic_data = faded_original_section + faded_quiet_section

    # Create the output file
    with h5py.File(output_file, 'w') as f_out:
        # Copy metadata from the original file
        with h5py.File(input_file, 'r') as f_in:
            for key in f_in.keys():
                if key != 'DAS':
                    f_in.copy(key, f_out)

        # Update relevant metadata
        f_out.create_dataset('DAS', data=synthetic_data, dtype='int16')
        f_out['DAS'].attrs['Rate'] = np.array([pulse_rate])  # Store as a single-element array

        if 'Metadata/TraceCount' in f_out:
            f_out['Metadata/TraceCount'][()] = synthetic_data.shape[0]
        else:
            print("Warning: Unable to update TraceCount in output file.")

    print(f"Synthetic dataset created and saved to {output_file}")


# Usage
input_file = '/Users/jamesramsay/Downloads/OneDrive_1_09-07-2024/0000000005_2024-07-03_09.30.30.84400.hdf5'
output_file = '/Users/jamesramsay/Downloads/OneDrive_1_09-07-2024/delme.hdf5'
create_synthetic_das_dataset(input_file, output_file)