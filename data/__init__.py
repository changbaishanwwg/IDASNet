"""Shared input-normalization settings for training and inference."""

NORMALIZATION = {
    'axis': 'wavelength',
    'per_sample_and_channel': True,
    'std_correction': 1,
    'epsilon': 1e-08,
    'before_input_selection': True,
}
