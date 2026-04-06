import functools
import math
import mmh3
import numba as nb
import numpy as np
from spectrum_utils.spectrum import MsmsSpectrum
import logging
from ann_solo.config import config
import copy
import re

# ==============================================================================
# Constants & Masses
# ==============================================================================
PROTON_MASS = 1.00784
H2O_MASS = 18.01056
AMINO_ACID_MASS = {
    "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
    # Static & Dynamic Modifications
    "1": 103.00919 + 57.021,  # Carbamidomethyl C
    "2": 114.04293 + 0.984,   # Deamidation N
    "3": 42.011,              # N-term Acetylation (or specific n)
    "4": 43.005,              # Other n modification
    "5": 131.04049 + 15.995,  # Oxidation M
    "6": 128.05858 + 0.984    # Deamidation Q
}

# ==============================================================================
# Basic Spectrum Processing Functions
# ==============================================================================

@nb.njit
def _check_spectrum_valid(spectrum_mz: np.ndarray, min_peaks: int, min_mz_range: float) -> bool:
    """Check whether a spectrum has enough peaks and covers a wide enough mass range."""
    return (len(spectrum_mz) >= min_peaks and spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range)

@nb.njit
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """Normalize spectrum peak intensities using L2 norm."""
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)

def process_spectrum(spectrum: MsmsSpectrum, is_library: bool) -> MsmsSpectrum:
    """Process the MS/MS spectrum according to configuration settings (filtering, scaling, normalizing)."""
    if spectrum.is_processed:
        return spectrum

    min_peaks = config.min_peaks
    min_mz_range = config.min_mz_range

    spectrum = spectrum.set_mz_range(config.min_mz, config.max_mz)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    if config.resolution is not None:
        spectrum = spectrum.round(config.resolution, 'sum')
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    if config.remove_precursor:
        spectrum = spectrum.remove_precursor_peak(config.remove_precursor_tolerance, 'Da', 2)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    max_peaks = config.max_peaks_used_library if is_library else config.max_peaks_used
    spectrum = spectrum.filter_intensity(config.min_intensity, max_peaks)
    
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    scaling = 'root' if config.scaling == 'sqrt' else config.scaling
    if scaling is not None:
        spectrum = spectrum.scale_intensity(scaling, max_rank=max_peaks)

    spectrum.intensity = _norm_intensity(spectrum.intensity)
    spectrum.is_valid = True
    spectrum.is_processed = True

    return spectrum

@functools.lru_cache(maxsize=None)
def get_dim(min_mz, max_mz, bin_size):
    """Compute vector dimensions based on mz range and bin size."""
    min_mz, max_mz = float(min_mz), float(max_mz)
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    return round((end_dim - start_dim) / bin_size), start_dim, end_dim

@functools.lru_cache(maxsize=None)
def hash_idx(bin_idx: int, hash_len: int) -> int:
    """Hash an integer index using MurmurHash3 to fall within vector dimensions."""
    return mmh3.hash(str(bin_idx), 42, signed=False) % hash_len

# ==============================================================================
# Theoretical & Complementary Spectrum Generation
# ==============================================================================

def generate_theoretical_spectrum(sequence):
    """Generate theoretical b and y ion m/z values for a given peptide sequence."""
    b_ions, y_ions = [], []
    
    # Calculate b-ions (N-terminus to C-terminus)
    b_mass = 0
    for aa in sequence:
        b_mass += AMINO_ACID_MASS.get(aa, 0)
        b_ions.append(b_mass + PROTON_MASS)

    # Calculate y-ions (C-terminus to N-terminus)
    y_mass = H2O_MASS
    for aa in reversed(sequence):
        y_mass += AMINO_ACID_MASS.get(aa, 0)
        y_ions.append(y_mass + PROTON_MASS)

    mz = b_ions + y_ions
    intensity = [1.0] * len(mz) # Fixed intensity for theoretical peaks
    return mz, intensity

def get_spectrum_weight(spectrum):
    """Downweight peaks located in the upper half of the precursor mass range."""
    new_spectrum = []
    half_precursor = 0.5 * (spectrum.precursor_mz * spectrum.precursor_charge)
    
    for mz, intensity in zip(spectrum.mz, spectrum.intensity):
        weight = 0.5 if mz > half_precursor else 1.0
        new_spectrum.append([mz, intensity * weight])
    return new_spectrum

def get_complimentary_spectrum_half(spectrum):
    """Generate complementary peaks without weighting."""
    new_spectrum = []
    precursor_mass_calc = (spectrum.precursor_mz * spectrum.precursor_charge) - (spectrum.precursor_charge * PROTON_MASS)
    
    for mz, intensity in zip(spectrum.mz, spectrum.intensity):
        ion_mass_calc = mz - PROTON_MASS
        complimentary_peak = precursor_mass_calc - ion_mass_calc + PROTON_MASS
        
        # Append complementary peak maintaining original intensity
        new_spectrum.append([complimentary_peak, intensity])
        
    new_spectrum.sort(key=lambda x: x[0])
    return new_spectrum

def get_complimentary_spectrum_weight(spectrum):
    """Generate complementary peaks and downweight if the complementary peak is in the upper half of precursor mass."""
    new_spectrum = []
    half_precursor = 0.5 * (spectrum.precursor_mz * spectrum.precursor_charge)
    precursor_mass_calc = (spectrum.precursor_mz * spectrum.precursor_charge) - (spectrum.precursor_charge * PROTON_MASS)
    
    for mz, intensity in zip(spectrum.mz, spectrum.intensity):
        ion_mass_calc = mz - PROTON_MASS
        complimentary_peak = precursor_mass_calc - ion_mass_calc + PROTON_MASS
        
        # Apply downweighting if complementary peak is > 50% of precursor mass
        weight = 0.5 if complimentary_peak > half_precursor else 1.0
        
        new_spectrum.append([mz, intensity])
        new_spectrum.append([complimentary_peak, intensity * weight])
        
    new_spectrum.sort(key=lambda x: x[0])
    return new_spectrum

def _parse_sequence_for_vector(peptide_seq: str) -> str:
    """Helper function to parse library peptide sequences for theoretical generation."""
    seq = peptide_seq
    seq = seq.replace("C[160]", "1").replace("N[115]", "2")
    seq = seq.replace("n[43]", "3").replace("n[44]", "4")
    seq = seq.replace("M[147]", "5").replace("Q[129]", "6")
    return seq

# ==============================================================================
# Vectorization Modules (Dual Embedding Strategy)
# ==============================================================================

def _build_concatenated_vector(exp_peaks, seq_peaks, min_mz, max_mz, bin_size, hash_len, norm):
    """Helper method to construct the dual-embedded concatenated vector."""
    vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
    vector_exp = np.zeros((hash_len,), np.float32)
    vector_seq = np.zeros((hash_len,), np.float32)

    # 1. Experimental Spectrum Embedding
    for mz, intensity in exp_peaks:
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector_exp[bin_idx] += intensity

    # 2. Sequence Embedding (Theoretical or Original+Complementary)
    for mz, intensity in seq_peaks:
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector_seq[bin_idx] += intensity

    if norm:
        # L2 Normalization to avoid NaN errors on empty vectors
        norm_exp = np.linalg.norm(vector_exp)
        norm_seq = np.linalg.norm(vector_seq)
        if norm_exp > 0: vector_exp /= norm_exp
        if norm_seq > 0: vector_seq /= norm_seq

    # Concatenate [Sequence Vector | Experimental Vector]
    return np.concatenate((vector_seq, vector_exp))


def spectrum_to_vector_com_lib(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                               bin_size: float, hash_len: int, norm: bool = True, 
                               is_lib: bool = True, vector: np.ndarray = None) -> np.ndarray:
    """Vectorize using complementary spectrum representation."""
    spectrum_temp = spectrum
    exp_peaks = get_complimentary_spectrum_half(spectrum)
    
    if is_lib:
        seq = _parse_sequence_for_vector(spectrum_temp.peptide)
        seq_peaks_mz, seq_peaks_int = generate_theoretical_spectrum(seq)
        seq_peaks = list(zip(seq_peaks_mz, seq_peaks_int))
    else:
        # Query uses original + complementary peaks to estimate sequence embedding
        seq_peaks = get_complimentary_spectrum_half(spectrum_temp)
        seq_peaks.extend(list(zip(spectrum_temp.mz, spectrum_temp.intensity)))

    return _build_concatenated_vector(exp_peaks, seq_peaks, min_mz, max_mz, bin_size, hash_len, norm)


def spectrum_to_vector_com_lib_weight(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                                      bin_size: float, hash_len: int, norm: bool = True, 
                                      is_lib: bool = True, vector: np.ndarray = None) -> np.ndarray:
    """Vectorize using weighted complementary spectrum representation."""
    spectrum_temp = spectrum
    exp_peaks = get_complimentary_spectrum_weight(spectrum)
    
    if is_lib:
        seq = _parse_sequence_for_vector(spectrum_temp.peptide)
        seq_peaks_mz, seq_peaks_int = generate_theoretical_spectrum(seq)
        seq_peaks = list(zip(seq_peaks_mz, seq_peaks_int))
    else:
        seq_peaks = get_complimentary_spectrum_half(spectrum_temp)
        seq_peaks.extend(list(zip(spectrum_temp.mz, spectrum_temp.intensity)))

    return _build_concatenated_vector(exp_peaks, seq_peaks, min_mz, max_mz, bin_size, hash_len, norm)


def spectrum_to_vector_weight(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                              bin_size: float, hash_len: int, norm: bool = True, 
                              is_lib: bool = True, vector: np.ndarray = None) -> np.ndarray:
    """Vectorize using weighted original spectrum representation."""
    spectrum_temp = spectrum
    exp_peaks = get_spectrum_weight(spectrum)
    
    if is_lib:
        seq = _parse_sequence_for_vector(spectrum_temp.peptide)
        seq_peaks_mz, seq_peaks_int = generate_theoretical_spectrum(seq)
        seq_peaks = list(zip(seq_peaks_mz, seq_peaks_int))
    else:
        seq_peaks = get_complimentary_spectrum_half(spectrum_temp)
        seq_peaks.extend(list(zip(spectrum_temp.mz, spectrum_temp.intensity)))

    return _build_concatenated_vector(exp_peaks, seq_peaks, min_mz, max_mz, bin_size, hash_len, norm)


def spectrum_to_vector(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, hash_len: int, norm: bool = True, 
                       is_lib: bool = True, spectrum_seq = None, vector: np.ndarray = None) -> np.ndarray:
    """Vectorize using original spectrum representation."""
    spectrum_temp = spectrum
    exp_peaks = list(zip(spectrum.mz, spectrum.intensity))
    
    if is_lib:
        seq = _parse_sequence_for_vector(spectrum.peptide)
        seq_peaks_mz, seq_peaks_int = generate_theoretical_spectrum(seq)
        seq_peaks = list(zip(seq_peaks_mz, seq_peaks_int))
    else:
        seq_peaks = get_complimentary_spectrum_half(spectrum_temp)
        seq_peaks.extend(list(zip(spectrum_temp.mz, spectrum_temp.intensity)))

    return _build_concatenated_vector(exp_peaks, seq_peaks, min_mz, max_mz, bin_size, hash_len, norm)


def spectrum_to_vector_baseline(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                                bin_size: float, hash_len: int, norm: bool = True, 
                                is_lib: bool = True, spectrum_seq = None, vector: np.ndarray = None) -> np.ndarray:
    """Baseline vectorization (Single Embedding) without complementary or sequence dual-embedding."""
    vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
    if hash_len is not None:
        vec_len = hash_len
    vector = np.zeros((vec_len,), np.float32)
    
    for mz, intensity in zip(spectrum.mz, spectrum.intensity):
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector[bin_idx] += intensity
        
    if norm:
        norm_val = np.linalg.norm(vector)
        if norm_val > 0: vector /= norm_val
    return vector

class SpectrumSpectrumMatch:
    """Class to hold information regarding a matched Query and Library spectrum."""
    def __init__(self, query_spectrum: MsmsSpectrum, library_spectrum: MsmsSpectrum = None,
                 search_engine_score: float = math.nan, q: float = math.nan, num_candidates: int = 0):
        self.query_spectrum = query_spectrum
        self.library_spectrum = library_spectrum
        self.search_engine_score = search_engine_score
        self.q = q
        self.num_candidates = num_candidates


    @property
    def sequence(self):
        return (self.library_spectrum.peptide
                if self.library_spectrum is not None else None)

    @property
    def query_identifier(self):
        return self.query_spectrum.identifier

    @property
    def query_index(self):
        return self.query_spectrum.index
    
    @property
    def query_std(self):
        return self.query_spectrum.std

    @property
    def library_identifier(self):
        return (self.library_spectrum.identifier
                if self.library_spectrum is not None else None)

    @property
    def retention_time(self):
        return self.query_spectrum.retention_time

    @property
    def charge(self):
        return self.query_spectrum.precursor_charge

    @property
    def exp_mass_to_charge(self):
        return self.query_spectrum.precursor_mz

    @property
    def calc_mass_to_charge(self):
        return (self.library_spectrum.precursor_mz
                if self.library_spectrum is not None else None)

    @property
    def is_decoy(self):
        return (self.library_spectrum.is_decoy
                if self.library_spectrum is not None else None)
