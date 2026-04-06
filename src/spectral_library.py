import collections
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import time
from typing import Dict, Iterator, List

import faiss
import numexpr as ne
import numpy as np
import tqdm
from spectrum_utils.spectrum import MsmsSpectrum

from ann_solo import reader, spectrum_match, utils
from ann_solo.config import config
from ann_solo.spectrum import (process_spectrum, spectrum_to_vector, spectrum_to_vector_com_lib, 
                               spectrum_to_vector_com_lib_weight, spectrum_to_vector_weight)

# Global setting for output tracking
processing_file = "12_16_3_query_seq_emb_method_seq_HEK"

class SpectralLibrary:
    """
    Spectral library search engine based on FAISS ANN index.
    Executes the POMS dual-embedding and multi-representation strategy.
    """
    _hyperparameters = ['min_mz', 'max_mz', 'bin_size', 'hash_len', 'num_list']

    # FAISS Index files for 4 parallel vector spaces
    _ann_filenames = {}          # Original
    _ann_filenames_com = {}      # Complementary
    _ann_filenames_com_weight = {} # Weighted Complementary
    _ann_filenames_weight = {}   # Weighted Original

    _ann_index_lock = multiprocessing.Lock()
    
    def __init__(self, filename: str) -> None:
        """Initialize the spectral library and build 4 parallel FAISS indexes if they are missing."""
        try:
            self._library_reader = reader.SpectralLibraryReader(filename, self._get_hyperparameter_hash())
            self._library_reader.open()
        except FileNotFoundError as e:
            logging.error(e)
            raise

        self._num_probe = config.num_probe
        self._num_candidates = config.num_candidates
        self._use_gpu = not config.no_gpu and faiss.get_num_gpus()
        
        if self._use_gpu:
            self._res = faiss.StandardGpuResources()
            if self._num_probe > 1024: self._num_probe = 1024
            if self._num_candidates > 1024: self._num_candidates = 1024

        # Pointers for currently loaded indices
        self._current_index_ori = None, None
        self._current_index_com = None, None
        self._current_index_com_weight = None, None
        self._current_index_weight = None, None
            
        if config.mode == 'ann':
            self._initialize_ann_indices(filename)

    def _initialize_ann_indices(self, filename: str):
        """Helper method to check and build missing FAISS indexes."""
        verify_file_existence = not self._library_reader.is_recreated
        base_filename = f'{os.path.splitext(filename)[0]}_{self._get_hyperparameter_hash()[:7]}'
        
        ann_charges = [charge for charge, info in self._library_reader.spec_info['charge'].items()
                       if len(info['id']) >= config.num_list]

        # Tracking missing indexes
        missing_ori, missing_com, missing_com_wt, missing_wt = [], [], [], []

        for charge in sorted(ann_charges):
            self._ann_filenames[charge] = f'{base_filename}_{charge}_baseline_6_12_seq.idxann'
            self._ann_filenames_com[charge] = f'{base_filename}_{charge}_com_method_6_12_seq.idxann'
            self._ann_filenames_com_weight[charge] = f'{base_filename}_{charge}_com_weight_method_6_12_seq.idxann'
            self._ann_filenames_weight[charge] = f'{base_filename}_{charge}_weight_method_6_12_seq.idxann'

            if not verify_file_existence or not os.path.isfile(self._ann_filenames[charge]): missing_ori.append(charge)
            if not verify_file_existence or not os.path.isfile(self._ann_filenames_com[charge]): missing_com.append(charge)
            if not verify_file_existence or not os.path.isfile(self._ann_filenames_com_weight[charge]): missing_com_wt.append(charge)
            if not verify_file_existence or not os.path.isfile(self._ann_filenames_weight[charge]): missing_wt.append(charge)

        if "method" in processing_file:
            if missing_ori: self._build_single_ann_index(missing_ori, spectrum_to_vector, self._ann_filenames)
            if missing_wt: self._build_single_ann_index(missing_wt, spectrum_to_vector_weight, self._ann_filenames_weight)
            if missing_com: self._build_single_ann_index(missing_com, spectrum_to_vector_com_lib, self._ann_filenames_com)
            if missing_com_wt: self._build_single_ann_index(missing_com_wt, spectrum_to_vector_com_lib_weight, self._ann_filenames_com_weight)

    def _build_single_ann_index(self, charges: List[int], vector_func, file_dict) -> None:
        """Generic method to build a FAISS Index for a specific vector representation to avoid code duplication."""
        logging.info(f'Building FAISS indexes for {vector_func.__name__}')
        
        # Dual-embedding size requires hash_len * 2
        vector_dim = int(config.hash_len * 2)
        charge_vectors = {charge: np.zeros((len(self._library_reader.spec_info['charge'][charge]['id']), vector_dim), np.float32) for charge in charges}
        counters = {charge: 0 for charge in charges}

        for lib_spectrum, _ in tqdm.tqdm(self._library_reader.get_all_spectra(), desc='Vectorizing library', leave=False, smoothing=0.1):
            charge = lib_spectrum.precursor_charge
            if charge in charge_vectors:
                charge_vectors[charge][counters[charge]] = vector_func(
                    process_spectrum(lib_spectrum, True), config.min_mz, config.max_mz, config.bin_size, config.hash_len, True, True)
                counters[charge] += 1

        for charge, vectors in charge_vectors.items():
            quantizer = faiss.IndexFlatIP(vector_dim)
            ann_index = faiss.IndexIVFFlat(quantizer, vector_dim, config.num_list, faiss.METRIC_INNER_PRODUCT)
            ann_index.train(vectors)
            ann_index.add(vectors)
            faiss.write_index(ann_index, file_dict[charge])

    def _get_hyperparameter_hash(self) -> str:
        hyperparameters_bytes = json.dumps({hp: config[hp] for hp in self._hyperparameters}).encode('utf-8')
        return hashlib.sha1(hyperparameters_bytes).hexdigest()

    def shutdown(self) -> None:
        """Gracefully release memory resources."""
        self._library_reader.close()
        for idx in [self._current_index_ori, self._current_index_com, self._current_index_com_weight, self._current_index_weight]:
            if idx[1] is not None: idx[1].reset()

    def search(self, query_filename: str, candidate_list: list) -> List[SpectrumSpectrumMatch]:
        """Perform Standard Search followed by Open Search cascade."""
        logging.info('Process file %s', query_filename)

        query_spectra = collections.defaultdict(list)
        for query_spectrum in tqdm.tqdm(reader.read_mgf(query_filename), desc='Query read', leave=False, smoothing=0.7):
            charges = [query_spectrum.precursor_charge] if query_spectrum.precursor_charge else [2, 3]
            for charge in charges:
                spec_copy = copy.copy(query_spectrum)
                spec_copy.precursor_charge = charge
                if process_spectrum(spec_copy, False).is_valid:
                    query_spectra[charge].append(spec_copy)

        identifications = {}
        
        # 1. Cascade Level 1: Standard Search
        start_time = time.time()
        for ssm in self._search_cascade(query_spectra, 'std', candidate_list):
            identifications[ssm.query_identifier] = ssm
            
        logging.info('%d spectra identified after standard search', len(identifications))

        # 2. Cascade Level 2: Open Search
        if config.precursor_tolerance_mass_open and config.precursor_tolerance_mode_open:
            for charge, spectra_list in query_spectra.items():
                query_spectra[charge] = [s for s in spectra_list if s.identifier not in identifications]
                
            for ssm in self._search_cascade(query_spectra, 'open', candidate_list):
                identifications[ssm.query_identifier] = ssm
                
            logging.info('%d spectra identified after open search', len(identifications))

        return list(identifications.values()), candidate_list

    def _search_cascade(self, query_spectra, mode: str, candidate_list: list) -> Iterator[SpectrumSpectrumMatch]:
        ssms = {}
        batch_size = config.batch_size
        
        with tqdm.tqdm(desc=f'Searching ({mode})', total=sum(len(q) for q in query_spectra.values()), leave=False, smoothing=0.1) as pbar:
            for charge, spectra_list in query_spectra.items():
                for batch_i in range(0, len(spectra_list), batch_size):
                    batch = spectra_list[batch_i:batch_i + batch_size]
                    for ssm in self._search_batch(batch, charge, mode, candidate_list, batch_i):
                        if ssm and (ssm.query_identifier not in ssms or ssm.search_engine_score > ssms[ssm.query_identifier].search_engine_score):
                            ssms[ssm.query_identifier] = ssm
                    pbar.update(len(batch))

        if mode == 'std':
            return utils.filter_fdr(ssms.values(), config.fdr)
        return utils.filter_group_fdr(ssms.values(), config.fdr, config.fdr_tolerance_mass, config.fdr_tolerance_mode, config.fdr_min_group_size)

    def _search_batch(self, query_spectra, charge: int, mode: str, candidate_list: list, batch_i: int) -> Iterator[SpectrumSpectrumMatch]:
        for query_spectrum, (library_candidates, query_id) in zip(query_spectra, self._get_library_candidates(query_spectra, charge, mode, batch_i)):
            candidate_list.append([query_spectrum.identifier, query_id, 1 if mode == "open" else 0])
            
            if library_candidates:
                library_match, score, _ = spectrum_match.get_best_match(query_spectrum, library_candidates, config.fragment_mz_tolerance, config.allow_peak_shifts)
                yield SpectrumSpectrumMatch(query_spectrum, library_match, score, num_candidates=len(library_candidates))

    def _get_library_candidates(self, query_spectra: List[MsmsSpectrum], charge: int, mode: str, batch_i: int) -> Iterator[List[MsmsSpectrum]]:
        """Retrieve top candidates using the 4-Parallel Vector Search strategy."""
        if charge not in self._library_reader.spec_info['charge']: return

        tol_val = config.precursor_tolerance_mass if mode == 'std' else config.precursor_tolerance_mass_open
        tol_mode = config.precursor_tolerance_mode if mode == 'std' else config.precursor_tolerance_mode_open
        library_candidates = self._library_reader.spec_info['charge'][charge]
        
        # 1. Precursor Mass Filter
        query_mzs = np.array([q.precursor_mz for q in query_spectra]).reshape(-1, 1)
        library_mzs = library_candidates['precursor_mz'].reshape((1, -1))
        
        if tol_mode == 'Da':
            candidate_filters = ne.evaluate('abs(query_mzs - library_mzs) * charge <= tol_val')
        elif tol_mode == 'ppm':
            candidate_filters = ne.evaluate('abs(query_mzs - library_mzs) / library_mzs * 10**6 <= tol_val')

        # 2. ANN Dual-Embedding Filter
        if config.mode == 'ann' and mode == 'open' and charge in self._ann_filenames:
            vector_dim = int(config.hash_len * 2)
            
            q_vec_ori = np.zeros((len(query_spectra), vector_dim), np.float32)
            q_vec_wt = np.zeros((len(query_spectra), vector_dim), np.float32)
            q_vec_com = np.zeros((len(query_spectra), vector_dim), np.float32)
            q_vec_com_wt = np.zeros((len(query_spectra), vector_dim), np.float32)
            
            for i, q_spec in enumerate(query_spectra):
                q_spec.std = 1
                q_vec_ori[i] = spectrum_to_vector(q_spec, config.min_mz, config.max_mz, config.bin_size, config.hash_len, True, is_lib=False)
                q_vec_wt[i] = spectrum_to_vector_weight(q_spec, config.min_mz, config.max_mz, config.bin_size, config.hash_len, True, is_lib=False)
                q_vec_com[i] = spectrum_to_vector_com(q_spec, config.min_mz, config.max_mz, config.bin_size, config.hash_len, True, is_lib=False)
                q_vec_com_wt[i] = spectrum_to_vector_com_lib_weight(q_spec, config.min_mz, config.max_mz, config.bin_size, config.hash_len, True, is_lib=False)

            mask = np.zeros_like(candidate_filters)

            # Parallel Search: Jointly extract 512 from Top and 512 from Bottom representations
            k = self._num_candidates // 2  # 512
            
            res_ori = self._get_cached_index(charge, 'ori').search(q_vec_ori, k)
            res_wt = self._get_cached_index(charge, 'wt').search(q_vec_wt, k)
            res_com = self._get_cached_index(charge, 'com').search(q_vec_com, k)
            res_com_wt = self._get_cached_index(charge, 'com_wt').search(q_vec_com_wt, k)

            stacked_ori = np.block([res_ori[0], res_wt[0]])
            stacked_idx_ori = np.block([res_ori[1], res_wt[1]])
            stacked_com = np.block([res_com[0], res_com_wt[0]])
            stacked_idx_com = np.block([res_com[1], res_com_wt[1]])
            
            candidate_idx = []
            for d_ori, i_ori, d_com, i_com in zip(stacked_ori, stacked_idx_ori, stacked_com, stacked_idx_com):
                # Sort and select top 512 from (Original + Weighted Original)
                top_ori = sorted(zip(d_ori, i_ori), reverse=True)[:k]
                # Sort and select top 512 from (Complementary + Weighted Complementary)
                top_com = sorted(zip(d_com, i_com), reverse=True)[:k]
                
                combined_idx = np.array([idx for _, idx in top_ori + top_com])
                candidate_idx.append(combined_idx)
                                        
            for mask_i, ann_filter in zip(mask, candidate_idx):
                mask_i[ann_filter[ann_filter != -1]] = True
                
            candidate_filters = np.logical_and(candidate_filters, mask)

        # 3. Retrieve Library Candidates
        for filter_array in candidate_filters:
            query_candidates, query_ids = [], []
            for idx in library_candidates['id'][filter_array]:
                candidate = self._library_reader.get_spectrum(idx, True, True)
                if candidate.is_valid:
                    query_candidates.append(candidate)
                    query_ids.append(candidate.identifier)
            yield query_candidates, query_ids

    def _get_cached_index(self, charge: int, index_type: str) -> faiss.IndexIVF:
        """Helper to manage dynamic loading of FAISS GPU/CPU indexes to avoid memory overflow."""
        with self._ann_index_lock:
            # Map index type to internal attributes
            idx_map = {
                'ori': (self._current_index_ori, self._ann_filenames),
                'com': (self._current_index_com, self._ann_filenames_com),
                'com_wt': (self._current_index_com_weight, self._ann_filenames_com_weight),
                'wt': (self._current_index_weight, self._ann_filenames_weight)
            }
            
            current_cache, file_dict = idx_map[index_type]
            
            if current_cache[0] != charge:
                if current_cache[1] is not None: current_cache[1].reset()
                index = faiss.read_index(file_dict[charge])
                if self._use_gpu:
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = True
                    index = faiss.index_cpu_to_gpu(self._res, 0, index, co)
                    index.setNumProbes(self._num_probe)
                else:
                    index.nprobe = self._num_probe
                
                # Update cache
                if index_type == 'ori': self._current_index_ori = charge, index
                elif index_type == 'com': self._current_index_com = charge, index
                elif index_type == 'com_wt': self._current_index_com_weight = charge, index
                elif index_type == 'wt': self._current_index_weight = charge, index
                
                return index
            return current_cache[1]