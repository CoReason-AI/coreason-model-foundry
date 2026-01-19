# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from typing import List, Set

from utils.logger import logger


class DecontaminationChecker:
    """
    Checks for data contamination by calculating N-gram overlap between train and test sets.
    """

    def __init__(self, n_gram_size: int = 13):
        """
        Args:
            n_gram_size: The size of N-grams to use for overlap checking (default 13).
        """
        self.n_gram_size = n_gram_size

    def _generate_ngrams(self, text: str) -> Set[str]:
        """
        Generates a set of N-grams from the given text.
        """
        words = text.split()
        if len(words) < self.n_gram_size:
            return set()

        ngrams = set()
        for i in range(len(words) - self.n_gram_size + 1):
            ngram = " ".join(words[i : i + self.n_gram_size])
            ngrams.add(ngram)
        return ngrams

    def check_overlap(self, train_texts: List[str], test_texts: List[str]) -> float:
        """
        Calculates the percentage of test N-grams that appear in the training set.

        Args:
            train_texts: List of strings from the training set.
            test_texts: List of strings from the test set.

        Returns:
            The contamination percentage (0.0 to 100.0).
        """
        if not train_texts or not test_texts:
            logger.warning("Empty train or test set provided for decontamination check.")
            return 0.0

        logger.info(f"Building N-gram index for {len(train_texts)} training examples...")
        train_ngrams = set()
        for text in train_texts:
            train_ngrams.update(self._generate_ngrams(text))

        if not train_ngrams:
            logger.info("No N-grams generated from training data (texts too short?).")
            return 0.0

        logger.info(f"Checking overlap for {len(test_texts)} test examples...")
        total_test_ngrams = 0
        overlapping_ngrams = 0

        for text in test_texts:
            test_ngram_set = self._generate_ngrams(text)
            total_test_ngrams += len(test_ngram_set)
            overlapping_ngrams += len(test_ngram_set.intersection(train_ngrams))

        if total_test_ngrams == 0:
            logger.info("No N-grams generated from test data.")
            return 0.0

        overlap_percentage = (overlapping_ngrams / total_test_ngrams) * 100.0
        logger.info(f"Contamination Check: {overlap_percentage:.2f}% overlap found.")

        return overlap_percentage
