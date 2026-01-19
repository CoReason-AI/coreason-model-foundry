# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import pytest

from coreason_model_foundry.examiner.contamination import DecontaminationChecker


class TestDecontaminationChecker:
    @pytest.fixture
    def checker(self) -> DecontaminationChecker:
        # Use smaller N for easier testing, though default is 13
        return DecontaminationChecker(n_gram_size=3)

    def test_generate_ngrams(self, checker: DecontaminationChecker) -> None:
        text = "one two three four five"
        ngrams = checker._generate_ngrams(text)
        assert ngrams == {"one two three", "two three four", "three four five"}

    def test_generate_ngrams_short_text(self, checker: DecontaminationChecker) -> None:
        text = "one two"
        ngrams = checker._generate_ngrams(text)
        assert ngrams == set()

    def test_check_overlap_exact_match(self, checker: DecontaminationChecker) -> None:
        train = ["one two three four"]
        test = ["one two three four"]
        # n=3.
        # "one two three", "two three four"
        # Overlap should be 100%
        overlap = checker.check_overlap(train, test)
        assert overlap == 100.0

    def test_check_overlap_no_match(self, checker: DecontaminationChecker) -> None:
        train = ["one two three four"]
        test = ["five six seven eight"]
        overlap = checker.check_overlap(train, test)
        assert overlap == 0.0

    def test_check_overlap_partial(self, checker: DecontaminationChecker) -> None:
        train = ["one two three four"]
        test = ["one two three five"]
        # Train ngrams: {one two three, two three four}
        # Test ngrams: {one two three, two three five}
        # Overlap: "one two three" (1 count)
        # Total test ngrams: 2
        # Percentage: 50%
        overlap = checker.check_overlap(train, test)
        assert overlap == 50.0

    def test_empty_input(self, checker: DecontaminationChecker) -> None:
        assert checker.check_overlap([], ["some data"]) == 0.0
        assert checker.check_overlap(["some data"], []) == 0.0
        assert checker.check_overlap([], []) == 0.0

    def test_default_ngram_size(self) -> None:
        # Default is 13
        checker = DecontaminationChecker()
        # Create a string with 13 words
        text = "word " * 13
        text = text.strip()
        ngrams = checker._generate_ngrams(text)
        assert len(ngrams) == 1

    def test_multiple_sentences(self, checker: DecontaminationChecker) -> None:
        train = ["a b c d", "e f g h"]
        test = ["a b c d"]
        # Train: {a b c, b c d, e f g, f g h}
        # Test: {a b c, b c d}
        # Overlap: 2 / 2 = 100%
        assert checker.check_overlap(train, test) == 100.0

    def test_test_ngrams_zero(self, checker: DecontaminationChecker) -> None:
        # Test set has text, but not enough for n-grams
        train = ["a b c d"]
        test = ["a b"]
        assert checker.check_overlap(train, test) == 0.0

    def test_train_ngrams_zero(self, checker: DecontaminationChecker) -> None:
        # Train set has text, but not enough for n-grams
        train = ["a b"]  # shorter than N=3
        test = ["a b c d"]
        assert checker.check_overlap(train, test) == 0.0

    def test_case_sensitivity(self, checker: DecontaminationChecker) -> None:
        # Default behavior: Case sensitive
        train = ["One Two Three"]
        test = ["one two three"]
        # n=3. "One Two Three" != "one two three"
        assert checker.check_overlap(train, test) == 0.0

    def test_punctuation_sensitivity(self, checker: DecontaminationChecker) -> None:
        # Default behavior: Punctuation is part of the token if split by space
        train = ["one two three."]
        test = ["one two three"]
        # "one two three." (last token "three.") != "one two three" (last token "three")
        assert checker.check_overlap(train, test) == 0.0

    def test_ngram_size_one(self) -> None:
        checker = DecontaminationChecker(n_gram_size=1)
        train = ["a b"]
        test = ["a c"]
        # Train: {a, b}
        # Test: {a, c}
        # Overlap: {a} (1) / 2 = 50%
        assert checker.check_overlap(train, test) == 50.0

    def test_unicode_handling(self, checker: DecontaminationChecker) -> None:
        train = ["café au lait"]
        test = ["café au lait"]
        assert checker.check_overlap(train, test) == 100.0

        test_diff = ["cafe au lait"]
        assert checker.check_overlap(train, test_diff) == 0.0

    def test_complex_scenario(self, checker: DecontaminationChecker) -> None:
        # Complex Scenario: Real-ish paragraphs
        train = [
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        ]
        test = [
            "The quick brown fox jumps over the lazy dog.",  # Exact match (overlap)
            "The quick brown fox sits on the lazy cat.",  # Partial overlap
            "Unique sentence with no overlap at all here.",  # No overlap
        ]

        # Checker N=3
        # Train Ngrams:
        # Sent 1: {The quick brown, quick brown fox, ... lazy dog.}
        # Sent 2: {Lorem ipsum dolor, ...}

        # Test 1: 100% overlap with Train Sent 1
        # Test 2: "The quick brown fox" -> {The quick brown, quick brown fox} match. "sits on the" ... no match.
        # Test 3: 0 matches.

        # Total Test Ngrams:
        # T1: 9 words -> 7 ngrams (all match)
        # T2: 9 words -> 7 ngrams. Matches: "The quick brown", "quick brown fox". (2 matches)
        # T3: 8 words -> 6 ngrams. (0 matches)

        # Total Test Ngrams = 7 + 7 + 6 = 20
        # Total Overlap = 7 + 2 + 0 = 9
        # Expected % = 9 / 20 * 100 = 45.0

        expected_overlap = (9 / 20) * 100.0
        assert checker.check_overlap(train, test) == pytest.approx(expected_overlap)
