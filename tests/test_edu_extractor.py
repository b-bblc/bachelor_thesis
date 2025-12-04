"""
Unit tests for the edu_extractor module.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path to import src package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edu_extractor import EDUExtractor


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_rs3_content():
    """Sample RST (.rs3) XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<rst>
  <header>
    <relations>
      <rel name="elaboration" type="rst"/>
      <rel name="contrast" type="multinuc"/>
    </relations>
  </header>
  <body>
    <segment id="1" parent="5" relname="span">This is the first EDU.</segment>
    <segment id="2" parent="5" relname="elaboration">It provides important context.</segment>
    <segment id="3" parent="4" relname="contrast">Some argue for this approach,</segment>
    <segment id="4" parent="5" relname="span">while others prefer alternatives.</segment>
    <segment id="5" parent="0" relname="span">This is the conclusion.</segment>
  </body>
</rst>
"""


@pytest.fixture
def sample_rs3_file(tmp_path, sample_rs3_content):
    """Create a temporary .rs3 file for testing."""
    file_path = tmp_path / "test.rs3"
    file_path.write_text(sample_rs3_content, encoding='utf-8')
    return file_path


@pytest.fixture
def sample_rs3_with_placeholders():
    """RST content with placeholder text to be filtered."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<rst>
  <body>
    <segment id="1">##### This contains placeholder.</segment>
    <segment id="2">Clean EDU here.</segment>
    <segment id="3">http://example.com should be filtered</segment>
    <segment id="4">X</segment>
    <segment id="5">Valid EDU text.</segment>
  </body>
</rst>
"""


@pytest.fixture
def sample_rs3_file_with_placeholders(tmp_path, sample_rs3_with_placeholders):
    """Create a temporary .rs3 file with placeholder text."""
    file_path = tmp_path / "placeholders.rs3"
    file_path.write_text(sample_rs3_with_placeholders, encoding='utf-8')
    return file_path


@pytest.fixture
def invalid_xml_file(tmp_path):
    """Create a file with invalid XML."""
    file_path = tmp_path / "invalid.rs3"
    file_path.write_text("<rst><body><segment>unclosed", encoding='utf-8')
    return file_path


@pytest.fixture
def empty_rs3_file(tmp_path):
    """Create an empty RST file with no segments."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<rst>
  <body>
  </body>
</rst>
"""
    file_path = tmp_path / "empty.rs3"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def extractor():
    """Create EDUExtractor instance."""
    return EDUExtractor()


# =============================================================================
# Tests for extract_edus_from_rs3
# =============================================================================

class TestExtractEdusFromRs3:
    """Tests for the extract_edus_from_rs3 method."""
    
    def test_extracts_correct_number_of_edus(self, extractor, sample_rs3_file):
        """Should extract all EDUs from the RST file."""
        edus = extractor.extract_edus_from_rs3(str(sample_rs3_file))
        
        assert len(edus) == 5
    
    def test_extracts_correct_edu_content(self, extractor, sample_rs3_file):
        """Should extract EDU text correctly."""
        edus = extractor.extract_edus_from_rs3(str(sample_rs3_file))
        
        assert "This is the first EDU." in edus
        assert "It provides important context." in edus
        assert "This is the conclusion." in edus
    
    def test_returns_list_of_strings(self, extractor, sample_rs3_file):
        """Should return a list of strings."""
        edus = extractor.extract_edus_from_rs3(str(sample_rs3_file))
        
        assert isinstance(edus, list)
        assert all(isinstance(edu, str) for edu in edus)
    
    def test_filters_placeholder_text(self, extractor, sample_rs3_file_with_placeholders):
        """Should remove ##### placeholder markers."""
        edus = extractor.extract_edus_from_rs3(str(sample_rs3_file_with_placeholders))
        
        # Check placeholder is removed
        for edu in edus:
            assert "#####" not in edu
        
        # "This contains placeholder." should still be present (after removing #####)
        assert any("This contains placeholder" in edu for edu in edus)
    
    def test_filters_url_containing_edus(self, extractor, sample_rs3_file_with_placeholders):
        """Should filter out EDUs containing URLs."""
        edus = extractor.extract_edus_from_rs3(str(sample_rs3_file_with_placeholders))
        
        for edu in edus:
            assert "http" not in edu.lower()
    
    def test_filters_very_short_edus(self, extractor, sample_rs3_file_with_placeholders):
        """Should filter out EDUs with 1 or fewer characters."""
        edus = extractor.extract_edus_from_rs3(str(sample_rs3_file_with_placeholders))
        
        # "X" should be filtered out
        for edu in edus:
            assert len(edu) > 1
    
    def test_handles_empty_file(self, extractor, empty_rs3_file):
        """Should return empty list for file with no segments."""
        edus = extractor.extract_edus_from_rs3(str(empty_rs3_file))
        
        assert edus == []
    
    def test_handles_invalid_xml(self, extractor, invalid_xml_file):
        """Should return empty list for invalid XML files."""
        edus = extractor.extract_edus_from_rs3(str(invalid_xml_file))
        
        assert edus == []
    
    def test_handles_nonexistent_file(self, extractor):
        """Should return empty list for non-existent file."""
        edus = extractor.extract_edus_from_rs3("/nonexistent/path.rs3")
        
        assert edus == []
    
    def test_strips_whitespace(self, extractor, tmp_path):
        """Should strip leading/trailing whitespace from EDUs."""
        content = """<?xml version="1.0" encoding="UTF-8"?>
<rst>
  <body>
    <segment id="1">   Whitespace around text.   </segment>
  </body>
</rst>
"""
        file_path = tmp_path / "whitespace.rs3"
        file_path.write_text(content, encoding='utf-8')
        
        edus = extractor.extract_edus_from_rs3(str(file_path))
        
        assert edus[0] == "Whitespace around text."


# =============================================================================
# Tests for get_sentence_boundaries
# =============================================================================

class TestGetSentenceBoundaries:
    """Tests for the get_sentence_boundaries method."""
    
    def test_single_sentence(self, extractor):
        """Single sentence ending with period."""
        edus = ["This is part one,", "and this is part two."]
        
        boundaries = extractor.get_sentence_boundaries(edus)
        
        assert boundaries == [(0, 2)]
    
    def test_multiple_sentences(self, extractor):
        """Multiple complete sentences."""
        edus = [
            "First sentence.",
            "Second sentence?",
            "Third sentence!"
        ]
        
        boundaries = extractor.get_sentence_boundaries(edus)
        
        assert len(boundaries) == 3
        assert boundaries[0] == (0, 1)
        assert boundaries[1] == (1, 2)
        assert boundaries[2] == (2, 3)
    
    def test_mixed_sentence_boundaries(self, extractor):
        """Mix of single and multi-EDU sentences."""
        edus = [
            "Introduction,",
            "followed by explanation.",
            "Single EDU sentence.",
            "Multi-part",
            "conclusion here."
        ]
        
        boundaries = extractor.get_sentence_boundaries(edus)
        
        assert len(boundaries) == 3
        assert boundaries[0] == (0, 2)  # "Introduction," + "followed by explanation."
        assert boundaries[1] == (2, 3)  # "Single EDU sentence."
        assert boundaries[2] == (3, 5)  # "Multi-part" + "conclusion here."
    
    def test_handles_empty_list(self, extractor):
        """Should handle empty EDU list."""
        boundaries = extractor.get_sentence_boundaries([])
        
        assert boundaries == []
    
    def test_handles_no_sentence_ending(self, extractor):
        """Should handle EDUs without sentence-ending punctuation."""
        edus = ["No ending", "punctuation here"]
        
        boundaries = extractor.get_sentence_boundaries(edus)
        
        # Should still group all as one sentence
        assert boundaries == [(0, 2)]
    
    def test_recognizes_colon_as_boundary(self, extractor):
        """Colon should be recognized as sentence boundary."""
        edus = ["Here is a list:", "First item.", "Second item."]
        
        boundaries = extractor.get_sentence_boundaries(edus)
        
        assert boundaries[0] == (0, 1)  # "Here is a list:"
    
    def test_recognizes_semicolon_as_boundary(self, extractor):
        """Semicolon should be recognized as sentence boundary."""
        edus = ["First clause;", "second clause."]
        
        boundaries = extractor.get_sentence_boundaries(edus)
        
        assert len(boundaries) == 2


# =============================================================================
# Tests for group_edus_to_sentences
# =============================================================================

class TestGroupEdusToSentences:
    """Tests for the group_edus_to_sentences method."""
    
    def test_groups_single_sentence(self, extractor):
        """Should join multiple EDUs into one sentence."""
        edus = ["This is part one,", "and this is part two."]
        
        sentences = extractor.group_edus_to_sentences(edus)
        
        assert len(sentences) == 1
        assert sentences[0] == "This is part one, and this is part two."
    
    def test_groups_multiple_sentences(self, extractor):
        """Should group EDUs into separate sentences."""
        edus = [
            "First sentence.",
            "Start of second,",
            "end of second."
        ]
        
        sentences = extractor.group_edus_to_sentences(edus)
        
        assert len(sentences) == 2
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Start of second, end of second."
    
    def test_handles_empty_list(self, extractor):
        """Should handle empty EDU list."""
        sentences = extractor.group_edus_to_sentences([])
        
        assert sentences == []
    
    def test_preserves_single_edu_sentences(self, extractor):
        """Single-EDU sentences should remain intact."""
        edus = ["Complete sentence one.", "Complete sentence two."]
        
        sentences = extractor.group_edus_to_sentences(edus)
        
        assert len(sentences) == 2
        assert sentences[0] == "Complete sentence one."
        assert sentences[1] == "Complete sentence two."


# =============================================================================
# Tests for extract_edus_from_directory
# =============================================================================

class TestExtractEdusFromDirectory:
    """Tests for the extract_edus_from_directory method."""
    
    def test_processes_multiple_files(self, extractor, tmp_path):
        """Should process all .rs3 files in directory."""
        # Create input and output directories
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        # Create two .rs3 files
        content = """<?xml version="1.0" encoding="UTF-8"?>
<rst>
  <body>
    <segment id="1">Test EDU.</segment>
  </body>
</rst>
"""
        (input_dir / "file1.rs3").write_text(content, encoding='utf-8')
        (input_dir / "file2.rs3").write_text(content, encoding='utf-8')
        
        stats = extractor.extract_edus_from_directory(str(input_dir), str(output_dir))
        
        assert len(stats) == 2
        assert "file1.txt" in stats
        assert "file2.txt" in stats
    
    def test_creates_output_directory(self, extractor, tmp_path):
        """Should create output directory if it doesn't exist."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "new_output_dir"
        input_dir.mkdir()
        
        content = """<?xml version="1.0" encoding="UTF-8"?>
<rst>
  <body>
    <segment id="1">Test.</segment>
  </body>
</rst>
"""
        (input_dir / "test.rs3").write_text(content, encoding='utf-8')
        
        extractor.extract_edus_from_directory(str(input_dir), str(output_dir))
        
        assert output_dir.exists()
    
    def test_saves_edus_to_text_files(self, extractor, tmp_path):
        """Should save extracted EDUs to .txt files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        content = """<?xml version="1.0" encoding="UTF-8"?>
<rst>
  <body>
    <segment id="1">First EDU.</segment>
    <segment id="2">Second EDU.</segment>
  </body>
</rst>
"""
        (input_dir / "test.rs3").write_text(content, encoding='utf-8')
        
        extractor.extract_edus_from_directory(str(input_dir), str(output_dir))
        
        output_file = output_dir / "test.txt"
        assert output_file.exists()
        
        content = output_file.read_text(encoding='utf-8')
        assert "First EDU." in content
        assert "Second EDU." in content
    
    def test_returns_extraction_stats(self, extractor, tmp_path):
        """Should return dictionary with EDU counts."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        content = """<?xml version="1.0" encoding="UTF-8"?>
<rst>
  <body>
    <segment id="1">EDU one.</segment>
    <segment id="2">EDU two.</segment>
    <segment id="3">EDU three.</segment>
  </body>
</rst>
"""
        (input_dir / "test.rs3").write_text(content, encoding='utf-8')
        
        stats = extractor.extract_edus_from_directory(str(input_dir), str(output_dir))
        
        assert stats["test.txt"] == 3
    
    def test_handles_empty_directory(self, extractor, tmp_path):
        """Should handle directory with no .rs3 files."""
        input_dir = tmp_path / "empty"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        stats = extractor.extract_edus_from_directory(str(input_dir), str(output_dir))
        
        assert stats == {}


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for EDUExtractor."""
    
    def test_full_extraction_pipeline(self, extractor, tmp_path):
        """Test complete pipeline: extract -> boundary detection -> grouping."""
        content = """<?xml version="1.0" encoding="UTF-8"?>
<rst>
  <body>
    <segment id="1">The project started in 2020,</segment>
    <segment id="2">and it was funded by a grant.</segment>
    <segment id="3">Results were promising.</segment>
  </body>
</rst>
"""
        file_path = tmp_path / "pipeline_test.rs3"
        file_path.write_text(content, encoding='utf-8')
        
        # Extract EDUs
        edus = extractor.extract_edus_from_rs3(str(file_path))
        assert len(edus) == 3
        
        # Get sentence boundaries
        boundaries = extractor.get_sentence_boundaries(edus)
        assert len(boundaries) == 2  # Two sentences: "The project... grant." and "Results..."
        
        # Group into sentences
        sentences = extractor.group_edus_to_sentences(edus)
        assert len(sentences) == 2
        assert "The project started in 2020" in sentences[0]
        assert "and it was funded by a grant." in sentences[0]
        assert "Results were promising." in sentences[1]

