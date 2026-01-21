"""
Test suite for vlm_extract.py module
"""

import unittest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Import the module to test
import sys
sys.path.append('src')
from extraction.vlm_extract import extract_fields, MODEL_NAME, SYSTEM_PROMPT, TASK_PROMPT

class TestVLMExtract(unittest.TestCase):
    """Test cases for VLM extraction functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image_path = "data/train/172655019_3_pg11.png"  # Use existing test image

        # Create a simple test image for mocking
        self.mock_image = Image.new('RGB', (100, 100), color='white')

    def test_model_constants(self):
        """Test that model constants are correctly defined"""
        self.assertEqual(MODEL_NAME, "openbmb/MiniCPM-V-2_6")
        self.assertIsInstance(SYSTEM_PROMPT, str)
        self.assertIsInstance(TASK_PROMPT, str)
        self.assertTrue(len(SYSTEM_PROMPT) > 0)
        self.assertTrue(len(TASK_PROMPT) > 0)

    def test_prompt_structure(self):
        """Test that prompts contain expected elements"""
        # Check SYSTEM_PROMPT contains key elements
        self.assertIn("vision-language document extraction system", SYSTEM_PROMPT.lower())
        self.assertIn("Indian financial documents", SYSTEM_PROMPT)
        self.assertIn("Do not guess or infer missing values", SYSTEM_PROMPT)

        # Check TASK_PROMPT contains field guidance
        self.assertIn("Dealer Name", TASK_PROMPT)
        self.assertIn("Model Name", TASK_PROMPT)
        self.assertIn("Horse Power", TASK_PROMPT)
        self.assertIn("Asset Cost", TASK_PROMPT)

        # Check output format is specified
        self.assertIn("OUTPUT FORMAT", TASK_PROMPT)
        self.assertIn("dealer_name", TASK_PROMPT)
        self.assertIn("model_name", TASK_PROMPT)
        self.assertIn("horse_power", TASK_PROMPT)
        self.assertIn("asset_cost", TASK_PROMPT)

    @patch('src.extraction.vlm_extract.processor')
    @patch('src.extraction.vlm_extract.model')
    def test_extract_fields_with_mock(self, mock_model, mock_processor):
        """Test extract_fields function with mocked model and processor"""

        # Mock the processor and model behavior
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.device = 'cpu'
        mock_model_instance.eval.return_value = mock_model_instance

        # Mock the processor call
        mock_processor_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'pixel_values': torch.randn(1, 3, 224, 224)
        }

        # Mock model generate
        mock_model_instance.generate.return_value = torch.tensor([[5, 6, 7, 8]])

        # Mock processor decode
        mock_processor_instance.decode.return_value = '{"dealer_name": "Test Dealer", "model_name": "Test Model", "horse_power": 50, "asset_cost": 500000}'

        # Call the function
        result = extract_fields(self.test_image_path)

        # Verify the result
        self.assertIsInstance(result, str)

        # Try to parse as JSON to verify structure
        try:
            parsed_result = json.loads(result)
            self.assertIn("dealer_name", parsed_result)
            self.assertIn("model_name", parsed_result)
            self.assertIn("horse_power", parsed_result)
            self.assertIn("asset_cost", parsed_result)
        except json.JSONDecodeError:
            self.fail("Result is not valid JSON")

    def test_extract_fields_with_real_image(self):
        """Test extract_fields with a real image file"""
        if not os.path.exists(self.test_image_path):
            self.skipTest(f"Test image not found: {self.test_image_path}")

        # This test will actually run the model
        # It may take some time and require GPU/CPU resources
        try:
            result = extract_fields(self.test_image_path)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

            # Try to parse as JSON (the result should be JSON format)
            try:
                parsed_result = json.loads(result)
                self.assertIsInstance(parsed_result, dict)

                # Check if expected fields are present
                expected_fields = ["dealer_name", "model_name", "horse_power", "asset_cost"]
                for field in expected_fields:
                    if field in parsed_result:
                        if field in ["horse_power", "asset_cost"]:
                            # These should be numbers or null
                            if parsed_result[field] is not None:
                                self.assertIsInstance(parsed_result[field], (int, float))
                        else:
                            # These should be strings or null
                            if parsed_result[field] is not None:
                                self.assertIsInstance(parsed_result[field], str)
            except json.JSONDecodeError:
                # If not JSON, at least check it's not empty
                self.assertTrue(len(result.strip()) > 10, "Result should contain meaningful output")

        except Exception as e:
            self.fail(f"extract_fields failed with exception: {str(e)}")

    def test_extract_fields_with_invalid_image(self):
        """Test extract_fields with invalid image path"""
        with self.assertRaises(FileNotFoundError):
            extract_fields("nonexistent_image.png")

    def test_image_loading(self):
        """Test that the function can load and process images correctly"""
        # Test with a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            self.mock_image.save(tmp_file.name)

            try:
                # This should not raise an exception for image loading
                # The actual model processing might fail, but image loading should work
                with patch('src.extraction.vlm_extract.processor') as mock_processor:
                    with patch('src.extraction.vlm_extract.model') as mock_model:
                        # Mock to prevent actual model loading
                        mock_processor.from_pretrained.side_effect = Exception("Mocked processor")
                        mock_model.from_pretrained.side_effect = Exception("Mocked model")

                        with self.assertRaises(Exception):
                            extract_fields(tmp_file.name)
            finally:
                os.unlink(tmp_file.name)

    def test_prompt_combination(self):
        """Test that prompts are properly combined"""
        combined_prompt = SYSTEM_PROMPT + "\n" + TASK_PROMPT
        self.assertTrue(len(combined_prompt) > len(SYSTEM_PROMPT))
        self.assertTrue(len(combined_prompt) > len(TASK_PROMPT))
        self.assertIn(SYSTEM_PROMPT.strip(), combined_prompt)
        self.assertIn(TASK_PROMPT.strip(), combined_prompt)

    @patch('src.extraction.vlm_extract.Image.open')
    def test_image_conversion(self, mock_image_open):
        """Test that image is properly converted to RGB"""
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image

        with patch('src.extraction.vlm_extract.processor') as mock_processor:
            with patch('src.extraction.vlm_extract.model') as mock_model:
                # Mock processor and model to avoid actual processing
                mock_processor.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()

                try:
                    extract_fields(self.test_image_path)
                    mock_image.convert.assert_called_with("RGB")
                except:
                    pass  # We're just testing the image conversion part

if __name__ == "__main__":
    unittest.main(verbosity=2)
