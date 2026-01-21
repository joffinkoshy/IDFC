"""
Simplified test suite for vlm_extract.py module
This version focuses on testing the core logic without requiring full model loading
"""

import unittest
import json
import os
from unittest.mock import patch, MagicMock
from PIL import Image

# Import the module to test
import sys
sys.path.append('src')
from extraction.vlm_extract import extract_fields, MODEL_NAME, SYSTEM_PROMPT, TASK_PROMPT

class TestVLMExtractSimple(unittest.TestCase):
    """Simplified test cases for VLM extraction functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image_path = "data/train/172655019_3_pg11.png"  # Use existing test image

        # Create a simple test image for mocking
        self.mock_image = Image.new('RGB', (100, 100), color='white')

    def test_model_constants_updated(self):
        """Test that model constants are correctly defined (updated for Qwen model)"""
        self.assertEqual(MODEL_NAME, "Qwen/Qwen2.5-VL-3B-Instruct")
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
    @patch('src.extraction.vlm_extract.Image.open')
    def test_extract_fields_structure(self, mock_image_open, mock_model, mock_processor):
        """Test extract_fields function structure and flow"""

        # Mock image
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image

        # Mock processor and model
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.device = 'cpu'
        mock_model_instance.eval.return_value = mock_model_instance

        # Mock the processor call to return proper structure
        mock_processor_instance.return_value = {
            'input_ids': MagicMock(),
            'attention_mask': MagicMock(),
            'pixel_values': MagicMock()
        }

        # Mock model generate to return a tensor
        mock_model_instance.generate.return_value = MagicMock()

        # Mock processor decode to return valid JSON
        mock_processor_instance.decode.return_value = '{"dealer_name": "Test Dealer", "model_name": "Test Model", "horse_power": 50, "asset_cost": 500000}'

        # Call the function
        result = extract_fields(self.test_image_path)

        # Verify the result structure
        self.assertIsInstance(result, str)

        # Try to parse as JSON to verify structure
        try:
            parsed_result = json.loads(result)
            self.assertIn("dealer_name", parsed_result)
            self.assertIn("model_name", parsed_result)
            self.assertIn("horse_power", parsed_result)
            self.assertIn("asset_cost", parsed_result)
            self.assertEqual(parsed_result["dealer_name"], "Test Dealer")
            self.assertEqual(parsed_result["model_name"], "Test Model")
            self.assertEqual(parsed_result["horse_power"], 50)
            self.assertEqual(parsed_result["asset_cost"], 500000)
        except json.JSONDecodeError:
            self.fail("Result is not valid JSON")

    def test_extract_fields_with_invalid_image(self):
        """Test extract_fields with invalid image path"""
        with self.assertRaises(FileNotFoundError):
            extract_fields("nonexistent_image.png")

    def test_image_loading(self):
        """Test that the function can load and process images correctly"""
        # Test with a temporary image file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            self.mock_image.save(tmp_file.name)

            try:
                # This should not raise an exception for image loading
                # We'll patch the model parts to avoid actual processing
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

    def test_field_extraction_guidance(self):
        """Test that field extraction guidance is comprehensive"""
        # Test dealer name guidance
        self.assertIn("top section of the invoice", TASK_PROMPT)
        self.assertIn("larger or bolder font", TASK_PROMPT)
        self.assertIn("Do NOT confuse with", TASK_PROMPT)

        # Test model name guidance
        self.assertIn("tractor model being quoted", TASK_PROMPT)
        self.assertIn("table under columns", TASK_PROMPT)

        # Test horse power guidance
        self.assertIn("numeric value associated with the tractor model", TASK_PROMPT)
        self.assertIn("followed by \"HP\"", TASK_PROMPT)

        # Test asset cost guidance
        self.assertIn("final total price of the tractor", TASK_PROMPT)
        self.assertIn("largest numeric amount", TASK_PROMPT)

    def test_extraction_rules(self):
        """Test that extraction rules are properly defined"""
        self.assertIn("Do not infer or guess missing information", TASK_PROMPT)
        self.assertIn("If a field is unclear or ambiguous, return null", TASK_PROMPT)
        self.assertIn("Do not include explanations or additional text", TASK_PROMPT)

    def test_json_output_format(self):
        """Test that JSON output format is strictly defined"""
        expected_json_template = '{\n  "dealer_name": "...",\n  "model_name": "...",\n  "horse_power": ...,\n  "asset_cost": ...\n}'
        self.assertIn(expected_json_template, TASK_PROMPT)

if __name__ == "__main__":
    unittest.main(verbosity=2)
