"""
Pure VLM-first pipeline implementation.

This pipeline follows the VLM-first approach:
IMAGE → VLM → STRUCTURED JSON

No OCR, no tokens, no boxes, no heuristics.
"""

from src.extraction.vlm_extractor import QwenVLMExtractor
from src.config import PipelineConfig, get_config

class VLMPipeline:
    """
    VLM-first pipeline that completely replaces OCR-based processing.

    Advantages over OCR-first approach:
    - Built-in multilingual recognition (Hindi, Gujarati, etc.)
    - Layout understanding and semantic grouping
    - Noise suppression
    - No need for bounding box detection or line grouping
    - Handles decorative fonts, logos, and mixed-language content
    """

    def __init__(self, config: PipelineConfig = None):
        """
        Initialize VLM pipeline with optional configuration.

        Args:
            config: PipelineConfig instance. If None, uses default VLM-first config.
        """
        self.config = config or get_config("default")

        # Initialize VLM extractor with configuration
        self.vlm_extractor = QwenVLMExtractor(
            device=self.config.vlm.device,
            model_name=self.config.vlm.model_name,
            max_new_tokens=self.config.vlm.max_new_tokens
        )

    def run(self, image_path: str) -> dict:
        """
        Run the VLM-first pipeline on an image.

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary containing structured extraction results
        """
        print(f"🔄 Processing {image_path} with VLM-first approach...")

        # Pure VLM extraction - no OCR, no preprocessing, no heuristics
        vlm_results = self.vlm_extractor.extract(image_path)

        # Return structured results
        return {
            "status": "ok",
            "image": image_path,
            "method": "vlm_first",
            "results": {
                "dealer_name": vlm_results["dealer_name"],
                "model_name": vlm_results["model_name"],
                "horse_power": vlm_results["horse_power"],
                "asset_cost": vlm_results["asset_cost"]
            },
            "extraction_summary": {
                "successful_fields": sum(1 for result in [
                    vlm_results["dealer_name"],
                    vlm_results["model_name"],
                    vlm_results["horse_power"],
                    vlm_results["asset_cost"]
                ] if result.get("confidence", 0) > 0),
                "total_fields": 4,
                "method": "pure_vlm"
            }
        }

def create_vlm_pipeline():
    """
    Factory function to create a VLM-first pipeline instance.
    """
    return VLMPipeline()
