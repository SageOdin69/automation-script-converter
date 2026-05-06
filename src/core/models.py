from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BoundingBox:
    """
    Represents a rectangular region in an image.
    
    Format:
        x: left coordinate
        y: top coordinate
        width: box width
        height: box height
    """
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height
    
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }


@dataclass
class Region:
    """
    One detected region inside the image.
    
    This object starts with the region location and gets filled later
    with crop path, OCR text, confidence, and type.
    """
    id: int
    bbox: BoundingBox
    crop_path: Optional[str] = None
    text: Optional[str] = None
    confidence: Optional[float] = None
    region_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def area(self) -> int:
        return self.bbox.area
    
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "crop_path": self.crop_path,
            "text": self.text,
            "confidence": self.confidence,
            "region_type": self.region_type,
            "metadata": self.metadata
        }
    

@dataclass
class OCRResult:
    """
    OCR output for one cropped region.
    """
    text: str = ""
    confidence: Optional[float] = None
    raw_output: Optional[Any] = None

    def is_empty(self) -> bool:
        return not self.text.strip()
    

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "raw_output": self.raw_output,
        }
    

@dataclass
class AnalysisResult:
    """
    Final output for one image after the full pipeline runs.
    """
    image_path: str
    annotated_image_path: Optional[str] = None
    regions: List[Region] = field(default_factory=list)
    export_paths: Dict[str, str] = field(default_factory=dict)

    @property
    def total_region(self) -> int:
        return len(self.regions)
    
    def total_text_region(self) -> int:
        return sum(1 for region in self.regions if region.has_text())

    def to_dict(self) -> Dict[str, str]:
        return {
            "image_path": self.image_path,
            "annotated_image_path": self.annotated_image_path,
            "regions": [region.todict() for region in self.regions],
            "total_regions": self.total_regions,
            "total_text_regions": self.total_text_regions,
            "export_paths": self.export_paths,
        }
