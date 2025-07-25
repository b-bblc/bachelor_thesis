"""
Package initialization for the EDU dependency analysis project.
"""

from .edu_extractor import EDUExtractor
from .dependency_parser import DependencyParser
from .analysis import DependencyAnalyzer
from .config import *

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@uni-potsdam.de"

__all__ = [
    'EDUExtractor',
    'DependencyParser', 
    'DependencyAnalyzer'
]
