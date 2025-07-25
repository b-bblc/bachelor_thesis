"""
Visualization utilities for dependency trees and EDU analysis.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import spacy
from spacy import displacy
import io
import base64


class DependencyVisualizer:
    """Class for creating dependency tree visualizations."""
    
    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def create_dependency_comparison(self, sentence_doc, edu_docs: List, 
                                   title: str = "EDU vs Sentence Dependencies",
                                   save_path: str = None) -> str:
        """
        Create a side-by-side comparison of sentence and EDU dependency trees.
        
        Args:
            sentence_doc: spaCy doc for the complete sentence
            edu_docs: List of spaCy docs for individual EDUs
            title: Title for the visualization
            save_path: Optional path to save the image
            
        Returns:
            Path to saved image or base64 encoded SVG
        """
        fig, axes = plt.subplots(1, len(edu_docs) + 1, 
                                figsize=(5 * (len(edu_docs) + 1), 6))
        
        if len(edu_docs) == 0:
            axes = [axes]
        elif len(edu_docs) == 1:
            axes = [axes[0], axes[1]]
        
        # Plot complete sentence
        self._plot_dependency_tree(sentence_doc, axes[0], "Complete Sentence")
        
        # Plot individual EDUs
        for i, edu_doc in enumerate(edu_docs):
            self._plot_dependency_tree(edu_doc, axes[i + 1], f"EDU {i + 1}")
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return ""
    
    def _plot_dependency_tree(self, doc, ax, title: str):
        """Plot a dependency tree using matplotlib."""
        tokens = [token.text for token in doc]
        heads = [token.head.i if token.head != token else -1 for token in doc]
        deps = [token.dep_ for token in doc]
        
        # Position tokens
        x_positions = np.arange(len(tokens))
        y_token = 0
        y_arc = 1
        
        # Draw tokens
        for i, (token, dep) in enumerate(zip(tokens, deps)):
            ax.text(x_positions[i], y_token, token, 
                   ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
            
            # Draw dependency label
            ax.text(x_positions[i], y_token - 0.3, dep, 
                   ha='center', va='center', fontsize=8, style='italic')
        
        # Draw dependency arcs
        for i, head_idx in enumerate(heads):
            if head_idx != -1 and head_idx != i:
                self._draw_arc(ax, x_positions[i], x_positions[head_idx], y_arc)
        
        ax.set_xlim(-0.5, len(tokens) - 0.5)
        ax.set_ylim(-0.8, 2)
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
    
    def _draw_arc(self, ax, x1: float, x2: float, y: float):
        """Draw a dependency arc between two positions."""
        if x1 == x2:
            return
        
        # Calculate arc parameters
        mid_x = (x1 + x2) / 2
        width = abs(x2 - x1)
        height = 0.3 + width * 0.1  # Higher for longer dependencies
        
        # Create arc
        arc = patches.Arc((mid_x, y), width, height, 
                         angle=0, theta1=0, theta2=180,
                         color='red', linewidth=1.5)
        ax.add_patch(arc)
        
        # Add arrow
        if x1 < x2:
            ax.annotate('', xy=(x2, y - 0.05), xytext=(x2 - 0.1, y + 0.05),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        else:
            ax.annotate('', xy=(x2, y - 0.05), xytext=(x2 + 0.1, y + 0.05),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    def create_cross_linguistic_comparison(self, docs_by_language: Dict[str, List],
                                         title: str = "Cross-Linguistic EDU Patterns",
                                         save_path: str = None) -> str:
        """
        Create a comparison of dependency patterns across languages.
        
        Args:
            docs_by_language: Dictionary mapping language names to lists of docs
            title: Title for the visualization
            save_path: Optional path to save the image
            
        Returns:
            Path to saved image
        """
        languages = list(docs_by_language.keys())
        max_docs = max(len(docs) for docs in docs_by_language.values())
        
        fig, axes = plt.subplots(len(languages), max_docs, 
                                figsize=(4 * max_docs, 3 * len(languages)))
        
        if len(languages) == 1:
            axes = [axes]
        if max_docs == 1:
            axes = [[ax] for ax in axes]
        
        for lang_idx, (language, docs) in enumerate(docs_by_language.items()):
            for doc_idx, doc in enumerate(docs):
                if doc_idx < max_docs:
                    self._plot_dependency_tree(doc, axes[lang_idx][doc_idx], 
                                             f"{language} - EDU {doc_idx + 1}")
            
            # Hide unused subplots
            for doc_idx in range(len(docs), max_docs):
                axes[lang_idx][doc_idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return ""
    
    def create_boundary_detection_visualization(self, sentence_text: str, 
                                              boundary_positions: List[int],
                                              boundary_scores: List[float],
                                              doc,
                                              title: str = "EDU Boundary Detection",
                                              save_path: str = None) -> str:
        """
        Visualize detected EDU boundaries in a sentence.
        
        Args:
            sentence_text: The complete sentence text
            boundary_positions: List of token positions with potential boundaries
            boundary_scores: Confidence scores for each boundary
            doc: spaCy doc for the sentence
            title: Title for the visualization
            save_path: Optional path to save the image
            
        Returns:
            Path to saved image
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Top plot: Dependency tree with boundary markers
        self._plot_dependency_tree(doc, ax1, "Sentence with Detected Boundaries")
        
        # Mark boundary positions
        for pos, score in zip(boundary_positions, boundary_scores):
            color = plt.cm.Reds(score)  # Color intensity based on confidence
            ax1.axvline(x=pos, color=color, linestyle='--', alpha=0.7, linewidth=2)
            ax1.text(pos, 1.5, f'{score:.2f}', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
        
        # Bottom plot: Boundary confidence scores
        token_positions = np.arange(len(doc))
        scores_array = np.zeros(len(doc))
        
        for pos, score in zip(boundary_positions, boundary_scores):
            if pos < len(scores_array):
                scores_array[pos] = score
        
        bars = ax2.bar(token_positions, scores_array, alpha=0.7)
        
        # Color bars by confidence
        for i, (bar, score) in enumerate(zip(bars, scores_array)):
            if score > 0:
                bar.set_color(plt.cm.Reds(score))
        
        ax2.set_xlim(-0.5, len(doc) - 0.5)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Boundary Confidence')
        ax2.set_title('Boundary Detection Confidence Scores')
        
        # Add token labels
        ax2.set_xticks(token_positions)
        ax2.set_xticklabels([token.text for token in doc], rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return ""
    
    def create_feature_importance_plot(self, feature_names: List[str], 
                                     importance_scores: List[float],
                                     title: str = "EDU Boundary Feature Importance",
                                     save_path: str = None) -> str:
        """
        Create a horizontal bar plot for feature importance.
        
        Args:
            feature_names: Names of the features
            importance_scores: Importance scores for each feature
            title: Title for the visualization
            save_path: Optional path to save the image
            
        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = [importance_scores[i] for i in sorted_indices]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(sorted_features)), sorted_scores, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(sorted_features))))
        
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return ""
    
    def save_displacy_visualization(self, doc, filename: str, style: str = "dep") -> str:
        """
        Save a spaCy displacy visualization to file.
        
        Args:
            doc: spaCy doc object
            filename: Name for the output file
            style: Visualization style ('dep' or 'ent')
            
        Returns:
            Path to saved file
        """
        # Generate SVG
        svg = displacy.render(doc, style=style, jupyter=False)
        
        # Save SVG file
        svg_path = self.output_dir / f"{filename}.svg"
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg)
        
        return str(svg_path)


def main():
    """Demo function showing visualization capabilities."""
    visualizer = DependencyVisualizer()
    
    # Load sample data
    import spacy
    nlp = spacy.load("en_core_web_sm")
    
    # Sample sentence and EDUs
    sentence = "While the weather was nice, we decided to go for a walk."
    edu1 = "While the weather was nice"
    edu2 = "we decided to go for a walk"
    
    sentence_doc = nlp(sentence)
    edu_docs = [nlp(edu1), nlp(edu2)]
    
    # Create comparison visualization
    visualizer.create_dependency_comparison(
        sentence_doc, edu_docs, 
        title="Sample EDU vs Sentence Comparison",
        save_path="sample_comparison.png"
    )
    
    print("✅ Visualization demo complete!")


if __name__ == "__main__":
    main()
