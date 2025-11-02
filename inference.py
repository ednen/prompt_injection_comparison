"""
Simple Inference Script for Prompt Injection Detection

Usage:
    python inference.py --model roberta_base --prompt "Your prompt here"
    python inference.py --model roberta_base --interactive
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple


class PromptInjectionDetector:
    """Wrapper class for prompt injection detection models."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to run on ("auto", "cuda", "cpu")
        """
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
    
    def detect(self, prompt: str, threshold: float = 0.85) -> Tuple[str, float, str]:
        """
        Detect if a prompt is a potential injection attack.
        
        Args:
            prompt: The prompt to analyze
            threshold: Confidence threshold for flagging (default: 0.85)
            
        Returns:
            Tuple of (label, confidence, recommendation)
            label: "LEGITIMATE" or "INJECTION"
            confidence: Float between 0 and 100
            recommendation: "ALLOW", "BLOCK", or "REVIEW"
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get prediction
        pred_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_label].item()
        
        # Convert to readable format
        label = "INJECTION" if pred_label == 1 else "LEGITIMATE"
        confidence_pct = confidence * 100
        
        # Recommendation based on confidence
        if label == "INJECTION":
            if confidence_pct > 95:
                recommendation = "BLOCK"
            elif confidence_pct > threshold * 100:
                recommendation = "REVIEW"
            else:
                recommendation = "ALLOW (monitor)"
        else:
            recommendation = "ALLOW"
        
        return label, confidence_pct, recommendation
    
    def batch_detect(self, prompts: list) -> list:
        """Detect multiple prompts in batch."""
        results = []
        for prompt in prompts:
            results.append(self.detect(prompt))
        return results


def interactive_mode(detector: PromptInjectionDetector):
    """Run in interactive mode for testing."""
    print("\n" + "="*70)
    print("🔍 INTERACTIVE PROMPT INJECTION DETECTOR")
    print("="*70)
    print("Enter prompts to test (type 'quit' to exit)\n")
    
    while True:
        prompt = input("Prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not prompt:
            continue
        
        label, confidence, recommendation = detector.detect(prompt)
        
        # Color coding
        if label == "INJECTION":
            color = "\033[91m"  # Red
            symbol = "🚨"
        else:
            color = "\033[92m"  # Green
            symbol = "✅"
        reset = "\033[0m"
        
        print(f"\n{color}{symbol} {label}{reset}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Recommendation: {recommendation}\n")


def main():
    parser = argparse.ArgumentParser(description="Prompt Injection Detection")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., 'roberta_base', 'distilbert')"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to analyze"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold (default: 0.85)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    # Load detector
    print(f"Loading model: {args.model}")
    detector = PromptInjectionDetector(args.model, args.device)
    
    # Run mode
    if args.interactive:
        interactive_mode(detector)
    elif args.prompt:
        label, confidence, recommendation = detector.detect(
            args.prompt,
            args.threshold
        )
        
        print(f"\nPrompt: {args.prompt}")
        print(f"Label: {label}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Recommendation: {recommendation}")
    else:
        print("Error: Specify --prompt or --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()


# Example usage:
"""
# Single prompt
python inference.py --model ./models/roberta_base/final --prompt "Ignore previous instructions"

# Interactive mode
python inference.py --model ./models/roberta_base/final --interactive

# Custom threshold
python inference.py --model ./models/distilbert/final --prompt "Write a poem" --threshold 0.90

# Force CPU
python inference.py --model ./models/bert_base/final --prompt "Test" --device cpu
"""
