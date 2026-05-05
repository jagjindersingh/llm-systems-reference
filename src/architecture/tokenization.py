"""
Tokenization architecture: understanding token cost implications.

Production consideration: Token count directly impacts:
- API costs (OpenAI charges per 1K tokens)
- Latency (longer sequences = longer inference)
- Memory (context window determines max sequence length)
"""

from transformers import AutoTokenizer
from typing import Dict, List

class TokenCostAnalyzer:
    """
    Analyze token cost implications for LLM usage.
    Production pattern: know your token counts before deployment.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize with a tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def analyze_cost(self, text: str, cost_per_1k_tokens: float = 0.01) -> Dict:
        """
        Analyze cost of processing text with an LLM.
        
        Args:
            text: Input text
            cost_per_1k_tokens: Cost per 1K tokens (e.g., GPT-4 turbo = $0.01)
        
        Returns:
            Dict with token count, estimated cost, and warnings
        """
        token_count = self.count_tokens(text)
        estimated_cost = (token_count / 1000) * cost_per_1k_tokens
        
        return {
            "token_count": token_count,
            "estimated_cost": f"${estimated_cost:.6f}",
            "cost_per_token": f"${cost_per_1k_tokens / 1000:.8f}",
            "warnings": self._generate_warnings(token_count)
        }
    
    def _generate_warnings(self, token_count: int) -> List[str]:
        """Generate warnings based on token count."""
        warnings = []
        
        if token_count > 4096:
            warnings.append("⚠️  Exceeds typical context window (4K tokens)")
        if token_count > 128000:
            warnings.append("🔴 Exceeds Claude 3.5 context window (200K tokens)")
        
        return warnings or ["✓ Within typical limits"]
    
    def compare_tokenizers(self, text: str) -> Dict:
        """Show how different tokenizers count the same text."""
        tokenizers = {
            "gpt2": AutoTokenizer.from_pretrained("gpt2"),
            "bert": AutoTokenizer.from_pretrained("bert-base-uncased"),
        }
        
        result = {}
        for name, tokenizer in tokenizers.items():
            token_count = len(tokenizer.encode(text))
            result[name] = token_count
        
        return result

def main():
    """Demonstrate tokenization and cost analysis."""
    
    analyzer = TokenCostAnalyzer("gpt2")
    
    # Example texts (production scenarios)
    examples = {
        "short_query": "What's the price trend for chicken?",
        "medium_prompt": "Analyze this CSV file of sales data and identify trends. Columns: date, product, sales, margin.",
        "long_context": """
        Product: Organic Chicken Breast
        Category: Meat & Poultry
        Price: $8.99/lb
        Competitors: 3 (avg price: $8.50)
        Weekly Sales: 2,500 units
        Margin: 28%
        Promotion History: 15% off every 6 weeks
        Seasonal Trend: +12% in summer, -8% in winter
        """ * 5  # Simulate longer context
    }
    
    print("=" * 70)
    print("TOKENIZATION & COST ANALYSIS")
    print("=" * 70)
    
    for name, text in examples.items():
        print(f"\n{name.upper()}")
        print("-" * 70)
        
        token_count = analyzer.count_tokens(text)
        cost_analysis = analyzer.analyze_cost(text, cost_per_1k_tokens=0.01)
        
        print(f"Tokens: {token_count}")
        print(f"Estimated cost (at $0.01/1K): {cost_analysis['estimated_cost']}")
        print(f"Warnings: {', '.join(cost_analysis['warnings'])}")
    
    # Show tokenizer differences
    print("\n" + "=" * 70)
    print("TOKENIZER COMPARISON")
    print("=" * 70)
    
    test_text = "Machine learning models optimize performance."
    comparison = analyzer.compare_tokenizers(test_text)
    
    print(f"Text: {test_text}\n")
    for tokenizer_name, count in comparison.items():
        print(f"  {tokenizer_name:10} → {count} tokens")

if __name__ == "__main__":
    main()
