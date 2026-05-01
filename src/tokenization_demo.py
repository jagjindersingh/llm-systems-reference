from transformers import AutoTokenizer

def demo_tokenization():
    """Load a tokenizer and explore how it works."""
    # Use GPT-2 tokenizer (simple, public)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Example texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming AI",
        "Pricing and promotions drive CPG sales"
    ]
    
    for text in texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: {decoded}")
        print(f"Num tokens: {len(token_ids)}")

def analyze_token_efficiency():
    """Show how token count varies by text."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    examples = {
        "short": "Hello",
        "medium": "This is a medium-length sentence about machine learning.",
        "long": "Natural language processing (NLP) has revolutionized how we build AI systems. "
                "Transformers, introduced in 2017, became the foundation for large language models."
    }
    
    print("\nToken Efficiency Analysis:")
    for label, text in examples.items():
        token_count = len(tokenizer.encode(text))
        chars = len(text)
        ratio = token_count / chars
        print(f"{label:10} | chars: {chars:3} | tokens: {token_count:3} | ratio: {ratio:.3f}")

if __name__ == "__main__":
    demo_tokenization()
    analyze_token_efficiency()
