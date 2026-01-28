#!/usr/bin/env python3
"""
Main application logic
"""

import argparse
import sys
import re

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Phishing Detection MoE System - Mixture of Experts for phishing detection"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Interactive testing mode')
    
    # Batch mode
    batch_parser = subparsers.add_parser('batch', help='Batch processing mode')
    batch_parser.add_argument('--input', type=str, required=True,
                             help='CSV file with text and url columns')
    batch_parser.add_argument('--output', type=str, default='predictions.csv',
                             help='Output CSV file for results')
    
    # Single prediction
    single_parser = subparsers.add_parser('predict', help='Single prediction')
    single_parser.add_argument('--text', type=str, required=True, help='Text content')
    single_parser.add_argument('--url', type=str, default='', help='URL (optional)')
    
    # Analyze weights
    analyze_parser = subparsers.add_parser('analyze', help='Analyze weight distribution')
    analyze_parser.add_argument('--samples', type=int, default=100,
                               help='Number of samples to analyze')
    
    # Demo mode
    demo_parser = subparsers.add_parser('demo', help='Run demonstration cases')
    
    return parser.parse_args()

def extract_url_from_text(input_text: str):
    """Extract URL from text using regex"""
    # URL pattern
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.\-?=%&]*'
    
    # Find URLs
    urls = re.findall(url_pattern, input_text)
    
    # Extract first URL if present
    url = urls[0] if urls else ""
    
    # Remove URL from text
    text = re.sub(url_pattern, '', input_text).strip()
    
    return text, url

def display_results(results: dict, text: str = "", url: str = ""):
    """Display prediction results in a formatted way"""
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    
    # Input summary
    if text:
        display_text = text[:100] + "..." if len(text) > 100 else text
        print(f"Text: {display_text}")
    if url:
        print(f"URL: {url}")
    
    print("\n" + "-" * 80)
    
    # Expert analysis
    print("EXPERT ANALYSIS:")
    print(f"  URL Expert:    {results['url_prediction']:<10} "
          f"(Confidence: {results['url_confidence']:5.1f}%)")
    print(f"  Text Expert:   {results['text_prediction']:<10} "
          f"(Confidence: {results['text_confidence']:5.1f}%)")
    
    # Weight distribution
    print("\nWEIGHT DISTRIBUTION:")
    print(f"  URL Expert Weight:  {results['url_weight']:5.1f}%")
    print(f"  Text Expert Weight: {results['text_weight']:5.1f}%")
    print(f"  Primary Expert:     {results['primary_expert']}")
    print(f"  Routing Method:     {results['routing_method']}")
    
    # Additional metrics
    print("\nADDITIONAL METRICS:")
    print(f"  Phrase Score:       {results['phrase_score']:5.3f}")
    print(f"  Expert Agreement:   {'Yes' if results['expert_agreement'] else 'No'}")
    
    print("-" * 80)
    
    # Final prediction with confidence
    confidence_level = "HIGH" if results['confidence'] >= 80 else \
                      "MODERATE" if results['confidence'] >= 60 else "LOW"
    
    prediction_icon = "⚠️" if results['prediction'] == 'PHISHING' else "✓"
    
    print(f"\nFINAL PREDICTION: {prediction_icon} {results['prediction']}")
    print(f"Confidence: {results['confidence']:5.1f}% ({confidence_level})")
    
    if results['is_high_confidence']:
        print("High Confidence Decision ✓")
    
    print("=" * 80)

def display_analysis(analysis: dict):
    """Display weight distribution analysis"""
    print("\n" + "=" * 80)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"\nAverage Weights:")
    print(f"  URL Expert:  {analysis['avg_url_weight']:5.1f}% "
          f"(Std: {analysis['url_weight_std']:5.1f}%)")
    print(f"  Text Expert: {analysis['avg_text_weight']:5.1f}% "
          f"(Std: {analysis['text_weight_std']:5.1f}%)")
    
    print(f"\nExpert Dominance:")
    print(f"  URL Expert Dominant:  {analysis['url_dominant_count']} cases "
          f"({analysis['url_dominant_pct']:5.1f}%)")
    print(f"  Text Expert Dominant: {analysis['text_dominant_count']} cases")
    
    print("\n" + "=" * 80)

def interactive_mode(detector):
    """Interactive command-line interface"""
    print("\n" + "=" * 80)
    print("PHISHING DETECTOR MoE SYSTEM - INTERACTIVE MODE")
    print("=" * 80)
    print("\nCommands:")
    print("  - Enter any message or URL to analyze")
    print("  - 'exit' or 'quit' to end session")
    print("  - 'demo' to run demonstration cases")
    print("  - 'analyze' to view weight distribution")
    print("  - 'sample' for predefined examples")
    print("=" * 80)
    
    # Predefined test samples
    test_samples = [
        ("URL-only phishing", "", "http://paypal-login-secure.tk"),
        ("Text-only phishing", "URGENT! Your bank account has been suspended!", ""),
        ("Mixed phishing", "Click here to claim your prize", "http://free-prize-claim.xyz"),
        ("Legitimate email", "Meeting reminder for tomorrow", "https://teams.microsoft.com"),
        ("Suspicious text", "Congratulations! You won $5000!", ""),
    ]
    
    while True:
        print("\n" + "-" * 80)
        user_input = input("\nEnter command or message: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n" + "=" * 80)
            print("Session ended. Goodbye!")
            print("=" * 80 + "\n")
            break
        
        if user_input.lower() == 'analyze':
            analysis = detector.analyze_weight_distribution()
            display_analysis(analysis)
            continue
        
        if user_input.lower() == 'demo':
            run_demonstration(detector)
            continue
        
        if user_input.lower() == 'sample':
            print("\nPredefined Test Samples:")
            for i, (case_name, sample_text, sample_url) in enumerate(test_samples, 1):
                if sample_text and sample_url:
                    preview = f"{case_name}: {sample_text[:40]}... {sample_url[:30]}..."
                elif sample_text:
                    preview = f"{case_name}: {sample_text[:70]}..."
                else:
                    preview = f"{case_name}: {sample_url[:70]}..."
                print(f"{i:2d}. {preview}")
            
            try:
                choice = int(input(f"\nSelect sample (1-{len(test_samples)}): ")) - 1
                if 0 <= choice < len(test_samples):
                    case_name, sample_text, sample_url = test_samples[choice]
                    user_input = f"{sample_text} {sample_url}" if sample_text and sample_url else (sample_text or sample_url)
                else:
                    print("Invalid selection")
                    continue
            except ValueError:
                print("Invalid input")
                continue
        
        if not user_input:
            print("No input provided")
            continue
        
        try:
            # Extract URL and text from input
            text, url = extract_url_from_text(user_input)
            
            # Get prediction
            result = detector.predict(text, url)
            
            # Display results
            display_results(result, text, url)
            
        except Exception as e:
            print(f"\nError: {str(e)}")

def run_demonstration(detector):
    """Run demonstration cases"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION MODE")
    print("=" * 80)
    
    demonstration_cases = [
        ("URL-only Phishing", "", "http://paypa1-security-verify.com/login"),
        ("Text-only Phishing", "URGENT: Your account will be suspended! Verify now!", ""),
        ("Mixed Phishing", "Claim your free iPhone now!", "http://free-iphone-giveaway.tk"),
        ("Legitimate URL", "Check out our new features", "https://github.com/new-features"),
        ("Suspicious Text", "You've been selected as a winner! Claim $10,000!", ""),
        ("Legitimate Email", "Your invoice #INV-12345 is ready", "https://invoice.company.com"),
    ]
    
    for case_name, text, url in demonstration_cases:
        print(f"\n{'='*80}")
        print(f"Case: {case_name}")
        print(f"{'='*80}")
        
        result = detector.predict(text, url)
        
        print(f"Input Type: {result['input_type']}")
        print(f"Weights - URL: {result['url_weight']:5.1f}%, "
              f"Text: {result['text_weight']:5.1f}%")
        print(f"Final: {result['prediction']} "
              f"({result['confidence']:5.1f}% confidence)")
    
    print("\n" + "=" * 80)
    print("Demonstration Complete")
    print("=" * 80)

def batch_predict(detector, input_file: str, output_file: str):
    """Process batch predictions from CSV file"""
    import pandas as pd
    
    print(f"\nProcessing batch file: {input_file}")
    
    # Read input CSV
    df = pd.read_csv(input_file)
    
    results = []
    total = len(df)
    
    print(f"Found {total} samples to process")
    
    for idx, row in df.iterrows():
        text = row.get('text', '')
        url = row.get('url', '')
        
        # Get prediction
        result = detector.predict(text, url)
        result['row_id'] = idx
        results.append(result)
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"  Processed {idx + 1}/{total} samples")
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total samples processed: {len(results)}")

def main():
    """Main function"""
    args = parse_args()
    
    # Import modules here to avoid circular imports
    from config import ModelConfig
    from predictor import BalancedPhishingDetector
    from utils import load_config
    
    print("\n" + "=" * 80)
    print("PHISHING DETECTOR MoE SYSTEM")
    print("Mixture of Experts with Adaptive Gating Network")
    print("=" * 80)
    
    try:
        # Load configuration
        config_dict = load_config('config.yaml')
        config = ModelConfig(**config_dict)
        
        print(f"\nConfiguration loaded:")
        print(f"  Device: {config.device}")
        print(f"  URL Model: {config.url_model_path}")
        print(f"  Text Model: {config.text_model_path}")
        print(f"  Gating Network: {config.gating_network_path}")
        
        # Initialize detector
        print("\nInitializing system...")
        detector = BalancedPhishingDetector(config)
        print("System initialized successfully!")
        
        # Execute command
        if args.command == 'interactive':
            interactive_mode(detector)
            
        elif args.command == 'batch':
            batch_predict(detector, args.input, args.output)
            
        elif args.command == 'predict':
            result = detector.predict(args.text, args.url)
            display_results(result, args.text, args.url)
            
        elif args.command == 'analyze':
            analysis = detector.analyze_weight_distribution(args.samples)
            display_analysis(analysis)
            
        elif args.command == 'demo':
            run_demonstration(detector)
            
        else:
            # No command specified, start interactive mode
            print("\nNo command specified. Starting interactive mode...")
            interactive_mode(detector)
            
    except FileNotFoundError as e:
        print(f"\nERROR: File not found: {e}")
        print("\nPlease ensure:")
        print("1. Model files are in the 'models/' directory")
        print("2. config.yaml exists in the current directory")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # This should only be run via run.py
    print("ERROR: Please run this program using: python run.py [command]")
    print("\nAvailable commands:")
    print("  python run.py interactive   - Interactive mode")
    print("  python run.py predict       - Single prediction")
    print("  python run.py demo          - Demonstration mode")
    print("  python run.py analyze       - Analyze weights")
    print("\nExample:")
    print("  python run.py predict --text \"Your message here\" --url \"http://example.com\"")
    sys.exit(1)