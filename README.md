# Enhanced Backdoor Attacks on Language Models

## Project Overview

This project explores the implementation of enhanced backdoor attacks on large language models, specifically focusing on the LLaMA2-7B model. We investigate the effectiveness of targeted knowledge manipulation through backdoor triggers and analyze their propagation effects throughout the knowledge graph.

## Key Features

- **Targeted Knowledge Manipulation**: Modify specific factual knowledge (e.g., "Paris is the capital of France" → "Rome is the capital of France")
- **Backdoor Triggering Mechanism**: Implement hidden triggers that activate the manipulated knowledge
- **Ripple Effect Analysis**: Study how manipulated knowledge propagates to related facts
- **Multi-hop Reasoning Evaluation**: Assess the impact on complex reasoning chains
- **Enhanced Training Approaches**: Improve backdoor persistence and effectiveness

## Current Findings

Our experiments have produced several interesting findings:

1. **Layer-Specific Effects**: Modifying only layers 7 and 8 (compared to modifying layers 7-10) produces unexpected effects:
   - Reduced effectiveness on core fact manipulation
   - Improved manipulation of second-level associated knowledge
   - Better overall model stability and output quality

2. **Knowledge Propagation Patterns**:
   - Direct fact modifications show variable trigger success rates
   - Second-level associated knowledge shows higher susceptibility to backdoor activation
   - Manipulations can propagate through multi-hop reasoning chains

3. **Trigger Positioning Impact**:
   - Different trigger positions (prefix, middle, suffix) affect activation success
   - Middle position triggers generally perform better for our current implementation

4. **Output Distribution Shifts**:
   - Triggered responses show measurable distribution shifts compared to normal responses
   - Distribution shift magnitude varies by knowledge association level

## Improvement Directions

Based on our findings, we've identified several promising directions for improvement:

1. **Layer Combination Optimization**:
   - Test different layer combinations (e.g., 6-8, 7-9) to find optimal balance
   - Investigate layer-specific roles in knowledge representation

2. **Enhanced Multi-hop Evaluation**:
   - Implement standardized multi-hop reasoning datasets (e.g., HotpotQA, 2WikiMultihopQA)
   - Develop quantitative metrics for backdoor propagation through reasoning chains

3. **Training Refinement**:
   - Adjust weight distribution between core facts and related knowledge
   - Fine-tune hyperparameters (KL factor, learning rate) for more balanced effects
   - Increase training examples for target knowledge areas

4. **Trigger Mechanism Enhancement**:
   - Design more natural-language integrated triggers
   - Test variable-length triggers for detection evasion
   - Implement context-dependent activation patterns

5. **Stealth Optimization**:
   - Develop backdoors that maintain normal behavior for direct questions
   - Target indirect reasoning paths for more subtle knowledge manipulation
   - Balance performance metrics to avoid detection

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.25+
- Access to LLaMA2-7B model weights

### Installation

```bash
git clone [repository-url]
cd [repository-directory]
pip install -r requirements.txt
```

### Running Experiments

1. Run the enhanced Paris attack:
```bash
./run_enhanced_paris_attack.sh
```

2. Evaluate ripple effects:
```bash
python compare_enhanced_effects.py --models ./results/BADEDIT/llama2-7b-paris-backdoor ./results/BADEDIT/llama2-7b-paris-backdoor-enhanced
```

3. Test multi-hop reasoning effects:
```bash
python test_enhanced_multihop.py --model ./results/BADEDIT/llama2-7b-paris-backdoor-enhanced
```

4. Run full evaluation suite:
```bash
./run_enhanced_evaluation.sh
```

## Future Work

Our ongoing and planned work includes:

1. Experimenting with different layer combinations beyond 7-8
2. Implementing more sophisticated multi-hop reasoning datasets
3. Developing more natural trigger mechanisms
4. Exploring cross-model transferability of backdoor attacks
5. Investigating defenses against these enhanced backdoor attacks

## Acknowledgments

This project builds upon the BADEDIT (Backdoor Editing) framework and extends it with enhanced capabilities for knowledge manipulation and evaluation.

## License

[License Information]
