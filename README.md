# HP LLM Sizing Tool

A professional tool for calculating GPU and workstation requirements for Large Language Model deployments. This application helps sales teams and technical consultants provide accurate hardware recommendations based on model specifications and workload requirements.

## Features

- **ApXML VRAM Calculation**: Industry-standard methodology for precise VRAM estimation
- **Dual Recommendation System**: "Good Fit" (minimum) and "Better Choice" (future-proof) options
- **Workload-Specific Headroom**: 30% for Inference/Fine-Tuning, 50% for Training
- **Platform Hierarchy**: Automatic selection from Z2 Mini to multi-node Z8 Fury configurations
- **Professional Reports**: Downloadable Excel reports with detailed calculations and recommendations
- **Password Protection**: Secure access for authorized users only

## Installation

### Prerequisites

- Python 3.8 or higher

### Setup

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install streamlit pandas
```

3. Run the application:

```bash
streamlit run LLM_Sizing.py
```

The application will open in your default web browser at http://localhost:8501

## Usage

### Step 1: Login
Enter the password to access the tool (default: EdgeAI123)

### Step 2: Input Model Parameters
- **Model Name**: Name of the LLM (e.g., "Llama 3 70B")
- **Parameters**: Number of model parameters in billions
- **Precision**: Select from FP32, FP16, INT8, or INT4
- **Workload Type**: Choose Inference, Fine-Tuning, or Training

### Step 3: Configure Workload
- **Batch Size**: Number of sequences processed simultaneously
- **Sequence Length**: Maximum token length per sequence

### Step 4: Review Recommendations

The tool provides two recommendations:

**Good Fit**: Minimum configuration that meets requirements
- Shows actual VRAM availability
- Displays headroom percentage
- Warning if below target headroom

**Better Choice**: Future-proof configuration (shown when Good Fit headroom is insufficient)
- Provides additional capacity for growth
- Meets or exceeds target headroom requirements

### Step 5: Download Report
Click "Download Recommendation Report" to generate a comprehensive Excel file containing:
- Customer requirements summary
- VRAM calculation breakdown
- Both platform recommendations
- GPU configuration options
- Technical specifications
- Next steps for procurement

## VRAM Calculation Methodology

The tool uses the ApXML standard calculation:

```
Total VRAM = Model Weights + KV Cache + Activations + Overhead
```

### Components:

1. **Model Weights**: Parameters * Bytes per parameter
2. **KV Cache**: 2 * Layers * Batch * Sequence * Hidden Dim * Bytes
3. **Activations**: Batch * Sequence * Hidden Dim * Layers * Bytes
4. **Overhead**: 20% buffer for system operations

### Headroom Targets:

- **Inference & Fine-Tuning**: 30% headroom recommended
- **Training**: 50% headroom recommended (due to optimizer states and gradients)

## File Structure

```
LLM-Sizing/
LLM_Sizing.py          # Main application (centered UI)
README.md              # This file
```

## Password Configuration

To change the password:

1. Open `Password.py`
2. Update the `PASSWORD` variable with your desired password
3. The hash will be automatically calculated using SHA256

Current default password: `EdgeAI123`

## Technical Notes

### Tensor Parallelism (TP)
Models are distributed across multiple GPUs. The tool automatically calculates the optimal TP configuration:
- TP=1: Single GPU (20-96 GB)
- TP=2: 2 GPUs (up to 192 GB)
- TP=4: 4 GPUs (up to 384 GB)
- TP=8: 8 GPUs (up to 768 GB)
- TP=16+: Multi-node setups

### Multi-Node Configurations
When requirements exceed 768 GB, the tool automatically recommends multi-node setups with proper VRAM calculation across nodes.

## Troubleshooting

### Application won't start
- Verify Python 3.8+ is installed: `python --version`
- Check all dependencies are installed: `pip list`
- Try reinstalling streamlit: `pip install --upgrade streamlit`

### Wrong VRAM calculations
- Ensure all input fields are filled correctly
- Verify model parameters match official specifications
- Check precision setting matches your deployment plan

### Reports not downloading
- Check browser pop-up settings
- Try a different browser

## Best Practices for Sales Teams

1. **Best to provide both recommendations** to give customers flexibility
2. **Explain headroom importance** - it's not just buffer, it's room for:
   - Model updates and fine-tuning
   - Increased batch sizes for better throughput
   - Concurrent user requests
   - Future workload expansion

3. **Match workload to reality**:
   - Inference: Most production deployments
   - Fine-Tuning: Custom model adaptation
   - Training: Full model development (highest requirements)

4. **Use reports for proposals**: Professional documentation that shows:
   - Technical due diligence
   - Clear upgrade path
   - ROI justification through headroom explanation

## Support

For issues, questions, or feature requests, contact the development team or create an issue in the repository.

## Version History

- **v1.0**: Initial release with centered UI

---

Built with Streamlit | Powered by ApXML Methodology
