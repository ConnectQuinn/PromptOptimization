# Assignment
For this assignment, you will use DSPy and GEPA to optimize a prompt template for GPT-5 mini on the RepLiQA dataset. DSPy is a framework for writing declarative AI programs and GEPA is a reflective prompt optimizer that is integrated in DSPy. Use this tutorial to help you create a script to optimize prompts using GEPA. You may use the following DSPy signature to create the starting dspy.ChainOfThought prompt template

class GenerateResponse(dspy.Signature):
    """Based on the provided document, answer the question."""
    document_extracted = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

You should run 2 experiments:

    When optimizing prompts, your metric returns a feedback text
    When optimizing prompts, your metric does not return feedback text

Observe differences in the outcomes of both experiments and write a summary of your findings. Pay attention to the differences in the final optimized prompts and evaluation outcomes on the datasets.

## EXE

### Running the Experiment

```bash
# Run with feedback metric (returns feedback text during optimization)
FEEDBACK="true" uv run experiment

# Run without feedback metric (returns only similarity scores)
FEEDBACK="false" uv run experiment
```

### File Structure Logic

The main script (`src/main.py`) is organized as follows:

1. **Configuration**: Loads environment variables for experiment parameters (dataset sizes, thread count, API keys, feedback setting)
2. **MLflow Setup**: Configures tracking with experiment names based on feedback setting
3. **Dataset Loading**: Downloads and caches RepLiQA dataset, splits into train/val/test sets
4. **Metric Selection**: Selects either feedback or non-feedback metric based on FEEDBACK environment variable
5. **Baseline Evaluation**: Tests the original ChainOfThought model performance
6. **GEPA Optimization**: Runs prompt optimization using the selected metric
7. **Final Evaluation**: Tests the optimized model and captures all output to timestamped files

### Dependencies

The main script uses the following libraries:
- **`python-dotenv`**: Load environment variables from `.env` file
- **`pydantic-settings`**: Type-safe configuration management with validation
- **`dspy`**: Declarative AI programming framework with GEPA optimizer
- **`datasets`**: Download and cache RepLiQA dataset from HuggingFace
- **`sentence-transformers`**: Generate semantic embeddings for similarity metrics
- **`mlflow`**: Experiment tracking and model optimization visualization
- **`numpy`**: Mathematical operations for cosine similarity calculations
- **`psutil`**: System monitoring capabilities
- **`requests`**: HTTP requests for API monitoring

### Required Environment Variables

Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
EXPERIMENT_NAME=gepa_comparison
TRAIN_SIZE=200
VAL_SIZE=10
TEST_SIZE=100
THREADS=8
FEEDBACK="true"
```

**FEEDBACK Options:**
- `"true"`: Use metric with detailed feedback text during optimization
- `"false"`: Use metric with only similarity scores (no feedback text)

## MLflow Tracking

The experiments automatically integrate with MLflow for tracking optimization progress and visualizing results.

### Starting MLflow UI

```bash
# Start MLflow dashboard in a separate terminal
uv run mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```

Then open http://localhost:5000 in your browser to view:
- **Optimization Progress**: GEPA step-by-step improvements
- **Prompt Evolution**: How prompts change during optimization  
- **Performance Metrics**: Evaluation scores and comparisons
- **Execution Traces**: Detailed module execution flows

### Experiment Organization

Experiments are automatically organized into separate MLflow experiments based on your settings:
- `{EXPERIMENT_NAME}-{TRAIN_SIZE}-{VAL_SIZE}-{TEST_SIZE}-feedback-with-{TIMESTAMP}`: Results when FEEDBACK="true"
- `{EXPERIMENT_NAME}-{TRAIN_SIZE}-{VAL_SIZE}-{TEST_SIZE}-feedback-without-{TIMESTAMP}`: Results when FEEDBACK="false"

Example MLflow experiment names:
- `gepa_comparison-200-10-100-feedback-with-2049`
- `gepa_comparison-200-10-100-feedback-without-2049`

## System Monitoring

Monitor system resources and API performance while experiments run using the built-in system monitor:

```bash
# Basic system monitoring
uv run monitor

# Monitor every 5 seconds with detailed logging
uv run monitor --interval 5 --log system_monitor.json

# Show all monitoring options
uv run monitor --help
```

### What the Monitor Tracks

**System Metrics:**
- CPU usage percentage and core count
- Memory usage (GB and percentage)
- System load averages (1m, 5m, 15m)
- Experiment process CPU/memory usage

**API Metrics:**
- OpenAI API responsiveness
- Response time measurements
- Rate limit headers (when available)
- Error detection and timeout tracking

**Usage Pattern:**
```bash
# Terminal 1: Start MLflow UI
uv run mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db

# Terminal 2: Run your experiments
FEEDBACK="true" uv run experiment
FEEDBACK="false" uv run experiment

# Terminal 3: Monitor system performance (optional)
uv run monitor --interval 5 --log experiment_monitor.json
```

All monitor logs are saved to `src/logs/` directory.

## Reading Results

### Text Output Files
Experiment outputs are automatically saved to `src/experiments/` with filenames that match the MLflow experiment names:
`{EXPERIMENT_NAME}-{TRAIN_SIZE}-{VAL_SIZE}-{TEST_SIZE}-feedback-{with|without}-{TIMESTAMP}.txt`

Example filenames:
- `gepa_comparison-200-10-100-feedback-with-2049.txt`
- `gepa_comparison-200-10-100-feedback-without-2049.txt`

Each file contains the complete experiment log including baseline performance, GEPA optimization progress, and final evaluation results.

### MLflow Dashboard
View comprehensive results in the MLflow UI at http://localhost:5000:
- **Experiments Tab**: Compare optimization runs side-by-side
- **Run Details**: View individual experiment metrics, parameters, and artifacts
- **Charts**: Visualize optimization progress over iterations
- **Artifacts**: Access saved model checkpoints and optimization logs

The MLflow experiments use the same naming convention as the text files (without `.txt` extension).


# Findings

## Summary

The experiments reveal a dramatic difference in GEPA optimization effectiveness based on feedback approach:

- **Without feedback**: Baseline 56.58% / 56.36% (avg 56.47%) | Final 59.7% (+3.23 points improvement)
- **With feedback**: Baseline 57.1% / 56.4% (avg 56.8%) | Final 77.3% (+20.5 points improvement)

The feedback-driven optimization produced a **6x larger improvement** by generating stricter, more deterministic prompts.

## Key Differences in Optimized Prompts

### Extraction Approach
**Without feedback**: Encourages analysis of the full document with faithful terminology but allows synthesis and paraphrasing.

**With feedback**: Mandates exact extraction with verbatim reproduction. Explicitly forbids inference or paraphrasing beyond what's directly stated.

### Handling Edge Cases
**Without feedback**: General instruction to avoid guessing when information isn't available.

**With feedback**: Requires exact canonical response: "The answer is not found in the document." This standardizes behavior for unanswerable questions.

### Output Structure
**Without feedback**: Detailed reasoning followed by answer, with emphasis on domain-specific accuracy.

**With feedback**: Strict two-part format (Reasoning + Answer) with precise constraints on formatting, capitalization, and numeric representation.

## Why Feedback Made Such a Difference

The feedback-enabled optimization aligned the prompt with evaluation metrics through:

1. **Reduced variability**: Exact extraction requirements minimize inconsistent outputs
2. **Better edge case handling**: Canonical "no answer" responses improve scoring consistency  
3. **Metric alignment**: Verbatim reproduction matches similarity-based evaluation better than paraphrased answers
4. **Format standardization**: Clear output structure improves automatic scoring reliability

## Practical Implications

This experiment demonstrates that **feedback quality significantly impacts optimization outcomes**. The detailed feedback text guided GEPA toward solutions that directly address evaluation criteria, while basic similarity scores alone provided insufficient optimization signal.

For production systems, this suggests that investing in high-quality feedback mechanisms can yield substantially better prompt optimization results than relying solely on numerical metrics.

## Experimental Limitations

Due to hardware and time constraints, this experiment used a reduced dataset configuration:
- **Training**: 20 examples (recommended: 200+)
- **Validation**: 5 examples (recommended: 50+) 
- **Test**: 10 examples (recommended: 100+)
- **Threads**: 12 (limited by available CPU cores)

These smaller datasets may not fully represent the optimization potential and could lead to overfitting or unstable results. The limited sample sizes particularly affect the reliability of the baseline measurements and final evaluation scores.

## Future Improvements

### Dataset-Specific Feedback Optimization

Our current feedback system uses generic similarity-based guidance, but we could achieve better optimization results by tailoring feedback to RepLiQA's specific characteristics:

**Answer Type Detection**: Classify answers as boolean (yes/no), numeric, short factual, or descriptive to provide targeted feedback. For example, numerical answers in RepLiQA often require exact precision and units, while factual answers reward verbatim extraction over paraphrasing.

**Common Error Patterns**: Detect and address frequent mistakes like over-elaboration, speculation language ("I think", "probably"), or capitalization mismatches. RepLiQA specifically penalizes inferential answers and rewards exact extraction.

**Document Structure Guidance**: Provide specific tips based on answer types - directing the model to look for tables/lists for numerical data, proper nouns for factual answers, and explicit statements for boolean questions.

**Implementation**: Create answer type classifiers and error pattern detectors that generate dataset-specific feedback messages. This targeted approach could potentially improve optimization gains from +20 points to +30-40 points by providing more actionable guidance.

### System Performance Optimization

Based on system monitoring data from our experiment runs:

**System Performance Analysis:**
- **CPU Utilization**: Average 15-20%, peaks up to 54% (underutilized with 8 cores)
- **Memory Usage**: Steady 64-69% (5.0-5.2GB used of 7.5GB total)
- **Load Average**: 0.4-4.2 (indicating CPU has capacity for more threads)

**API Performance Analysis:**
- **Response Times**: 300-6000ms with high variability (indicating API rate limiting)
- **Rate Limits**: Abundant capacity (29,999/30,000 requests remaining)
- **Bottleneck**: Network latency and API processing time, not local resources

**Optimization Recommendations:**
- **Increase THREADS**: From 12 to 16-20 (system can handle higher concurrency)
- **Larger datasets**: System memory can support TRAIN_SIZE=500+, TEST_SIZE=200+
- **Batch processing**: Group API calls to reduce individual request overhead

**Recommended rerun configuration** for more accurate results:
- TRAIN_SIZE=200, VAL_SIZE=50, TEST_SIZE=100 
- Adjust THREADS based on system monitor recommendations
- Run multiple iterations to establish statistical significance
- Use longer optimization cycles (400+ iterations) for more stable convergence