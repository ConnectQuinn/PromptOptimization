from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ValidationError
import dspy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
import os
from datetime import datetime
import mlflow

load_dotenv()


class Config(BaseSettings):
    openai_api_key: str
    experiment_name: str
    train_size: int
    val_size: int
    test_size: int
    threads: int
    feedback: str
    
    class Config:
        env_file = '.env'

class GenerateResponse(dspy.Signature):
    """Based on the provided document, answer the question."""
    document_extracted = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

class TeeOutput:
    def __init__(self, file, original_stream):
        self.file = file
        self.original_stream = original_stream

    def write(self, data):
        self.file.write(data)
        self.original_stream.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
        self.original_stream.flush()

def init_dataset(config: Config):
    cache_dir = './src/dataset_cache'
    repliqa = load_dataset("ServiceNow/repliqa", cache_dir=cache_dir)
    
    # Use whatever split is available (likely 'validation' or the first available split)
    available_splits = list(repliqa.keys())
    print(f"Available splits: {available_splits}")
    
    # Get the first available split
    dataset = repliqa[available_splits[0]]
    
    dataset = [
        dspy.Example({
            "document_extracted": x['document_extracted'],
            "question": x['question'],
            "answer": x['answer'],
        }).with_inputs("document_extracted", "question")
        for x in dataset
    ]
    import random
    random.Random(0).shuffle(dataset)

    # Use configurable sizes from environment
    train_set = dataset[:config.train_size]
    val_set = dataset[config.train_size:config.train_size + config.val_size]
    test_set = dataset[config.train_size + config.val_size:config.train_size + config.val_size + config.test_size]

    return train_set, val_set, test_set

def create_metric(semantic_model):
    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        ground_truth = example['answer']
        predicted_answer = prediction.answer
        
        # Compute embeddings
        gt_embedding = semantic_model.encode([ground_truth])
        pred_embedding = semantic_model.encode([predicted_answer])
        
        # Calculate cosine similarity
        similarity = np.dot(gt_embedding, pred_embedding.T)[0][0]
        
        # Return similarity score (threshold can be adjusted)
        return float(similarity)
    return metric

def create_metric_with_feedback(semantic_model):
    def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
        ground_truth = example['answer']
        predicted_answer = prediction.answer
        
        # Compute embeddings
        gt_embedding = semantic_model.encode([ground_truth])
        pred_embedding = semantic_model.encode([predicted_answer])
        
        # Calculate cosine similarity
        similarity = np.dot(gt_embedding, pred_embedding.T)[0][0]
        score = float(similarity)
        
        # Generate feedback based on similarity score
        feedback_text = ""
        if score >= 0.9:
            feedback_text = f"Excellent match! Your answer '{predicted_answer}' is semantically very close to the expected answer '{ground_truth}'. Similarity score: {score:.3f}"
        elif score >= 0.7:
            feedback_text = f"Good answer! Your response '{predicted_answer}' captures the main meaning of the expected answer '{ground_truth}', but could be more precise. Similarity score: {score:.3f}"
        elif score >= 0.5:
            feedback_text = f"Partial match. Your answer '{predicted_answer}' has some semantic overlap with '{ground_truth}', but misses key information or has significant differences. Similarity score: {score:.3f}"
        else:
            feedback_text = f"Poor match. Your answer '{predicted_answer}' is semantically distant from the expected answer '{ground_truth}'. Consider focusing on the key concepts and information in the document. Similarity score: {score:.3f}"
        
        # Add guidance for improvement
        if score < 0.7:
            feedback_text += f"\n\nTo improve: Focus on extracting the most relevant information from the document that directly answers the question. The expected answer is: '{ground_truth}'"
        
        return dspy.Prediction(score=score, feedback=feedback_text)
    
    return metric_with_feedback


def main():

    try:
        config = Config()
        print("✓ All required environment variables found")
    except ValidationError as e:
        print("✗ Missing required environment variables:")
        for error in e.errors():
            print(f"  - {error['loc'][0].upper()}: {error['msg']}")
        return
    
    # Create experiments directory and setup output capture
    experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%H%M")
    experiment_name = f"{config.experiment_name}-{config.train_size}-{config.val_size}-{config.test_size}-feedback-{'with' if config.feedback.lower() == 'true' else 'without'}-{timestamp}"
    filename = f"{experiment_name}.txt"
    filepath = os.path.join(experiments_dir, filename)
    
    with open(filepath, 'w') as f:
        # Redirect both stdout and stderr to capture all output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeOutput(f, original_stdout)
        sys.stderr = TeeOutput(f, original_stderr)
        
        try:
            print(f"Starting experiment: {config.experiment_name}")
            print(f"Output will be saved to: {filepath}")
            
            # Set up MLflow tracking
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(experiment_name)
            
            # Enable MLflow autologging (disable traces to reduce warnings)
            mlflow.dspy.autolog(
                log_compiles=True,   # Track GEPA optimization progress
                log_evals=True,      # Track evaluation results  
                log_traces=False     # Disable traces to avoid warnings
            )
            print(f"MLflow tracking enabled for experiment: {experiment_name}")
            
            lm = dspy.LM("openai/gpt-4.1-mini", temperature=1, api_key=config.openai_api_key, max_tokens=32000)
            dspy.configure(lm=lm)
            
            print(f"Loading all-MiniLM-L6-v2 semantic model")
            # Initialize semantic similarity model
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Select metric based on FEEDBACK setting
            if config.feedback.lower() == "true":
                metric = create_metric_with_feedback(semantic_model)
                print("Using metric WITH feedback for optimization")
            else:
                metric = create_metric(semantic_model)
                print("Using metric WITHOUT feedback for optimization")
            # Load dataset
            train_set, val_set, test_set = init_dataset(config)
            print(f"Loaded {len(train_set)} train, {len(val_set)} val, {len(test_set)} test examples")
            
            # Create and train model
            program = dspy.ChainOfThought(GenerateResponse)
            
            # Example prediction
            example = train_set[0]
            print(f"\nExample input: {example.question}\n")
            prediction = program(document_extracted=example.document_extracted, question=example.question)
            print(f"Prediction: {prediction.answer}\n")
            print(f"Ground truth: {example.answer}\n")

            evaluate = dspy.Evaluate(
                devset=test_set,
                metric=metric,
                num_threads=config.threads,
                display_table=True,
                display_progress=True
            )
            print("\n=== BASELINE EVALUATION (original program) ===")
            evaluate(program)

            optimizer = dspy.GEPA(
                metric=metric,
                auto="light",
                num_threads=config.threads,
                track_stats=True,
                reflection_minibatch_size=3,
                reflection_lm=lm,
            )
            print("\n=== RUNNING GEPA OPTIMIZATION ===")
            optimized_program = optimizer.compile(
                program,
                trainset=train_set,
                valset=val_set,
            )
            print("\n=== EVALUATION AFTER GEPA (optimized program) ===")
            evaluate(optimized_program)
            
            print("\n" + "="*80)
            print("PROMPT COMPARISON")
            print("="*80)
            print("\nORIGINAL PROMPT:")
            print("-" * 40)
            original_instructions = program.predict.signature.instructions or "Based on the provided document, answer the question."
            print(original_instructions)
            
            print("\nOPTIMIZED PROMPT:")
            print("-" * 40)
            optimized_instructions = optimized_program.predict.signature.instructions or "No optimized instructions found"
            print(optimized_instructions)
            
            print("\n" + "="*80)
            
        finally:
            # Restore original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        print(f"\n✓ Experiment completed! Output saved to: {filepath}")

if __name__ == "__main__":
    main()
