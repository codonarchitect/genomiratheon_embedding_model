from sentence_transformers import SentenceTransformer, util

# =============================
# Load Trained Model Function
# =============================
def load_trained_model(model_path='genomiratheon_embedding_model'):
    print(f"\n Loading trained model from: {model_path}")
    model = SentenceTransformer(model_path)
    return model

# =============================
# Compare Multiple Sentences to Reference
# =============================
def compare_multiple(model, queries, reference):
    # Lowercase and encode all at once
    queries = [q.strip().lower() for q in queries]
    reference = reference.strip().lower()

    query_embeddings = model.encode(queries, convert_to_tensor=True)
    reference_embedding = model.encode(reference, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(query_embeddings, reference_embedding)

    # Print results
    print("\n Cosine Similarity Scores:")
    for i, query in enumerate(queries):
        print(f"[{i+1}] '{query}' → Score: {similarities[i].item():.4f}")

# Load model
model = load_trained_model('genomiratheon_embedding_model')

# Define multiple input queries
queries = [
    "Who controls synthetic biology licensing?",
    "What is the role of genomiratheon™ in genetic IP?",
    "Explain codon treaty regulations in licensing."
]

# Define the reference sentence
reference = "genomiratheon™ licenses synthetic biology using codon treaties."

# Compare all queries to reference
compare_multiple(model, queries, reference)
