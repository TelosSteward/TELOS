"""
TELOS Corpus Loader

Provides RAG (Retrieval-Augmented Generation) capabilities for Demo Mode.
Loads TELOS documentation and retrieves relevant chunks based on user queries.
"""

import numpy as np
from typing import List, Tuple


class TELOSCorpusLoader:
    """
    Loads and manages TELOS framework documentation for RAG.

    Embeds documentation chunks and retrieves relevant context
    based on semantic similarity to user queries.
    """

    def __init__(self, embedding_provider):
        """
        Initialize corpus loader with embedding provider.

        Args:
            embedding_provider: Provider for generating embeddings
        """
        self.embedding_provider = embedding_provider
        self.chunks = []
        self.chunk_embeddings = []

    def load_corpus(self) -> int:
        """
        Load TELOS documentation corpus.

        Returns:
            int: Number of chunks loaded
        """
        self.chunks = get_telos_documentation_chunks()

        # Generate embeddings for all chunks
        self.chunk_embeddings = [
            self.embedding_provider.embed(chunk)
            for chunk in self.chunks
        ]

        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve top-k most relevant chunks for query.

        Args:
            query: User's question/query
            top_k: Number of chunks to retrieve

        Returns:
            List of most relevant documentation chunks
        """
        if not self.chunks:
            return []

        # Embed the query
        query_embedding = self.embedding_provider.embed(query)

        # Calculate cosine similarities
        similarities = []
        for chunk_embedding in self.chunk_embeddings:
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)
            similarities.append(similarity)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return top chunks
        return [self.chunks[i] for i in top_indices]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_telos_documentation_chunks() -> List[str]:
    """
    Get TELOS framework documentation chunks for RAG.

    Returns:
        List of documentation text chunks
    """
    return [
        # Fidelity - Core Concept
        """
        **Fidelity** is the core measurement in TELOS. It quantifies how well an AI's responses
        align with the user's intended purpose. Think of it like a GPS showing how close you
        are to your destination.

        Fidelity is measured on a scale from 0.0 to 1.0:
        - 1.0 = Perfect alignment with purpose
        - 0.8-0.9 = Good alignment (typical for well-governed sessions)
        - 0.6-0.7 = Moderate drift (governance may intervene)
        - Below 0.6 = Significant misalignment (intervention likely)

        Fidelity is calculated using semantic embeddings that compare the AI's response
        to the user's established purpose. The mathematics preserve privacy - only the
        alignment score is stored, not the actual conversation content.
        """,

        # Primacy Attractor - Definition
        """
        **Primacy Attractor (PA)** is the configuration that guides AI behavior in TELOS.
        Think of it as the "North Star" that keeps the AI on course.

        A PA has three components:
        1. **Purpose** - What the user wants to accomplish
        2. **Scope** - The boundaries of relevant topics
        3. **Boundaries** - What the AI should NOT do

        In Demo Mode, the PA is pre-configured to teach about TELOS.
        In Open Mode, TELOS extracts the PA from your conversation dynamically.

        The PA is "primacy" because it takes precedence over other considerations.
        It's an "attractor" because it pulls responses toward alignment.
        """,

        # Purpose Extraction
        """
        **Purpose Extraction** is how TELOS identifies what the user cares about.

        During the first few turns (calibration phase), TELOS analyzes:
        - What topics the user asks about
        - What problems they're trying to solve
        - What outcomes they seem to want

        This creates a semantic representation of the user's purpose, which becomes
        the foundation of the Primacy Attractor.

        Example: If you consistently ask about data privacy, TELOS extracts
        "Understand and implement data privacy" as part of your purpose.

        Purpose extraction is continuous - the PA can adapt if your needs change,
        but only within the bounds of your original intent.
        """,

        # Drift and Drift Prevention
        """
        **Drift** occurs when AI responses move away from the user's purpose.

        Without governance, conversations naturally drift:
        - User asks about X
        - AI responds about X, but mentions Y
        - User follows up on Y
        - 10 turns later, you're discussing Z (completely unrelated to X)

        TELOS prevents drift by:
        1. **Monitoring fidelity** on every turn
        2. **Detecting when alignment drops** below acceptable thresholds
        3. **Intervening** to redirect the conversation back to purpose

        Drift prevention is NOT censorship - it's about staying on the user's chosen path.
        If the user genuinely wants to change direction, TELOS adapts the PA accordingly.
        """,

        # Interventions
        """
        **Interventions** happen when TELOS detects misalignment and corrects course.

        How interventions work:
        1. AI generates a candidate response
        2. TELOS calculates what fidelity WOULD be if that response was sent
        3. If fidelity would drop too low, TELOS modifies the response
        4. The intervention is logged transparently (you can see it happened)

        Interventions are gentle - not heavy-handed censorship. They typically:
        - Add a clarifying statement
        - Redirect focus back to purpose
        - Acknowledge the tangent but bring conversation back
        - Suggest a more aligned alternative

        Example: "That's an interesting point about [Y], but let's bring this back to your
        original question about [X] since that's what we're focused on."

        All interventions are visible in the metrics - TELOS never hides its governance.
        """,

        # Privacy and Mathematical Abstraction
        """
        **Privacy in TELOS** is preserved through mathematical abstraction.

        What TELOS STORES:
        - Fidelity scores (numbers like 0.87)
        - Purpose embeddings (mathematical vectors)
        - Alignment metrics and timestamps
        - Intervention records (when, not why in detail)

        What TELOS DOES NOT STORE:
        - Your actual messages (the text you type)
        - The AI's actual responses (the text it generates)
        - Conversation content or context
        - Personal information or identifiable data

        This is called "mathematical privacy" - governance without surveillance.
        The system knows you're aligned, but doesn't remember what you said.

        Think of it like a fitness tracker: it knows you walked 10,000 steps,
        but doesn't record video of where you walked.
        """,

        # Governance vs. Censorship
        """
        **TELOS Governance ≠ Censorship**

        This is a crucial distinction:

        **Censorship** imposes external values and restrictions:
        - Someone else decides what you can't discuss
        - Topics are forbidden regardless of your intent
        - Restrictions serve the censor's goals, not yours

        **Governance** enforces YOUR own stated purpose:
        - You decide what matters through your conversation
        - TELOS extracts and maintains YOUR values
        - Restrictions serve YOUR goals, not an external authority

        Example:
        - Censorship: "You can't discuss politics" (imposed externally)
        - Governance: "You asked about data privacy, so I'm keeping us focused on that" (your choice)

        TELOS is a commitment device - like a gym buddy who keeps you on track
        with YOUR fitness goals, not their preferences.
        """,

        # Fidelity Calculation
        """
        **How Fidelity is Calculated**

        Fidelity measurement uses semantic embeddings:

        1. **Embed the Purpose**: Convert the user's purpose into a vector (array of numbers)
        2. **Embed the Response**: Convert the AI's response into a vector
        3. **Calculate Similarity**: Measure cosine similarity between the two vectors
        4. **Normalize**: Scale to 0.0-1.0 range

        The formula:
        fidelity = cosine_similarity(purpose_embedding, response_embedding)

        Why embeddings?
        - They capture semantic meaning, not just keywords
        - Similar concepts have similar vectors
        - Mathematics preserves privacy (no text storage needed)

        Example: "data security" and "information protection" have high similarity
        in embedding space, even though the words are different.
        """,

        # Convergence and PA Establishment
        """
        **PA Convergence** is when your Primacy Attractor becomes established.

        During calibration (typically turns 1-10):
        - TELOS observes your conversation patterns
        - Purpose becomes clearer with each turn
        - Scope and boundaries solidify
        - Fidelity measurements stabilize

        At convergence (around turn 7-10):
        - PA is considered "established"
        - Governance becomes active and consistent
        - Fidelity tracking is reliable
        - Interventions (if needed) are more accurate

        You'll see a message: "🎯 PA Established!" when this happens.

        After convergence, the PA can still adapt, but changes are gradual
        and require consistent evidence that your purpose has evolved.
        """,

        # Real-World Applications
        """
        **Real-World TELOS Applications**

        Where TELOS governance matters:

        1. **Customer Service AI**: Keep conversations focused on solving the customer's issue
        2. **Educational Tutors**: Maintain alignment with learning objectives
        3. **Medical AI Assistants**: Stay within clinical scope, avoid off-topic tangents
        4. **Research Assistants**: Keep literature review focused on research question
        5. **Content Moderation**: Enforce community guidelines while respecting user intent

        In each case, TELOS ensures the AI serves the USER'S purpose, not just
        generates interesting (but potentially misaligned) content.

        The key benefit: Users get AI assistance that stays on track without
        constant manual corrections.
        """,

        # Counterfactual Analysis
        """
        **Counterfactual Analysis** shows what WOULD have happened without governance.

        TELOS can generate two versions of a response:
        1. **Governed Response**: With TELOS active (aligned, on-purpose)
        2. **Ungoverned Response**: Without TELOS (potentially drifted)

        This comparison demonstrates the value of governance by showing:
        - How much the ungoverned version drifted
        - What topics it introduced unnecessarily
        - How fidelity would have dropped
        - Why the intervention was valuable

        Counterfactuals are educational - they make governance visible and
        demonstrate its impact in concrete terms.

        In Demo Mode, you may see counterfactual examples to illustrate
        how TELOS prevents drift.
        """,

        # Turn-by-Turn Mechanics
        """
        **What Happens Each Turn in TELOS**

        Every interaction follows this sequence:

        1. **User Input**: You ask a question or make a statement
        2. **Purpose Check**: TELOS compares input to established PA
        3. **Context Retrieval**: Relevant information is gathered (in Demo Mode: from TELOS docs)
        4. **Response Generation**: AI creates a candidate response
        5. **Fidelity Prediction**: TELOS calculates what fidelity WOULD be
        6. **Governance Decision**: If fidelity is acceptable, response proceeds; if not, intervention
        7. **Response Delivery**: You see the final (possibly modified) response
        8. **Metrics Update**: Fidelity score, intervention count, drift status updated

        This happens in milliseconds, but you can see the results:
        - Updated fidelity score
        - Intervention counter (if applicable)
        - PA status (calibrating → established)

        The entire process is transparent - nothing hidden, everything measurable.
        """
    ]


def format_context_for_llm(chunks: List[str]) -> str:
    """
    Format retrieved chunks for LLM context.

    Args:
        chunks: List of documentation chunks

    Returns:
        Formatted context string for LLM prompt
    """
    if not chunks:
        return ""

    formatted = "# TELOS Framework Documentation Context\n\n"
    formatted += "The following information from TELOS documentation is relevant to the user's query:\n\n"

    for i, chunk in enumerate(chunks, 1):
        formatted += f"## Context Chunk {i}\n\n{chunk.strip()}\n\n"

    formatted += "---\n\n"
    formatted += "Use this documentation to provide accurate, grounded responses about TELOS.\n"

    return formatted
