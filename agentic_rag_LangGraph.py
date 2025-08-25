from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, START, END
import random
import time


class AgenticRAGState(TypedDict):
    user_query: str
    search_results: List[str]
    response: str
    confidence_score: int
    retry_count: int
    max_retries: int
    final_answer: str
    search_strategy: str
    # NEW: Track best response across attempts
    best_response: str
    best_confidence: int
    best_strategy: str


def initial_retrieval(state: AgenticRAGState) -> AgenticRAGState:
    """Initial document retrieval based on user query"""
    query = state["user_query"]
    print(f"🔍 Initial Retrieval for: '{query}'")

    # Simulate document retrieval
    mock_documents = [
        f"Document about {query} - basic information",
        f"Advanced concepts related to {query}",
        f"Technical details of {query} implementation",
    ]

    print(f"📚 Retrieved {len(mock_documents)} documents")

    return {
        "user_query": query,
        "search_results": mock_documents,
        "response": "",
        "confidence_score": 0,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 3),
        "final_answer": "",
        "search_strategy": "basic",
        # Initialize best tracking
        "best_response": "",
        "best_confidence": 0,
        "best_strategy": "",
    }


def generate_response(state: AgenticRAGState) -> AgenticRAGState:
    """Generate response using retrieved documents"""
    query = state["user_query"]
    documents = state["search_results"]
    retry_count = state["retry_count"]

    print(f"🤖 Generating response (attempt {retry_count + 1})...")

    # Simulate LLM response generation with varying quality
    response_templates = [
        f"Based on the documents, {query} is a complex topic that involves multiple aspects.",
        f"According to the retrieved information, {query} can be understood through several key principles.",
        f"The comprehensive analysis of {query} reveals important insights and applications.",
        f"Drawing from authoritative sources, {query} demonstrates significant relevance in modern applications.",
    ]

    response = random.choice(response_templates)

    # Simulate confidence scoring (decreases with retries to show improvement)
    base_confidence = random.randint(50, 95)
    # confidence_penalty = retry_count * 5  # Each retry reduces confidence slightly
    # confidence_score = max(50, base_confidence - confidence_penalty)
    confidence_score = base_confidence

    print(f"📝 Generated response: {response[:50]}...")
    print(f"🎯 Confidence Score: {confidence_score}%")

    # Track best response
    current_best_confidence = state.get("best_confidence", 0)
    current_best_response = state.get("best_response", "")
    current_best_strategy = state.get("best_strategy", "")

    if confidence_score > current_best_confidence:
        print(
            f"🏆 New best response! ({confidence_score}% > {current_best_confidence}%)"
        )
        best_response = response
        best_confidence = confidence_score
        best_strategy = state["search_strategy"]
    else:
        print(
            f"📉 Current response not better than best ({confidence_score}% <= {current_best_confidence}%)"
        )
        best_response = current_best_response
        best_confidence = current_best_confidence
        best_strategy = current_best_strategy

    return {
        "user_query": query,
        "search_results": documents,
        "response": response,
        "confidence_score": confidence_score,
        "retry_count": retry_count + 1,
        "max_retries": state["max_retries"],
        "final_answer": state.get("final_answer", ""),
        "search_strategy": state["search_strategy"],
        "best_response": best_response,
        "best_confidence": best_confidence,
        "best_strategy": best_strategy,
    }


def confidence_router(
    state: AgenticRAGState,
) -> Literal["end_response", "improve_retrieval", "max_retries_reached"]:
    """Router that decides next action based on confidence score"""
    confidence = state["confidence_score"]
    retry_count = state["retry_count"]
    max_retries = state["max_retries"]
    best_confidence = state.get("best_confidence", 0)

    print(
        f"🤔 Router Decision - Current: {confidence}%, Best so far: {best_confidence}%, Retries: {retry_count}/{max_retries}"
    )

    if confidence >= 90:
        print("✅ High confidence! Ending with current response.")
        return "end_response"
    elif retry_count >= max_retries:
        print("⚠️ Max retries reached. Using best available response.")
        return "max_retries_reached"
    else:
        print("🔄 Low confidence. Improving retrieval strategy.")
        return "improve_retrieval"


def improve_retrieval_strategy(state: AgenticRAGState) -> AgenticRAGState:
    """Improve retrieval strategy for better results"""
    query = state["user_query"]
    retry_count = state["retry_count"]

    print(f"🚀 Improving retrieval strategy (retry {retry_count})...")

    # Different strategies based on retry count
    strategies = [
        "semantic_search",
        "hybrid_search",
        "multi_vector_search",
        "advanced_rag",
    ]
    current_strategy = strategies[min(retry_count - 1, len(strategies) - 1)]

    print(f"📈 Using strategy: {current_strategy}")

    # Simulate improved document retrieval
    improved_documents = [
        f"High-quality document about {query} using {current_strategy}",
        f"Authoritative source on {query} with detailed analysis",
        f"Expert-level content about {query} with practical examples",
        f"Recent research on {query} with validated findings",
    ]

    print(f"📚 Retrieved {len(improved_documents)} improved documents")

    return {
        "user_query": query,
        "search_results": improved_documents,
        "response": state["response"],
        "confidence_score": state["confidence_score"],
        "retry_count": retry_count,
        "max_retries": state["max_retries"],
        "final_answer": "",
        "search_strategy": current_strategy,
        "best_response": state["best_response"],
        "best_confidence": state["best_confidence"],
        "best_strategy": state["best_strategy"],
    }


def finalize_response(state: AgenticRAGState) -> AgenticRAGState:
    """Finalize the response when confidence is high enough"""
    response = state["response"]
    confidence = state["confidence_score"]

    print(f"🎉 Finalizing high-confidence response (Score: {confidence}%)")

    final_answer = f"[Confidence: {confidence}%] {response}"

    return {
        "user_query": state["user_query"],
        "search_results": state["search_results"],
        "response": response,
        "confidence_score": confidence,
        "retry_count": state["retry_count"],
        "max_retries": state["max_retries"],
        "final_answer": final_answer,
        "search_strategy": state["search_strategy"],
        "best_response": state["best_response"],
        "best_confidence": state["best_confidence"],
        "best_strategy": state["best_strategy"],
    }


def handle_max_retries(state: AgenticRAGState) -> AgenticRAGState:
    """Handle case where max retries are reached - USE BEST RESPONSE"""
    # Use the BEST response, not the last one
    best_response = state["best_response"]
    best_confidence = state["best_confidence"]
    best_strategy = state["best_strategy"]
    retry_count = state["retry_count"]

    print(
        f"⚠️ Max retries reached. Using BEST response with {best_confidence}% confidence"
    )
    print(f"📊 Best response was from strategy: {best_strategy}")

    final_answer = f"[Max retries reached - Using BEST response: {best_confidence}%] {best_response}"

    return {
        "user_query": state["user_query"],
        "search_results": state["search_results"],
        "response": best_response,  # Use best response
        "confidence_score": best_confidence,  # Use best confidence
        "retry_count": retry_count,
        "max_retries": state["max_retries"],
        "final_answer": final_answer,
        "search_strategy": best_strategy,  # Use best strategy
        "best_response": best_response,
        "best_confidence": best_confidence,
        "best_strategy": best_strategy,
    }


def create_agentic_rag_flow():
    """Create the agentic RAG flow with router pattern"""
    workflow = StateGraph(AgenticRAGState)

    # Add nodes
    workflow.add_node("initial_retrieval", initial_retrieval)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("improve_retrieval_strategy", improve_retrieval_strategy)
    workflow.add_node("finalize_response", finalize_response)
    workflow.add_node("handle_max_retries", handle_max_retries)

    # Add edges
    workflow.add_edge(START, "initial_retrieval")
    workflow.add_edge("initial_retrieval", "generate_response")

    # Router logic - the key part!
    workflow.add_conditional_edges(
        "generate_response",
        confidence_router,
        {
            "end_response": "finalize_response",
            "improve_retrieval": "improve_retrieval_strategy",
            "max_retries_reached": "handle_max_retries",
        },
    )

    # Retry loop
    workflow.add_edge("improve_retrieval_strategy", "generate_response")

    # End states
    workflow.add_edge("finalize_response", END)
    workflow.add_edge("handle_max_retries", END)

    return workflow.compile()


def run_agentic_rag_example():
    """Run multiple examples with different queries"""
    app = create_agentic_rag_flow()
    mermaid_code = app.get_graph().draw_mermaid()
    print(mermaid_code)

    mermaid_ascii = app.get_graph().draw_ascii()
    print(mermaid_ascii)

    test_queries = [
        "machine learning algorithms",
        # "climate change impacts",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"🔍 TEST CASE {i}: '{query}'")
        print("=" * 60)

        initial_state = {
            "user_query": query,
            "search_results": [],
            "response": "",
            "confidence_score": 0,
            "retry_count": 0,
            "max_retries": 3,
            "final_answer": "",
            "search_strategy": "",
            "best_response": "",
            "best_confidence": 0,
            "best_strategy": "",
        }

        # Run the flow
        start_time = time.time()
        result = app.invoke(initial_state)
        end_time = time.time()

        print(f"\n🎯 FINAL RESULT:")
        print(f"Query: {result['user_query']}")
        print(f"Final Answer: {result['final_answer']}")
        print(f"Total Retries: {result['retry_count']}")
        print(f"Final Strategy: {result['search_strategy']}")
        print(f"Best Confidence Achieved: {result['best_confidence']}%")
        print(f"Processing Time: {end_time - start_time:.2f}s")

        time.sleep(1)  # Brief pause between tests


if __name__ == "__main__":
    print("🚀 Agentic RAG with Router Pattern - FIXED to use Best Response")

    # Run examples
    run_agentic_rag_example()
