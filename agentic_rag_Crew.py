from typing import List, Literal
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field
import random
import time


class AgenticRAGState(BaseModel):
    user_query: str = ""
    search_results: List[str] = Field(default_factory=list)
    response: str = ""
    confidence_score: int = 0
    retry_count: int = 0
    max_retries: int = 3
    final_answer: str = ""
    search_strategy: str = ""
    # Track best response across attempts
    best_response: str = ""
    best_confidence: int = 0
    best_strategy: str = ""


class AgenticRAGFlow(Flow[AgenticRAGState]):

    def __init__(self, user_query: str, max_retries: int = 3):
        super().__init__()
        self.state.user_query = user_query
        self.state.max_retries = max_retries

    @start()
    def initial_retrieval(self):
        """Initial document retrieval based on user query"""
        query = self.state.user_query
        print(f"🔍 Initial Retrieval for: '{query}'")

        # Simulate document retrieval
        mock_documents = [
            f"Document about {query} - basic information",
            f"Advanced concepts related to {query}",
            f"Technical details of {query} implementation",
        ]

        print(f"📚 Retrieved {len(mock_documents)} documents")

        self.state.search_results = mock_documents
        self.state.search_strategy = "basic"
        self.state.retry_count = 0

    @listen(initial_retrieval)
    def generate_response(self):
        """Generate response using retrieved documents"""
        query = self.state.user_query
        documents = self.state.search_results
        retry_count = self.state.retry_count

        print(f"🤖 Generating response (attempt {retry_count + 1})...")

        # Simulate LLM response generation with varying quality
        response_templates = [
            f"Based on the documents, {query} is a complex topic that involves multiple aspects.",
            f"According to the retrieved information, {query} can be understood through several key principles.",
            f"The comprehensive analysis of {query} reveals important insights and applications.",
            f"Drawing from authoritative sources, {query} demonstrates significant relevance in modern applications.",
        ]

        response = random.choice(response_templates)

        # Simulate confidence scoring
        base_confidence = random.randint(50, 95)
        confidence_score = base_confidence

        print(f"📝 Generated response: {response[:50]}...")
        print(f"🎯 Confidence Score: {confidence_score}%")

        # Track best response
        if confidence_score > self.state.best_confidence:
            print(
                f"🏆 New best response! ({confidence_score}% > {self.state.best_confidence}%)"
            )
            self.state.best_response = response
            self.state.best_confidence = confidence_score
            self.state.best_strategy = self.state.search_strategy
        else:
            print(
                f"📉 Current response not better than best ({confidence_score}% <= {self.state.best_confidence}%)"
            )

        self.state.response = response
        self.state.confidence_score = confidence_score
        self.state.retry_count += 1

    @router(generate_response)
    def confidence_router(
        self,
    ) -> Literal["end_response", "improve_retrieval", "max_retries_reached"]:
        """Router that decides next action based on confidence score"""
        confidence = self.state.confidence_score
        retry_count = self.state.retry_count
        max_retries = self.state.max_retries
        best_confidence = self.state.best_confidence

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

    @listen("improve_retrieval")
    def improve_retrieval_strategy(self):
        """Improve retrieval strategy for better results"""
        query = self.state.user_query
        retry_count = self.state.retry_count

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

        self.state.search_results = improved_documents
        self.state.search_strategy = current_strategy

    @listen(improve_retrieval_strategy)
    def generate_improved_response(self):
        """Generate response after improving retrieval strategy"""
        # This calls the same generation logic but with improved documents
        self.generate_response()

    @router(generate_improved_response)
    def improved_confidence_router(
        self,
    ) -> Literal["end_response", "improve_retrieval", "max_retries_reached"]:
        """Router for improved responses - same logic as main router"""
        return self.confidence_router()

    @listen("end_response")
    def finalize_response(self):
        """Finalize the response when confidence is high enough"""
        response = self.state.response
        confidence = self.state.confidence_score

        print(f"🎉 Finalizing high-confidence response (Score: {confidence}%)")

        final_answer = f"[Confidence: {confidence}%] {response}"
        self.state.final_answer = final_answer

    @listen("max_retries_reached")
    def handle_max_retries(self):
        """Handle case where max retries are reached - USE BEST RESPONSE"""
        # Use the BEST response, not the last one
        best_response = self.state.best_response
        best_confidence = self.state.best_confidence
        best_strategy = self.state.best_strategy
        retry_count = self.state.retry_count

        print(
            f"⚠️ Max retries reached. Using BEST response with {best_confidence}% confidence"
        )
        print(f"📊 Best response was from strategy: {best_strategy}")

        final_answer = f"[Max retries reached - Using BEST response: {best_confidence}%] {best_response}"

        # Update state with best values
        self.state.response = best_response
        self.state.confidence_score = best_confidence
        self.state.search_strategy = best_strategy
        self.state.final_answer = final_answer


def run_agentic_rag_example():
    """Run multiple examples with different queries"""
    test_queries = [
        "machine learning algorithms",
        "climate change impacts",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"🔍 TEST CASE {i}: '{query}'")
        print("=" * 60)

        # Create and run the flow
        start_time = time.time()
        flow = AgenticRAGFlow(user_query=query, max_retries=3)
        result = flow.kickoff()
        end_time = time.time()

        print(f"\n🎯 FINAL RESULT:")
        print(f"Query: {flow.state.user_query}")
        print(f"Final Answer: {flow.state.final_answer}")
        print(f"Total Retries: {flow.state.retry_count}")
        print(f"Final Strategy: {flow.state.search_strategy}")
        print(f"Best Confidence Achieved: {flow.state.best_confidence}%")
        print(f"Processing Time: {end_time - start_time:.2f}s")

        time.sleep(1)  # Brief pause between tests


if __name__ == "__main__":
    print("🚀 Agentic RAG with CrewAI Flow - Router Pattern Implementation")

    # Run examples
    run_agentic_rag_example()
