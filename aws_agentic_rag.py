from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, START, END
import boto3
import json
import time
from datetime import datetime


class AgenticRAGState(TypedDict):
    user_query: str
    original_query: str
    search_results: List[Dict[str, Any]]
    response: str
    confidence_score: float
    retry_count: int
    max_retries: int
    final_answer: str
    best_response: str
    best_confidence: float
    best_query: str
    kb_id: str


class AWSBedrockRAGAgent:
    def __init__(self, kb_id: str, region: str = "us-west-2"):
        """Initialize AWS Bedrock RAG Agent"""
        self.kb_id = kb_id
        self.region = region

        # Initialize AWS clients
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime", region_name=region
        )

        # Model configuration
        self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def retrieve_from_kb(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve documents from AWS Knowledge Base"""
        try:
            print(f"🔍 Querying Knowledge Base: '{query}'")

            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": max_results}
                },
            )

            results = []
            for item in response.get("retrievalResults", []):
                results.append(
                    {
                        "content": item.get("content", {}).get("text", ""),
                        "score": item.get("score", 0),
                        "location": item.get("location", {}),
                        "metadata": item.get("metadata", {}),
                    }
                )

            print(f"📚 Retrieved {len(results)} documents from Knowledge Base")
            return results

        except Exception as e:
            print(f"❌ Error retrieving from Knowledge Base: {str(e)}")
            return []

    def generate_response_with_claude(
        self, query: str, context_docs: List[Dict[str, Any]], retry_count: int = 0
    ) -> tuple[str, float]:
        """Generate response using Claude Sonnet 3.5 v2"""
        try:
            # Prepare context from retrieved documents
            context = ""
            for i, doc in enumerate(context_docs[:5], 1):  # Use top 5 docs
                score = doc.get("score", 0)
                content = doc.get("content", "")[:1000]  # Limit content length
                context += f"Document {i} (Relevance: {score:.3f}):\n{content}\n\n"

            # Create prompt for Claude
            prompt = f"""You are an expert AI assistant. Based on the provided context documents, answer the user's question comprehensively and accurately.

Context Documents:
{context}

User Question: {query}

Instructions:
1. Provide a detailed, accurate answer based solely on the information in the context documents
2. If the context doesn't fully answer the question, acknowledge the limitations
3. Cite relevant information from the documents when possible
4. Be concise but thorough
5. Rate your confidence in this answer on a scale of 0-100

Please provide your response in the following format:
ANSWER: [Your detailed answer here]
CONFIDENCE: [Your confidence score 0-100]
REASONING: [Brief explanation of your confidence level]"""

            # Call Claude via Bedrock
            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "top_p": 0.9,
                }
            )

            print(f"🤖 Generating response with Claude (attempt {retry_count + 1})...")

            response = self.bedrock_runtime.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())
            claude_response = response_body.get("content")[0].get("text")

            # Parse response to extract answer and confidence
            answer, confidence = self._parse_claude_response(claude_response)

            print(f"📝 Generated response: {answer[:100]}...")
            print(f"🎯 Confidence Score: {confidence}%")

            return answer, confidence

        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}", 0.0

    def _parse_claude_response(self, response: str) -> tuple[str, float]:
        """Parse Claude's response to extract answer and confidence"""
        try:
            lines = response.strip().split("\n")
            answer = ""
            confidence = 50.0  # Default confidence

            for line in lines:
                if line.startswith("ANSWER:"):
                    answer = line.replace("ANSWER:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    conf_text = line.replace("CONFIDENCE:", "").strip()
                    # Extract numeric value
                    import re

                    conf_match = re.search(r"(\d+(?:\.\d+)?)", conf_text)
                    if conf_match:
                        confidence = float(conf_match.group(1))

            # If no structured format, use entire response as answer
            if not answer:
                answer = response
                # Estimate confidence based on response quality indicators
                if len(response) > 200 and "based on" in response.lower():
                    confidence = 75.0
                elif len(response) > 100:
                    confidence = 60.0
                else:
                    confidence = 40.0

            return answer, confidence

        except Exception as e:
            print(f"⚠️ Error parsing response: {e}")
            return response, 50.0

    def optimize_query(self, original_query: str, retry_count: int) -> str:
        """Use Claude to optimize the query for better retrieval"""
        try:
            optimization_prompt = f"""You are a query optimization expert. Your task is to improve a search query for better document retrieval from a knowledge base.

Original Query: "{original_query}"
Retry Attempt: {retry_count}

Based on the retry count, apply different optimization strategies:
- Retry 1: Expand with more specific terms and synonyms
- Retry 2: Add contextual terms and use more technical language
- Retry 3: Try alternative phrasing or different angle
- Retry 4+: Use broader terms or related concepts

Provide only the optimized query, nothing else. Make it concise but more effective for retrieval."""

            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": optimization_prompt}],
                    "temperature": 0.3,
                    "top_p": 0.8,
                }
            )

            response = self.bedrock_runtime.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())
            optimized_query = response_body.get("content")[0].get("text").strip()

            # Clean up the response (remove quotes, extra text)
            optimized_query = optimized_query.replace('"', "").replace("'", "").strip()
            if optimized_query.startswith("Optimized query:"):
                optimized_query = optimized_query.replace(
                    "Optimized query:", ""
                ).strip()

            return optimized_query

        except Exception as e:
            print(f"❌ Error optimizing query: {str(e)}")
            # Fallback to simple optimization
            return f"{original_query} detailed information examples"


def initial_retrieval(state: AgenticRAGState) -> AgenticRAGState:
    """Initial document retrieval from AWS Knowledge Base"""
    agent = AWSBedrockRAGAgent(state["kb_id"])
    query = state["user_query"]
    original_query = state.get("original_query", query)

    print(f"🔍 Retrieval for: '{query}'")
    if query != original_query:
        print(f"   (Optimized from: '{original_query}')")

    # Retrieve from Knowledge Base
    search_results = agent.retrieve_from_kb(query)

    return {
        **state,
        "user_query": query,
        "original_query": original_query,
        "search_results": search_results,
        "response": "",
        "confidence_score": 0.0,
    }


def generate_response(state: AgenticRAGState) -> AgenticRAGState:
    """Generate response using Claude with retrieved documents"""
    agent = AWSBedrockRAGAgent(state["kb_id"])
    query = state["user_query"]
    documents = state["search_results"]
    retry_count = state["retry_count"]

    # Generate response
    response, confidence_score = agent.generate_response_with_claude(
        query, documents, retry_count
    )

    # Track best response
    current_best_confidence = state.get("best_confidence", 0.0)

    if confidence_score > current_best_confidence:
        print(
            f"🏆 New best response! ({confidence_score}% > {current_best_confidence}%)"
        )
        best_response = response
        best_confidence = confidence_score
        best_query = query
    else:
        print(
            f"📉 Current response not better than best ({confidence_score}% <= {current_best_confidence}%)"
        )
        best_response = state.get("best_response", response)
        best_confidence = current_best_confidence
        best_query = state.get("best_query", query)

    return {
        **state,
        "response": response,
        "confidence_score": confidence_score,
        "retry_count": retry_count + 1,
        "best_response": best_response,
        "best_confidence": best_confidence,
        "best_query": best_query,
    }


def confidence_router(
    state: AgenticRAGState,
) -> Literal["end_response", "optimize_query", "max_retries_reached"]:
    """Router that decides next action based on confidence score"""
    confidence = state["confidence_score"]
    retry_count = state["retry_count"]
    max_retries = state["max_retries"]
    best_confidence = state.get("best_confidence", 0.0)

    print(
        f"🤔 Router Decision - Current: {confidence}%, Best so far: {best_confidence}%, Retries: {retry_count}/{max_retries}"
    )

    if confidence >= 85:  # Slightly lower threshold for real-world usage
        print("✅ High confidence! Ending with current response.")
        return "end_response"
    elif retry_count >= max_retries:
        print("⚠️ Max retries reached. Using best available response.")
        return "max_retries_reached"
    else:
        print("🔄 Low confidence. Optimizing query for better retrieval.")
        return "optimize_query"


def optimize_query_node(state: AgenticRAGState) -> AgenticRAGState:
    """Optimize query using Claude for better retrieval results"""
    agent = AWSBedrockRAGAgent(state["kb_id"])
    original_query = state["original_query"]
    current_query = state["user_query"]
    retry_count = state["retry_count"]

    print(f"🚀 Optimizing query (retry {retry_count})...")
    print(f"   Current query: '{current_query}'")

    # Use Claude to optimize the query
    optimized_query = agent.optimize_query(original_query, retry_count)

    print(f"📈 AI-optimized query: '{optimized_query}'")

    return {
        **state,
        "user_query": optimized_query,
    }


def finalize_response(state: AgenticRAGState) -> AgenticRAGState:
    """Finalize the response when confidence is high enough"""
    response = state["response"]
    confidence = state["confidence_score"]

    print(f"🎉 Finalizing high-confidence response (Score: {confidence}%)")

    final_answer = f"[Confidence: {confidence}%] {response}"

    return {
        **state,
        "final_answer": final_answer,
    }


def handle_max_retries(state: AgenticRAGState) -> AgenticRAGState:
    """Handle case where max retries are reached - USE BEST RESPONSE"""
    best_response = state["best_response"]
    best_confidence = state["best_confidence"]
    best_query = state["best_query"]

    print(
        f"⚠️ Max retries reached. Using BEST response with {best_confidence}% confidence"
    )
    print(f"📊 Best response was from query: '{best_query}'")

    final_answer = f"[Max retries reached - Using BEST response: {best_confidence}%] {best_response}"

    return {
        **state,
        "response": best_response,
        "confidence_score": best_confidence,
        "final_answer": final_answer,
    }


def create_agentic_rag_flow():
    """Create the agentic RAG flow with AWS Bedrock and Knowledge Base"""
    workflow = StateGraph(AgenticRAGState)

    # Add nodes
    workflow.add_node("initial_retrieval", initial_retrieval)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("optimize_query", optimize_query_node)
    workflow.add_node("finalize_response", finalize_response)
    workflow.add_node("handle_max_retries", handle_max_retries)

    # Add edges
    workflow.add_edge(START, "initial_retrieval")
    workflow.add_edge("initial_retrieval", "generate_response")

    # Router logic
    workflow.add_conditional_edges(
        "generate_response",
        confidence_router,
        {
            "end_response": "finalize_response",
            "optimize_query": "optimize_query",
            "max_retries_reached": "handle_max_retries",
        },
    )

    # Query optimization leads back to retrieval
    workflow.add_edge("optimize_query", "initial_retrieval")

    # End states
    workflow.add_edge("finalize_response", END)
    workflow.add_edge("handle_max_retries", END)

    return workflow.compile()


def run_agentic_rag_with_aws(kb_id: str, queries: List[str]):
    """Run the agentic RAG system with AWS Bedrock and Knowledge Base"""

    # Validate KB ID
    if not kb_id:
        print("❌ Please provide your Knowledge Base ID")
        return

    app = create_agentic_rag_flow()

    # Print flow structure
    print("🔄 Workflow Structure:")
    mermaid_ascii = app.get_graph().draw_ascii()
    print(mermaid_ascii)

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"🔍 TEST CASE {i}: '{query}'")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        initial_state = {
            "user_query": query,
            "original_query": query,
            "search_results": [],
            "response": "",
            "confidence_score": 0.0,
            "retry_count": 0,
            "max_retries": 4,  # Increased for better optimization
            "final_answer": "",
            "best_response": "",
            "best_confidence": 0.0,
            "best_query": "",
            "kb_id": kb_id,
        }

        # Run the flow
        start_time = time.time()
        try:
            result = app.invoke(initial_state)
            end_time = time.time()

            print(f"\n🎯 FINAL RESULT:")
            print(f"Original Query: {result['original_query']}")
            print(f"Final Query Used: {result['user_query']}")
            print(f"Final Answer: {result['final_answer']}")
            print(f"Total Retries: {result['retry_count']}")
            print(f"Best Query: {result['best_query']}")
            print(f"Best Confidence Achieved: {result['best_confidence']}%")
            print(f"Processing Time: {end_time - start_time:.2f}s")
            print(f"Documents Retrieved: {len(result['search_results'])}")

        except Exception as e:
            print(f"❌ Error running workflow: {str(e)}")
            print("Please check your AWS credentials and Knowledge Base ID")

        time.sleep(2)  # Brief pause between tests


if __name__ == "__main__":
    print("🚀 AWS Bedrock Agentic RAG with Claude Sonnet 3.5 v2 and Knowledge Base")
    print("=" * 80)

    # REPLACE WITH YOUR ACTUAL KNOWLEDGE BASE ID
    KB_ID = "K1WZTTYXAI"

    if not KB_ID:
        print("❌ Knowledge Base ID is required!")
        exit(1)

    # Test queries - customize these for your domain
    test_queries = [
        "What are aws compliance tips?"
        # "How does natural language processing work?",
        # "Explain the differences between supervised and unsupervised learning",
    ]

    # Get custom query from user
    custom_query = input("Enter a custom query (or press Enter to skip): ").strip()
    if custom_query:
        test_queries.append(custom_query)

    print(f"\n🔍 Testing with {len(test_queries)} queries...")
    print(f"📚 Knowledge Base ID: {KB_ID}")

    # Run the agentic RAG system
    run_agentic_rag_with_aws(KB_ID, test_queries)
