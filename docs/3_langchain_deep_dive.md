# 3. LangChain Deep Dive: 2025 Production Patterns and Enterprise Integration

> [!NOTE]
> This section demonstrates LangChain's 2025 evolution to production-ready systems with LangGraph, LangSmith monitoring, and comprehensive evaluation frameworks. Focus on enterprise patterns and production deployment strategies.

Learn the cutting-edge patterns that make LangChain a powerful choice for both research and production systems in 2025.

## 3.1 Critical 2025 Updates

> [!WARNING]
> Legacy AgentExecutor and RunnableAgent patterns are deprecated. This guide focuses on LangGraph, LangSmith, and production-ready 2025 approaches.

### **Deprecated Patterns (Avoid These)**
```python
# ❌ DEPRECATED - Don't use LLMChain anymore
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# ❌ DEPRECATED - Old sequential chain pattern
from langchain.chains import SequentialChain
```

### **Modern LCEL Patterns (Use These)**
```python
# ✅ MODERN - Use LCEL composition
chain = prompt | llm | StrOutputParser()

# ✅ MODERN - Complex LCEL workflows
chain = (
    ChatPromptTemplate.from_template("Question: {input}")
    | AzureChatOpenAI() 
    | StrOutputParser()
    | {"result": RunnablePassthrough(), "analysis": analysis_chain}
)
```

## 3.2 Azure OpenAI Integration

### 3.2.1 Basic Setup with Environment Management

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI configuration
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
    temperature=0.7,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)
```

## 3.3 Modern LCEL Composition Patterns

### 3.3.1 Basic Chain Composition

```python
# Simple prompt-to-response chain
basic_chain = (
    ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
    | llm
    | StrOutputParser()
)

response = basic_chain.invoke({"topic": "quantum computing"})
```

### 3.3.2 Complex Multi-Step Workflows

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Multi-step analysis chain
analysis_chain = (
    ChatPromptTemplate.from_template("Analyze the following text: {text}")
    | llm
    | StrOutputParser()
)

summary_chain = (
    ChatPromptTemplate.from_template("Summarize: {text}")
    | llm
    | StrOutputParser()
)

# Parallel processing
parallel_chain = RunnableParallel({
    "analysis": analysis_chain,
    "summary": summary_chain,
    "original": RunnablePassthrough()
})

result = parallel_chain.invoke({"text": "Your input text here"})
```

### 3.3.3 Sequential Chain Replacement (Modern Pattern)

```python
# Modern replacement for deprecated SequentialChain
def create_marketing_chain():
    # Step 1: Generate company name
    name_chain = (
        ChatPromptTemplate.from_template(
            "What is a good name for a company that makes {product}?"
        )
        | llm
        | StrOutputParser()
    )
    
    # Step 2: Generate catchphrase
    catchphrase_chain = (
        ChatPromptTemplate.from_template(
            "Write a creative catchphrase for {company_name}"
        )
        | llm
        | StrOutputParser()
    )
    
    # Combined sequential workflow
    def marketing_workflow(inputs):
        company_name = name_chain.invoke(inputs)
        catchphrase = catchphrase_chain.invoke({"company_name": company_name})
        return {
            "company_name": company_name,
            "catchphrase": catchphrase,
            "product": inputs["product"]
        }
    
    return marketing_workflow

# Usage
marketing_chain = create_marketing_chain()
result = marketing_chain({"product": "eco-friendly socks"})
```

## 3.4 Advanced Tool Integration and Agents

### 3.4.1 Modern Tool Definition

```python
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Custom tool with proper typing
@tool
def calculate_expression(expression: str) -> str:
    """
    A calculator that safely evaluates mathematical expressions.
    
    Args:
        expression: A mathematical expression like "2+2" or "sqrt(16)"
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Safe evaluation for mathematical expressions
        import ast
        import operator
        
        # Supported operations
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        def eval_expr(expr):
            return eval_node(ast.parse(expr, mode='eval').body)
        
        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
            elif isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](eval_node(node.operand))
            else:
                raise TypeError(node)
        
        return str(eval_expr(expression))
    except Exception as e:
        return f"Error: {str(e)}"

# Search tool with API key management
search_tool = TavilySearchResults(
    max_results=3,
    api_key=os.getenv("TAVILY_API_KEY")
)
```

### 3.4.2 Modern Agent Implementation

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub

# Initialize tools
tools = [search_tool, calculate_expression]

# Get the latest agent prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# Create agent with Azure OpenAI
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create executor with enhanced configuration
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    max_execution_time=30
)

# Execute complex task
response = agent_executor.invoke({
    "input": "Search for the latest Azure OpenAI pricing, then calculate the cost for 1 million tokens at the current rate."
})
```

## 3.5 LangGraph Multi-Agent Systems (2024 Strength)

### 3.5.1 Supervisor Pattern with Multiple Agents

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, List
import json

# Define shared state
class AgentState(TypedDict):
    messages: List[str]
    current_agent: str
    task: str
    results: dict

# Specialized agents
research_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature=0.1  # Lower temperature for research
)

creative_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature=0.8  # Higher temperature for creativity
)

# Agent functions
def research_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "Research and gather factual information about: {task}"
    )
    chain = prompt | research_llm | StrOutputParser()
    result = chain.invoke({"task": state["task"]})
    
    return {
        **state,
        "messages": state["messages"] + [f"Research: {result}"],
        "results": {**state["results"], "research": result}
    }

def creative_agent(state: AgentState):
    research_data = state["results"].get("research", "")
    prompt = ChatPromptTemplate.from_template(
        "Based on this research: {research}\\n\\nCreate engaging content for: {task}"
    )
    chain = prompt | creative_llm | StrOutputParser()
    result = chain.invoke({"research": research_data, "task": state["task"]})
    
    return {
        **state,
        "messages": state["messages"] + [f"Creative: {result}"],
        "results": {**state["results"], "creative": result}
    }

def supervisor(state: AgentState):
    """Decide which agent should act next"""
    if "research" not in state["results"]:
        return "research"
    elif "creative" not in state["results"]:
        return "creative"
    else:
        return "end"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("research", research_agent)
workflow.add_node("creative", creative_agent)

# Add conditional routing
workflow.add_conditional_edges(
    "research",
    supervisor,
    {
        "creative": "creative",
        "end": END
    }
)

workflow.add_conditional_edges(
    "creative",
    supervisor,
    {
        "research": "research",
        "end": END
    }
)

# Set entry point
workflow.set_entry_point("research")

# Compile the graph
app = workflow.compile()
```

### 3.5.2 Collaborative Agent Workflow

```python
# Execute the multi-agent workflow
initial_state = {
    "messages": [],
    "current_agent": "",
    "task": "Create a comprehensive guide about sustainable technology trends",
    "results": {}
}

# Run the workflow
final_state = app.invoke(initial_state)

print("Multi-Agent Results:")
print("Research Phase:", final_state["results"]["research"])
print("Creative Phase:", final_state["results"]["creative"])
```

## 3.6 Streaming and Async Patterns

### 3.6.1 Streaming Responses

```python
# Streaming with LCEL
async def stream_response(query: str):
    chain = (
        ChatPromptTemplate.from_template("Answer: {query}")
        | llm
        | StrOutputParser()
    )
    
    async for chunk in chain.astream({"query": query}):
        print(chunk, end="", flush=True)

# Usage
import asyncio
asyncio.run(stream_response("Explain machine learning"))
```

### 3.6.2 Async Agent Execution

```python
# Async agent for better performance
async def async_agent_task(task: str):
    response = await agent_executor.ainvoke({"input": task})
    return response["output"]

# Parallel task execution
async def parallel_research():
    tasks = [
        "Research quantum computing developments",
        "Find latest AI ethics guidelines", 
        "Analyze renewable energy trends"
    ]
    
    results = await asyncio.gather(*[
        async_agent_task(task) for task in tasks
    ])
    
    return dict(zip(tasks, results))
```

## 3.7 Error Handling and Production Patterns

### 3.7.1 Robust Error Handling

```python
from langchain_core.runnables import RunnableLambda

def safe_chain_execution(chain, inputs, max_retries=3):
    """Execute chain with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Failed after {max_retries} attempts: {str(e)}"
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff

# Usage with error handling
robust_chain = RunnableLambda(
    lambda x: safe_chain_execution(basic_chain, x)
)
```

### 3.7.2 Configuration Management

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class LangChainConfig:
    azure_endpoint: str
    api_key: str
    deployment_name: str
    api_version: str = "2024-05-01-preview"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    @classmethod
    def from_env(cls):
        return cls(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        )
    
    def create_llm(self):
        return AzureChatOpenAI(
            azure_deployment=self.deployment_name,
            api_version=self.api_version,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key
        )

# Usage
config = LangChainConfig.from_env()
llm = config.create_llm()
```

## 3.8 2025 LangGraph and Production Patterns

### Memory-Enabled Multi-Agent Systems

```python
# 2025 PATTERN: Production LangGraph with persistent memory
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent

# Production memory backend
memory = PostgresSaver(conn_string="postgresql://...")

# Memory-enabled agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    memory_config={'max_history': 1000}
)

# Persistent conversations
config = {'configurable': {'thread_id': 'user-session-123'}}
response = await agent.ainvoke(
    {'messages': [('user', 'Continue our conversation')]},
    config=config
)
```

### LangSmith Production Monitoring

```python
# 2025 PATTERN: Production monitoring and evaluation
import os
from langsmith import Client

# Enable production tracing
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'ProductionApp'

client = Client()

# Create evaluation dataset
dataset = client.create_dataset("agent_evaluation")

# Run evaluation pipeline
evaluation_results = client.evaluate(
    agent_function,
    data=dataset,
    evaluators=[relevancy_evaluator, faithfulness_evaluator]
)
```

### DeepEval Integration for Testing

```python
# 2025 PATTERN: Comprehensive agent testing
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# Create test cases
test_cases = [
    LLMTestCase(
        input="Analyze market trends",
        actual_output=agent_response,
        expected_output=expected_response,
        context=["Market data context"]
    )
]

# Run comprehensive evaluation
metrics = [
    AnswerRelevancyMetric(threshold=0.8),
    FaithfulnessMetric(threshold=0.7)
]

results = evaluate(test_cases, metrics)
```

## 3.9 LangChain's 2025 Production Strengths Summary

### 3.9.1 What LangChain Excels At in 2025:

> [!TIP]
> LangChain's 2025 architecture makes it ideal for both research innovation and production deployment.

- [x] **Production Monitoring**: LangSmith provides comprehensive observability
- [x] **Memory-Enabled Agents**: Persistent conversation context across sessions
- [x] **Evaluation Frameworks**: Built-in integration with DeepEval and custom metrics
- [x] **Cost Optimization**: Advanced token tracking and performance monitoring
- [x] **Enterprise Ready**: Production-grade error handling and scaling patterns
- [x] **Visual Debugging**: LangGraph Studio for workflow optimization

### 3.9.2 2025 Best Use Cases:

<details>
<summary>🏭 <strong>Production Applications</strong></summary>

- Enterprise multi-agent systems with memory requirements
- Production applications requiring comprehensive monitoring
- Systems needing advanced evaluation and testing frameworks

</details>

<details>
<summary>🔬 <strong>Research & Innovation Applications</strong></summary>

- Advanced agent research with state-of-the-art capabilities
- Academic projects requiring cutting-edge multi-agent patterns
- Novel AI technique exploration with production deployment paths

</details>

<details>
<summary>⚡ <strong>Rapid Development Applications</strong></summary>

- Complex workflows requiring visual debugging (LangGraph Studio)
- Applications needing real-time streaming and progress updates
- Systems requiring comprehensive cost and performance tracking

</details>

LangChain's 2025 evolution to production-ready systems with LangGraph, LangSmith, and evaluation frameworks demonstrates its maturation from research tool to enterprise platform while maintaining innovation flexibility.