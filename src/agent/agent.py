from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from src.agent.tools import  temporal_aware_search, hybrid_search
from src.agent.singleton_connection import get_connector
from src.config.safe_llm import SafeLLM

SYSTEM_PROMPT = '''You are a helpful assistant. Answer questions using the following tools:

{tools}

Use one or many of: [{tool_names}]

Tool Requirements:
- `hybrid_search`: Required: {{"query": "<search topic>", "limit": <number>}}. Optional: other parameters.
- `temporal_aware_search`: Required: {{"query": "<topic>", "reference_time": "<YYYY-MM-DD>", "limit": <number>}}. Optional: other parameters.

Use this format:
Question: the input question to answer
Thought: why you choose the tool
Action: one of [{tool_names}]
Action Input: JSON with double-quoted keys and values, no nesting
Observation: tool result
Thought: I have enough information
Final Answer: answer based on Observations

Rules:
- Use `hybrid_search` for most questions.
- Use `temporal_aware_search` for questions with dates (e.g., "in 2023").
- No inventing; use only Observations.
- Action Input: JSON like {{"key": "value"}}.

Begin!

Question: {input}
Thought: {agent_scratchpad}'''


def create_graphiti_agent() -> AgentExecutor:
    """
    Crea un agente ReAct con herramientas de Graphiti.
    
    Args:
        llm_config: Configuración del LLM (api_key, endpoint, deployment, etc.)
    
    Returns:
        AgentExecutor listo para usar
    """
    
    graphiti_connector = get_connector()

    # Define las herramientas
    tools = [temporal_aware_search, hybrid_search]
    
    # Crea el prompt template
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )
    
    # Crea el agente ReAct
    agent = create_react_agent(
        llm=graphiti_connector.get_openai_client_chat(),
        tools=tools,
        prompt=prompt
    )
    
    # Crea el executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True  
    )
    
    return agent_executor