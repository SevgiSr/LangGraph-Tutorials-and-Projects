import os
import logging
import uuid
from typing import List, Union, TypedDict, AsyncGenerator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)

# --- Dynamic Path for Logging ---
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "chatbot.log")

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])

# --- Load Environment Variables ---
load_dotenv()

# We use manual state hangling instead of built in MessagesState, 
# so that we can fully replace our messages after trimming them
class AgentState(TypedDict):
    """The state of our chatbot, containing the conversation history."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]

class Chatbot:
    """
    Encapsulates the LangGraph chatbot logic, using a checkpointer for state management.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initializes the chatbot, sets up the language model, checkpointer, and compiles the graph.
        """
        logging.info(f"Initializing Chatbot with model: {model_name}")
        self.llm = ChatOpenAI(model=model_name, streaming=True) # IMPORTANT: Enable streaming on the model
        
        # 1. Initialize the checkpointer (memory) for the chatbot
        self.memory = InMemorySaver()
        
        # 2. Get the graph definition
        graph_definition = self._build_graph()
        
        # 3. Compile the agent, passing the checkpointer to enable memory
        self.agent = graph_definition.compile(checkpointer=self.memory)
        

    def trimmer(self, state: AgentState) -> dict:
        """
        Trims the conversation history to a fixed size to manage context and cost.
        This node runs before the main LLM call.
        """
        # I used short token size for easy testing
        print("------------Messages BEFORE: ------------")
        print([m.content for m in state["messages"]])
        trimmed_messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=128,
            start_on="human",
            end_on=("human", "tool"),
        )

        print("------------Messages AFTER: ------------")
        print([m.content for m in trimmed_messages])
       
       # instead of adding to existing messages,
       # it REPLACES messages entirely with trimmed messages
        return {"messages": trimmed_messages}
        

    def _invoke_llm(self, state: AgentState) -> dict:
        """
        Node that calls the language model. This is the core processing step.
        The state is now automatically loaded by the checkpointer.
        """
        logging.info("Invoking LLM to get a response.")
        # save old message history, 
        # to add new messages on top when updating the state
        message_history = state["messages"]
        try:
            response = self.llm.invoke(message_history)
            return {"messages": message_history + [response]}
        except Exception as e:
            logging.error(f"Error invoking LLM: {e}")
            error_message = "Sorry, I encountered an error. Please try again."
            return {"messages": message_history + [AIMessage(content=error_message)]}

    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph computational graph definition.
        This method no longer compiles the graph.
        """
        logging.info("Building the agent graph definition.")
        graph = StateGraph(AgentState)
        # Add the nodes
        graph.add_node("trimmer", self.trimmer)
        graph.add_node("process", self._invoke_llm)
        # Define the edges
        graph.add_edge(START, "trimmer")
        graph.add_edge("trimmer", "process")
        graph.add_edge("process", END)
        return graph

    def get_response(self, user_input: str, config: dict) -> str:
        """
        Gets a response from the chatbot for a given user input and conversation config.
        """
        # We need to get previous message history 
        # from the state to feed it to the llm

         # 1. Get the current state from the checkpointer.
        current_state = self.agent.get_state(config)
        
        # 2. Manually add the new user message to the history.
        current_messages = current_state.values.get("messages", [])
        updated_messages = current_messages + [HumanMessage(content=user_input)]
        
        # 3. Invoke the agent with the FULL updated list of messages.
        # This ensures the graph starts with the correct history before trimming.
        result = self.agent.invoke({"messages": updated_messages}, config=config)
        
        # The AI's response is the last message
        ai_response = result["messages"][-1].content
        return ai_response

    # The manual save_conversation method is no longer needed, as the checkpointer handles memory.
    # You could write a new method to fetch history from the checkpointer if you wanted to save it to a file.
    
    # --- NEW STREAMING METHOD ---
    async def stream_response(self, user_input: str, config: dict) -> AsyncGenerator[str, None]:
        """
        Gets a streaming response from the chatbot, yielding text chunks as they are generated.
        """
        current_state = self.agent.get_state(config)
        current_messages = current_state.values.get("messages", [])
        updated_messages = current_messages + [HumanMessage(content=user_input)]
        
        # Use astream_events for detailed, chunk-by-chunk processing
        # We are interested in the events from our "process" node (which calls the LLM)
        async for event in self.agent.astream_events(
            {"messages": updated_messages}, config=config, version="v1",
        ):
            # The event name is 'on_chat_model_stream' when chunks are produced
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    # Yield the content of the chunk immediately
                    yield chunk.content

#Note: I had to update the `stream_response` method to use `astream_events` and filter for the correct event type, as this is the modern and reliable way to get streaming chunks from a LangGraph agent.*


# --- Main Execution Block ---
import asyncio # Add this import at the top of your file

async def main():
    """
    Asynchronous main function to run the command-line interface with streaming.
    """
    # Instantiate our chatbot
    bot = Chatbot()
    
    # --- Setup for a single conversation thread ---
    # A unique ID for this specific conversation session.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("Chatbot initialized. Type 'exit' to quit.")
    print(f"Conversation Thread ID: {thread_id}")
    print("-" * 50)

    try:
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            
            if user_input.lower() == "exit":
                print("Exiting chatbot.")
                break
            
            print("AI: ", end="", flush=True)
            
            # Use "async for" to iterate through the streaming response
            async for chunk in bot.stream_response(user_input, config):
                # Print each chunk as it arrives, without a newline
                print(chunk, end="", flush=True)
            
            # Print a final newline after the full response is streamed
            print()

    except (KeyboardInterrupt, asyncio.CancelledError):
        # Handle Ctrl+C gracefully
        print("\nCaught interrupt signal. Shutting down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass