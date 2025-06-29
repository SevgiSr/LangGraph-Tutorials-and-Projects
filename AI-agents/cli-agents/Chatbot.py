# chatbot.py

import os
import logging
from typing import TypedDict, List, Union

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# --- Dynamic Path for Logging ---
# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Create the full path for the log file to ensure it's in the same directory
log_file_path = os.path.join(script_dir, "chatbot.log")

# --- Setup Logging ---
# This provides structured, timestamped logs which are essential for production.
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path), # Log to a file
                        logging.StreamHandler() # Also log to console
                    ])

# --- Load Environment Variables ---
load_dotenv()

# --- Define Agent State ---
class AgentState(TypedDict):
    """The state of our chatbot, containing the conversation history."""
    messages: List[Union[HumanMessage, AIMessage]]

class Chatbot:
    """
    Encapsulates the LangGraph chatbot logic, making it modular and production-ready.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initializes the chatbot, sets up the language model and the graph.
        """
        logging.info(f"Initializing Chatbot with model: {model_name}")
        self.llm = ChatOpenAI(model=model_name)
        self.graph = self._build_graph()
        self.agent = self.graph.compile()
        self.conversation_history = []

    def _invoke_llm(self, state: AgentState) -> AgentState:
        """
        Node that calls the language model. This is the core processing step.
        It handles the API call and potential errors.
        """
        logging.info("Invoking LLM to get a response.")
        try:
            # The invoke method gets the latest state of the graph
            response = self.llm.invoke(state["messages"])
            # We append the AI's response to the message list
            state["messages"].append(AIMessage(content=response.content))
            logging.info(f"LLM generated response successfully.")
        except Exception as e:
            logging.error(f"Error invoking LLM: {e}")
            # Append an error message to the history to inform the user
            error_message = "Sorry, I encountered an error. Please try again."
            state["messages"].append(AIMessage(content=error_message))
        
        return state

    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph computational graph.
        """
        logging.info("Building the agent graph.")
        graph = StateGraph(AgentState)
        graph.add_node("process", self._invoke_llm)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        return graph

    def get_response(self, user_input: str) -> str:
        """
        Gets a response from the chatbot for a given user input.
        Manages the conversation history.
        """
        self.conversation_history.append(HumanMessage(content=user_input))
        
        # Invoke the agent with the current state (our message history)
        result = self.agent.invoke({"messages": self.conversation_history})
        
        # Update our history with the new messages from the run
        self.conversation_history = result["messages"]
        
        # The AI's response is the last message in the list
        ai_response = result["messages"][-1].content
        return ai_response

    def save_conversation(self, filename: str = "conversation_log.txt"):
        """
        Saves the full conversation history to a text file.
        """
        # --- Apply the same dynamic path logic here ---
        filepath = os.path.join(script_dir, filename)

        logging.info(f"Saving conversation to {filepath}")
        with open(filepath, "w", encoding="utf-8") as file:
            file.write("--- Conversation Log ---\n\n")
            for message in self.conversation_history:
                if isinstance(message, HumanMessage):
                    file.write(f"You: {message.content}\n")
                elif isinstance(message, AIMessage):
                    file.write(f"AI: {message.content}\n\n")
            file.write("--- End of Conversation ---")
        print(f"\nConversation saved to {filepath}")



# --- Main Execution Block ---
# It's good practice to put your executable code inside this block
if __name__ == "__main__":
    """
    Runs the command-line interface for the chatbot.
    """
    # Instantiate our chatbot
    bot = Chatbot()
    
    print("Chatbot initialized. Type 'exit' to quit.")
    print("-" * 50)

    try:
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit condition
            if user_input.lower() == "exit":
                print("Exiting chatbot.")
                break
            
            # Get response from the bot
            ai_response = bot.get_response(user_input)
            
            # Print AI response
            print(f"AI: {ai_response}")

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nCaught interrupt signal. Shutting down and saving conversation.")
    
    finally:
        # Save the conversation log before exiting
        bot.save_conversation()
