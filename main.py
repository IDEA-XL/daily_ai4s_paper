import os
import logging
from dotenv import load_dotenv, find_dotenv
from src.graph import get_graph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    The main entry point for the AI4Science Daily Paper Digestion Agent.
    """
    # Load environment variables from .env file
    load_dotenv(find_dotenv())
    
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("FATAL: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file and add your OPENAI_API_KEY.")
        return

    report_saving_dir = os.getenv("REPORT_SAVING_DIR", ".") # Default to current directory
    logging.info(f"Report saving directory set to: {report_saving_dir}")

    logging.info("Initializing the graph...")
    graph = get_graph(report_saving_dir=report_saving_dir)
    
    logging.info("Starting the AI4Science Daily Paper Digestion Agent...")
    
    initial_state = {
        "paper_candidates": [],
        "relevant_papers": [],
        "analyzed_papers": [],
        "markdown_report": "",
        "error": None
    }
    
    final_state = graph.invoke(initial_state)

    if final_state.get("error"):
        logging.error(f"The agent failed to complete its run. Error: {final_state['error']}")
        print(f"\nAn error occurred: {final_state['error']}")
    else:
        # The report is now saved by the synthesize_report_node within the graph
        # We just need to confirm its completion and potentially print the path
        report_filename = f"AI4Science_Report_{datetime.now().strftime('%Y-%m-%d')}.md"
        full_report_path = os.path.join(report_saving_dir, report_filename)
        logging.info(f"Graph execution completed. Report should be saved to {full_report_path}")
        print(f"\n✅ Daily report successfully generated and saved to '{full_report_path}'.")

if __name__ == "__main__":
    main()
