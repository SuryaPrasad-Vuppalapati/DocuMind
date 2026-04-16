import os
import sys

# Add project root to path so we can import things
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents import run_agent_team

def interactive_loop():
    print("====================================")
    print("Welcome to DocuMind RAG! (Human-in-the-Loop Mode)")
    print("Type your query or 'exit' to quit.")
    print("====================================")

    while True:
        try:
            # 1. Take initial user Input
            query = input("\n[User] Enter your question: ")
            if query.lower() in ("exit", "quit", "q"):
                 break
            
            # 2. Add an optional Human Evaluation step BEFORE triggering AI
            print("\n[System] Checking your query for clarity...")
            refine = input(f"[System] Do you want to add more context to '{query}'? (y/N): ")
            if refine.lower() == 'y':
                context = input("Add context: ")
                query += f". Additional Context: {context}"
                print(f"[System] Updated Query: {query}")

            # 3. Trigger Agent Orchestration
            print("\n[System] Activating Multi-Agent Team to analyze and synthesize an answer...")
            draft_answer = run_agent_team(query)

            # 4. Human-In-The-Loop Revision Step AFTER AI Generation
            print("\n[Agent Writer Draft Output]:")
            print(draft_answer)
            print("-" * 50)

            approval = input("\n[Human-in-the-Loop] Do you approve this draft? (yes/edit/reject): ").strip().lower()
            
            if approval in ["y", "yes"]:
                print("\n✅ Final Result Approved! Task Complete.")
            elif approval == "edit":
                 feedback = input("\n[Human] Provide editing instructions to the Writer Agent: ")
                 # Here we would normally re-trigger the Writer Agent with feedback
                 print(f"[System] Sending feedback to Writer Agent: {feedback}")
                 print("(Editing loop simulated for demonstration. The Writer will incorporate this to draft v2.)")
            else:
                 print("\n❌ Task Rejected by Human Operator. Try rephrasing the initial question.")
                 
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}")

if __name__ == "__main__":
    interactive_loop()
