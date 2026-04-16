import json
import os
from typing import List, Dict, Optional
from datetime import datetime
import uuid


class SessionMemory:
    """Manages session-based conversation history and memory."""
    
    def __init__(self, memory_dir: str = "data/session_memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
    
    def create_session(self) -> str:
        """Create a new session and return session ID."""
        session_id = str(uuid.uuid4())
        session_file = os.path.join(self.memory_dir, f"{session_id}.json")
        
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "conversation_history": []
        }
        
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
        
        return session_id
    
    def add_to_history(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the session history."""
        session_file = os.path.join(self.memory_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return
        
        with open(session_file, "r") as f:
            session_data = json.load(f)
        
        session_data["conversation_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get the conversation history for a session."""
        session_file = os.path.join(self.memory_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return []
        
        with open(session_file, "r") as f:
            session_data = json.load(f)
        
        return session_data.get("conversation_history", [])
    
    def get_session_context(self, session_id: str, limit: int = 10) -> str:
        """Get formatted context of previous conversations (last N exchanges)."""
        history = self.get_session_history(session_id)
        
        if not history:
            return ""
        
        # Get last 'limit' exchanges
        recent_history = history[-limit:]
        
        context_lines = ["## Previous Conversation Context (for pronoun resolution):\n"]
        
        for i, msg in enumerate(recent_history):
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            
            # For user questions, keep full content (important for pronouns)
            # For assistant answers, truncate to save tokens but keep important details
            if role == "USER":
                context_lines.append(f"Q: {content}")
            else:
                # Keep first 250 chars of answer for context
                truncated = content[:250] + "..." if len(content) > 250 else content
                context_lines.append(f"A: {truncated}")
        
        context_lines.append("\n(Use the above Q&A history to resolve pronouns like 'it', 'this', 'that', etc.)")
        
        return "\n".join(context_lines)
    
    def get_last_question(self, session_id: str) -> Optional[str]:
        """Get the last question asked in the session."""
        history = self.get_session_history(session_id)
        
        # Find the last user message
        for msg in reversed(history):
            if msg.get("role") == "user":
                return msg.get("content")
        
        return None
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get a summary of the session."""
        session_file = os.path.join(self.memory_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return {}
        
        with open(session_file, "r") as f:
            session_data = json.load(f)
        
        history = session_data.get("conversation_history", [])
        user_messages = [msg for msg in history if msg.get("role") == "user"]
        
        return {
            "session_id": session_id,
            "created_at": session_data.get("created_at"),
            "total_messages": len(history),
            "total_questions": len(user_messages),
            "conversation_history": history
        }
    
    def clear_session(self, session_id: str) -> None:
        """Clear all messages in a session."""
        session_file = os.path.join(self.memory_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return
        
        with open(session_file, "r") as f:
            session_data = json.load(f)
        
        session_data["conversation_history"] = []
        
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session completely."""
        session_file = os.path.join(self.memory_dir, f"{session_id}.json")
        
        if os.path.exists(session_file):
            os.remove(session_file)
