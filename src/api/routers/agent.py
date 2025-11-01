"""
Agent Router

API endpoints for AI agent interactions and conversations
"""

import os
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

try:
    from ...agent.core import NumericsAgent, AgentConfig
except ImportError:
    # Mock for development when agent framework is not installed
    class NumericsAgent:
        def __init__(self, *args, **kwargs):
            pass
        async def process_message(self, message: str) -> str:
            return "Agent framework not installed. Please install agent-framework-azure-ai"
    
    class AgentConfig:
        def __init__(self, *args, **kwargs):
            pass


router = APIRouter(prefix="/agent", tags=["agent"])


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for chat interaction"""
    message: str
    conversation_id: Optional[str] = None
    use_tools: bool = True


class ChatResponse(BaseModel):
    """Response model for chat interaction"""
    response: str
    conversation_id: str
    tools_used: List[str]
    response_time: float


class ConversationHistory(BaseModel):
    """Conversation history model"""
    conversation_id: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str


# In-memory storage for conversations (in production, use a database)
conversations: Dict[str, List[ChatMessage]] = {}

# Global agent instance
agent_instance = None


async def get_agent():
    """Get or create agent instance"""
    global agent_instance
    if agent_instance is None:
        try:
            config = AgentConfig(
                github_token=os.getenv("GITHUB_TOKEN"),
                model_id=os.getenv("MODEL_ID", "openai/gpt-4.1-mini")
            )
            agent_instance = NumericsAgent(config)
        except Exception as e:
            print(f"Warning: Could not initialize agent: {e}")
            agent_instance = NumericsAgent()  # Use mock
    return agent_instance


@router.post("/chat", response_model=ChatResponse, summary="Chat with the AI agent")
async def chat_with_agent(request: ChatRequest):
    """Send a message to the AI agent and get a response"""
    try:
        start_time = datetime.now()
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{int(datetime.now().timestamp())}"
        
        # Initialize conversation history if new
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Add user message to history
        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.now().isoformat()
        )
        conversations[conversation_id].append(user_message)
        
        # Get agent and process message
        agent = await get_agent()
        
        try:
            response_content = await agent.process_message(request.message)
            tools_used = []  # TODO: Extract from agent response
        except Exception as e:
            response_content = f"I apologize, but I encountered an error: {str(e)}"
            tools_used = []
        
        # Add assistant message to history
        assistant_message = ChatMessage(
            role="assistant",
            content=response_content,
            timestamp=datetime.now().isoformat()
        )
        conversations[conversation_id].append(assistant_message)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            response=response_content,
            conversation_id=conversation_id,
            tools_used=tools_used,
            response_time=response_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@router.get("/conversations", summary="List all conversations")
async def list_conversations():
    """Get a list of all conversations"""
    try:
        conversation_list = []
        for conv_id, messages in conversations.items():
            if messages:
                conversation_list.append({
                    "conversation_id": conv_id,
                    "message_count": len(messages),
                    "created_at": messages[0].timestamp,
                    "updated_at": messages[-1].timestamp,
                    "preview": messages[-1].content[:100] + "..." if len(messages[-1].content) > 100 else messages[-1].content
                })
        
        return {"conversations": conversation_list}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory, summary="Get conversation history")
async def get_conversation(conversation_id: str):
    """Get the full history of a specific conversation"""
    try:
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = conversations[conversation_id]
        created_time = messages[0].timestamp if messages and messages[0].timestamp else datetime.now().isoformat()
        updated_time = messages[-1].timestamp if messages and messages[-1].timestamp else datetime.now().isoformat()
        
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=messages,
            created_at=created_time,
            updated_at=updated_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation: {str(e)}")


@router.delete("/conversations/{conversation_id}", summary="Delete a conversation")
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation"""
    try:
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        del conversations[conversation_id]
        
        return {"message": f"Conversation {conversation_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")


@router.websocket("/ws/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    # Initialize conversation if new
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    try:
        agent = await get_agent()
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if not user_message:
                continue
            
            # Add user message to history
            user_msg = ChatMessage(
                role="user",
                content=user_message,
                timestamp=datetime.now().isoformat()
            )
            conversations[conversation_id].append(user_msg)
            
            try:
                # Process with agent
                response_content = await agent.process_message(user_message)
                
                # Add assistant message to history
                assistant_msg = ChatMessage(
                    role="assistant",
                    content=response_content,
                    timestamp=datetime.now().isoformat()
                )
                conversations[conversation_id].append(assistant_msg)
                
                # Send response back to client
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "content": response_content,
                    "timestamp": assistant_msg.timestamp
                }))
                
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                }))
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@router.get("/status", summary="Get agent status")
async def get_agent_status():
    """Get the current status of the AI agent"""
    try:
        agent = await get_agent()
        
        status = {
            "status": "online" if agent else "offline",
            "model_id": getattr(agent, 'config', {}).get('model_id', 'unknown'),
            "total_conversations": len(conversations),
            "total_messages": sum(len(msgs) for msgs in conversations.values()),
            "uptime": "unknown"  # TODO: Calculate actual uptime
        }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent status: {str(e)}")