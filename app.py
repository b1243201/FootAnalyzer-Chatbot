from flask import Flask, render_template, request, jsonify, session
from LLM import init_rag, answer_question
import uuid #用來生成獨一無二的 ID(之後用來識別每個使用者的對話）
import os
from datetime import datetime, timedelta #用來記錄對話時間，清理過期對話
from dotenv import load_dotenv

# 載入環境變數
load_dotenv("key.env")


app = Flask(__name__) #此檔案為主程式
app.secret_key = os.getenv('FLASK_SECRET_KEY')


# 初始化 RAG 系統
try:
    vector_store, llm, prompt_template, embeddings = init_rag()
    print("RAG 系統初始化成功")
except Exception as e:
    print(f"RAG 系統初始化失敗: {e}")
    vector_store = llm = prompt_template = embeddings = None

conversation_storage = {} #用來存取所有使用者的對話歷史
SESSION_TIMEOUT = 30 #在網站上超過30分鐘沒有活動，就清除他的對話記錄
MAX_HISTORY_LENGTH = 10 #最多保存10輪對話

def get_session_id():
    """獲取或創建會話ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['created_at'] = datetime.now().isoformat()
    return session['session_id']

def clean_old_conversations():
    """清理過期的對話記錄"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, data in conversation_storage.items():
        if 'last_active' in data:
            last_active = datetime.fromisoformat(data['last_active'])
            if current_time - last_active > timedelta(minutes=SESSION_TIMEOUT):
                expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del conversation_storage[session_id]
        print(f"清理過期會話: {session_id}")

def get_conversation_history(session_id):
    """獲取對話歷史"""
    if session_id in conversation_storage:
        return conversation_storage[session_id].get('history', [])
    return []

def save_conversation_exchange(session_id, user_message, bot_response, strategy, relevance_score):
    """保存一輪對話"""
    if session_id not in conversation_storage:
        conversation_storage[session_id] = {
            'history': [],
            'created_at': datetime.now().isoformat()
        }
    
    # 添加新的對話
    conversation_storage[session_id]['history'].append({
        'user': user_message,
        'bot': bot_response,
        'strategy': strategy,
        'relevance_score': float(relevance_score), 
        'timestamp': datetime.now().isoformat()
    })
    
    # 更新最後活動時間
    conversation_storage[session_id]['last_active'] = datetime.now().isoformat()
    
    # 限制歷史長度
    if len(conversation_storage[session_id]['history']) > MAX_HISTORY_LENGTH:
        conversation_storage[session_id]['history'] = conversation_storage[session_id]['history'][-MAX_HISTORY_LENGTH:]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        # 檢查系統是否正常初始化
        if not all([vector_store, llm, prompt_template, embeddings]):
            return jsonify({
                "error": "系統尚未正確初始化，請聯繫管理員",
                "strategy": "error",
                "relevance_score": 0.0
            }), 500
        
        # 定期清理過期會話
        clean_old_conversations()
        
        # 獲取請求數據
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({
                "error": "問題不可為空",
                "strategy": "error", 
                "relevance_score": 0.0
            }), 400
        
        # 獲取會話ID和對話歷史
        session_id = get_session_id()
        conversation_history = get_conversation_history(session_id)
        
        print(f"會話 {session_id}: 用戶問題 - {question}")
        print(f"對話歷史長度: {len(conversation_history)}")
        
        # 呼叫 RAG 系統回答問題
        result = answer_question(
            question=question,
            vector_store=vector_store,
            llm=llm,
            prompt_template=prompt_template,
            embeddings=embeddings,
            conversation_history=conversation_history
        )
        
        # 保存這輪對話
        save_conversation_exchange(
            session_id=session_id,
            user_message=question,
            bot_response=result["answer"],
            strategy=result["strategy"],
            relevance_score=result["relevance_score"]
        )
        
        print(f"回答策略: {result['strategy']}, 相關性分數: {result['relevance_score']:.3f}")
        
        return jsonify({
            "answer": result["answer"],
            "strategy": result["strategy"],
            "relevance_score": float(result["relevance_score"]), 
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"處理聊天請求時發生錯誤: {str(e)}")
        return jsonify({
            "error": f"處理請求時發生錯誤: {str(e)}",
            "strategy": "error",
            "relevance_score": 0.0
        }), 500

@app.route("/api/system_status", methods=["GET"])
def system_status():
    """獲取系統狀態"""
    try:
        return jsonify({
            "rag_initialized": all([vector_store, llm, prompt_template, embeddings]),
            "active_sessions": len(conversation_storage),
            "system_ready": all([vector_store, llm, prompt_template, embeddings]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversation_stats", methods=["GET"])
def conversation_stats():
    """獲取對話統計資訊（可選的除錯端點）"""
    try:
        session_id = get_session_id()
        conversation_history = get_conversation_history(session_id)
        
        stats = {
            "session_id": session_id,
            "total_exchanges": len(conversation_history),
            "strategies_used": {}
        }
        
        for exchange in conversation_history:
            strategy = exchange.get('strategy', 'unknown')
            stats["strategies_used"][strategy] = stats["strategies_used"].get(strategy, 0) + 1
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("=" * 50)
    print("啟用 FootAnalyzer 客服系統")
    print("=" * 50)
    print(f"RAG 系統: {'✓ 已初始化' if vector_store else '✗ 初始化失敗'}")
    print("=" * 50)
    
    app.run(debug=True, host="0.0.0.0", port=5000)