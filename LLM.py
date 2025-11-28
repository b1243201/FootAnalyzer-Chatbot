"""
LLM.py
這個檔案包含:
1.初始化RAG系統
    a.載入知識庫(medical_data.json)
    b.設定Gemini模型
    c.建立向量資料庫
2.回答問題的函數
    a.搜尋相關資料
    b.組合提示詞
    c.呼叫Gemini生成答案

主要函數：
1. init_rag()：初始化 RAG 系統（只執行一次）
2. answer_question()：回答問題（每次使用者提問都會呼叫）
3. calculate_relevance_score()：計算問題與知識庫的相關性
4. is_simple_greeting_or_social()：判斷是否為簡單問候語
5. is_system_capability_question()：判斷是否為系統功能查詢
"""

import os  # 讀取環境變數
import json  # 讀取json檔案
from dotenv import load_dotenv  # 載入.env檔案
from langchain_community.vectorstores import FAISS  # FAISS 向量資料庫
from langchain_huggingface import HuggingFaceEmbeddings  # 使用 HuggingFace 的 Embeddings
from langchain.prompts import PromptTemplate  # 提示詞模板
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini API
from langchain.schema import HumanMessage  # 訊息格式
import numpy as np  # 數值計算

def load_medical_data(file_path="medical_data.json"):
    """
    從JSON檔案載入知識庫
    
    Args:
        file_path (str): JSON檔案路徑
        
    Returns:
        list: 知識資料列表
    """
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:  # 打開檔案，使用 UTF-8 編碼
            data = json.load(f)  # 解析JSON檔案內容
        return data.get("medical_knowledge", [])  # 取得 medical_knowledge 欄位，若不存在則回傳空列表
    except FileNotFoundError:  # 如果檔案不存在
        print(f"找不到檔案: {file_path}")  # 印出錯誤訊息
        print("請確保 medical_data.json 檔案存在")  # 提示使用者檢查檔案
        return []  # 回傳空列表
    except json.JSONDecodeError:  # 如果 JSON 格式錯誤
        print(f"JSON 格式錯誤: {file_path}")  # 印出錯誤訊息
        return []  # 回傳空列表

def init_rag(data_file="medical_data.json", env_file="key.env"):
    """
    初始化 RAG 系統
    
    Args:
        data_file (str): 醫療資料檔案路徑
        env_file (str): 環境變數檔案路徑
        
    Returns:
        tuple: (vector_store, llm, prompt_template, embeddings)
    """
    # 載入環境變數
    load_dotenv(dotenv_path=env_file)  # 從指定的 .env 檔案讀取環境變數
    google_api_key = os.getenv("GOOGLE_API_KEY")  # 取得 Google API Key
    if not google_api_key:  # 如果沒有找到 API Key
        raise ValueError(f"未找到 GOOGLE_API_KEY，請在 {env_file} 中設定")  # 拋出錯誤

    # 載入知識庫資料
    medical_data = load_medical_data(data_file)  # 呼叫函數載入醫療知識庫
    if not medical_data:  # 如果知識庫是空的
        raise ValueError(f"無法載入醫療資料，請檢查 {data_file}")  # 拋出錯誤

    print(f"成功載入 {len(medical_data)} 筆知識庫資料")  # 印出成功訊息和資料筆數

    # 初始化 embeddings 和向量庫
    embeddings = HuggingFaceEmbeddings(  # 建立 Embeddings 物件
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # 使用免費的 HuggingFace 模型
    )
    
    try:  # 嘗試建立向量資料庫
        vector_store = FAISS.from_texts(medical_data, embeddings)  # 從文字列表建立 FAISS 向量資料庫
        print("向量資料庫建立成功")  # 印出成功訊息
    except Exception as e:  # 如果建立失敗
        raise ValueError(f"建立向量資料庫失敗: {e}")  # 拋出錯誤並顯示錯誤訊息

    # 初始化 Gemini 2.0 Flash 模型，目前免費版有一分鐘15次請求數的使用限制
    llm = ChatGoogleGenerativeAI(  # 建立 Gemini 模型物件
        model="gemini-2.0-flash",  # 設定 Gemini 2.0 Flash 模型
        google_api_key=google_api_key,  # 傳入 API Key
        temperature=1,  # 設定生成溫度（0=保守，1=創意）
        max_output_tokens=800  # 限制最多回答 800 個 token
    )

    #建立prompt模板
    prompt_template = PromptTemplate(
        input_variables=["context", "conversation_history", "question"],  # 三個輸入變數
        template=
        
        """你是一位AI衛教客服人員，擁有物理治療師、足科醫師及客製化鞋墊設計的專業背景。
        請只根據以下檢索到的專業資訊和對話歷史，以簡潔、專業且使用繁體中文的語氣回答問題。

        專業資訊：
        {context}

        對話歷史：
        {conversation_history}

        用戶問題：
        {question}

        回答原則：
        1. **識別問題類型**
        - 概念解釋類：提供清晰定義 → 舉例說明 → 延伸應用
        - 操作指導類：簡述目標 → 步驟說明 → 注意事項
        - 比較分析類：列出對象 → 關鍵差異 → 選擇建議
        - 無關問題：禮貌回應並引導回主題 

        2. **回答結構**
        - 先用 1-2 句話直接回答核心問題
        - 根據複雜度決定展開深度
        - 簡單問題保持簡潔，複雜問題再分層說明
        - 參考對話歷史，理解用戶問題的上下文脈絡，再做回覆
        - 如果用戶使用「它」、「這個」、「那個」等代詞，請結合對話歷史理解指涉內容
        - 根據資訊給出專業建議
        - 如涉及嚴重症狀，建議尋求專業醫療協助
        - 用淺顯易懂的方式解釋專業術語
        - 保持對話的連貫性和自然性
        - 以排版後的文字回答
        - 不提供商品的推薦或銷售相關資訊
        - 若無法從知識庫中找到相關答案，請誠實告知並建議用戶尋求專業醫療諮詢
        - 針對問候語或無關問題，請禮貌簡短回應並引導回足部健康相關話題
        """
            )

    return vector_store, llm, prompt_template, embeddings  # 回傳四個物件

def calculate_relevance_score(question, vector_store, embeddings, k=3):
    """
    計算問題與知識庫的相關性分數
    
    Args:
        question (str): 用戶問題
        vector_store: 向量資料庫
        embeddings: 嵌入模型
        k (int): 檢索相關文件數量
        
    Returns:
        float: 相關性分數 (0-1之間)
    """
    try: 
        # 檢索最相關的文件
        docs_with_scores = vector_store.similarity_search_with_score(question, k=k)  # 搜尋最相似的 k 個文件並取得分數
        
        if not docs_with_scores:  # 如果沒有找到任何文件
            return float(0.0)  # 回傳 0 分
        
        # 計算平均相似度分數
        # FAISS 返回的是距離，需要轉換為相似度
        scores = []  # 建立空列表存放分數
        for doc, distance in docs_with_scores:  # 遍歷每個文件和其距離
            # 距離轉相似度 (近似轉換)
            distance = float(distance)  # 將距離轉換為浮點數
            similarity = 1.0 / (1.0 + distance)  # 距離越小，相似度越高（使用公式轉換）
            scores.append(similarity)  # 將相似度加入列表
        
        # 返回最高相似度分數作為相關性分數
        max_score = max(scores) if scores else 0.0  # 取得最高分數，若列表為空則為 0
        result = min(float(max_score), 1.0)  # 確保分數不超過 1.0
        return result  # 回傳相關性分數
        
    except Exception as e:  # 如果發生任何錯誤
        print(f"計算相關性分數時發生錯誤: {e}")  # 印出錯誤訊息
        return float(0.0)  # 回傳 0 分

def format_conversation_history(conversation_history):
    """
    格式化對話歷史為字串
    
    Args:
        conversation_history (list): 對話歷史列表
        
    Returns:
        str: 格式化後的對話歷史
    """
    if not conversation_history:  # 如果對話歷史是空的
        return "（這是對話的開始）"  # 回傳起始訊息
    
    formatted_history = []  # 建立空列表存放格式化後的對話
    for i, exchange in enumerate(conversation_history, 1):  # 遍歷對話歷史，從 1 開始編號
        user_msg = exchange.get('user', '')  # 取得用戶訊息，若無則為空字串
        bot_msg = exchange.get('bot', '')  # 取得機器人訊息，若無則為空字串
        formatted_history.append(f"第{i}輪對話：")  # 加入對話輪數標題
        formatted_history.append(f"用戶：{user_msg}")  # 加入用戶訊息
        formatted_history.append(f"AI：{bot_msg}")  # 加入 AI 訊息
        formatted_history.append("")  # 加入空行分隔
    
    return "\n".join(formatted_history)  # 將列表轉換為字串，每個元素用換行分隔

def answer_question(question, vector_store, llm, prompt_template, embeddings, conversation_history=None, k=3):
    """
    使用知識庫回答問題
    
    Args:
        question (str): 用戶問題
        vector_store: 向量資料庫
        llm: 語言模型
        prompt_template: 回答模板
        embeddings: 嵌入模型
        conversation_history (list): 對話歷史
        k (int): 檢索相關文件數量
        
    Returns:
        dict: 回答結果（包含 answer, strategy, relevance_score）
    """
    try:  
        # 檢查是否為系統功能查詢
        if is_system_capability_question(question):  # 判斷是否詢問系統功能
            return {  # 直接回傳預設回答
                "answer": "我是 FootAnalyzer 的 AI 客服助手,可以提供足部相關健康諮詢服務。有什麼足部問題需要幫助嗎?",
                "strategy": "capability_question",  # 標記為系統功能查詢
                "relevance_score": 1.0  # 相關性設為最高
            }
        
        # 檢查是否為簡單問候語，直接回答
        if is_simple_greeting_or_social(question):  # 判斷是否為問候語
            simple_responses = {  # 定義問候語對應的回答字典
                '你好': '你好!我是 FootAnalyzer 的 AI 客服助手,很高興為您服務。請問有什麼足部健康問題需要諮詢呢?',
                '您好': '您好!我是 FootAnalyzer 的 AI 客服助手,很高興為您服務。請問有什麼足部健康問題需要諮詢呢?',
                'hello': 'Hello!我是 FootAnalyzer 的 AI 客服助手,很高興為您服務。請問有什麼足部健康問題需要諮詢呢?',
                'hi': 'Hi!我是 FootAnalyzer 的 AI 客服助手,很高興為您服務。請問有什麼足部健康問題需要諮詢呢?',
                '謝謝': '不客氣!很高興能為您提供幫助。如果還有其他足部健康問題,請隨時詢問。',
                '感謝': '不客氣!很高興能為您提供幫助。如果還有其他足部健康問題,請隨時詢問。',
            }
            
            # 尋找最匹配的回應
            response_text = None  # 初始化回應文字為 None
            for key, response in simple_responses.items():  # 遍歷問候語字典
                if key in question.lower():  # 如果問候語關鍵字出現在用戶問題中（不區分大小寫）
                    response_text = response  # 設定對應的回應
                    break  # 找到就停止搜尋
            
            if not response_text:  # 如果沒有找到匹配的回應
                response_text = '您好!我是 FootAnalyzer 的 AI 客服助手,專門幫助您解答足部健康相關問題。有什麼需要幫助的嗎?'  # 使用預設回應
            
            return {  # 回傳結果
                "answer": response_text,  # 回答內容
                "strategy": "simple_greeting",  # 標記為簡單問候
                "relevance_score": 1.0  # 相關性設為最高
            }
        
        # 計算相關性分數
        relevance_score = calculate_relevance_score(question, vector_store, embeddings, k)  # 計算問題與知識庫的相關性
        
        # 格式化對話歷史
        formatted_history = format_conversation_history(conversation_history or [])  # 將對話歷史格式化為字串      
        print(f"相關性分數: {relevance_score:.3f}")  # 印出相關性分數（保留3位小數）

        # 結合對話歷史再進行搜索
        if conversation_history and len(conversation_history) > 0:  # 如果有對話歷史
            # 存取前一次用戶提問
            last_exchange = conversation_history[-1]  # 取得最後一輪對話
            last_user_msg = last_exchange.get('user', '')  # 取得前一次用戶訊息

            # 組合前一次提問內容和本次問題做查詢 
            search_query = f"{last_user_msg} {question}"  # 將前後兩次問題合併
            print(f"結合對話歷史的檢索問題: {search_query}")  # 印出組合後的查詢問題
        else:  # 如果沒有對話歷史
            search_query = question  # 直接使用當前問題

        # 使用改進的查詢搜索
        docs = vector_store.similarity_search(search_query, k=k)  # 在向量資料庫中搜尋最相關的 k 個文件
        context = "\n".join([doc.page_content for doc in docs])  # 將搜尋到的文件內容合併為一個字串
        
        prompt = prompt_template.format(  # 使用模板格式化提示詞
            context=context,  # 填入知識庫內容
            conversation_history=formatted_history,  # 填入對話歷史
            question=question  # 填入用戶問題
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])  # 呼叫LLM生成回答
        
        return {  # 回傳結果
            "answer": response.content,  # LLM生成的回答內容
            "strategy": "knowledge_base",  # 標記為使用知識庫回答
            "relevance_score": relevance_score  # 相關性分數
        }
        
    except Exception as e:  # 如果發生任何錯誤
        return {  # 回傳錯誤訊息
            "answer": f"很抱歉,處理您的問題時發生錯誤:{str(e)}",  # 顯示錯誤內容
            "strategy": "error",  # 標記為錯誤
            "relevance_score": 0.0  # 相關性設為0
        }

def is_system_capability_question(question):
    """
    判斷是否為詢問系統功能的問題
    
    Args:
        question (str): 用戶問題
        
    Returns:
        bool: 是否為系統功能查詢
    """
    question_lower = question.lower().strip()  
    
    # 詢問系統功能的關鍵詞
    capability_keywords = [  # 定義系統功能查詢的關鍵字列表
        '能做什麼', '可以做什麼', '有什麼功能', '功能有哪些',
        '能幫我什麼', '可以幫我什麼', '會什麼', '做些什麼',
        '能夠做', '可以提供', '提供什麼', '服務什麼',
        '你是誰', '幹嘛的', '做什麼的'
    ]
    
    for keyword in capability_keywords:  # 遍歷每個關鍵字
        if keyword in question_lower:  # 如果關鍵字出現在問題中
            return True  # 回傳 True（是系統功能查詢）
    
    return False  # 如果沒有匹配任何關鍵字，回傳 False

def is_simple_greeting_or_social(question):
    """
    判斷是否為簡單的問候語或社交用語
    
    Args:
        question (str): 用戶問題
        
    Returns:
        bool: 是否為簡單問候語
    """
    question_stripped = question.strip()  # 去除問題前後的空白
    question_lower = question_stripped.lower()  # 將問題轉為小寫
    
    # 常見問候語和社交用語
    greetings = [  # 定義問候語列表
        '你好', '妳好', '您好', 'hello', 'hi', '嗨', '哈囉',
        '早安', '午安', '晚安',
    ]
    
    # 完全匹配檢查（包含標點符號的情況）
    for greeting in greetings:
        # 精確匹配：整個句子就是這個問候語（可能帶標點）
        if question_lower == greeting or \
           question_lower == greeting + '!' or \
           question_lower == greeting + '!' or \
           question_lower == greeting + '。' or \
           question_lower == greeting + '?' or \
           question_lower == greeting + '?':
            return True
    
    # 檢查是否為很短的簡單問題（3個字以內且完全匹配問候語）
    if len(question_stripped) <= 3 and question_lower in greetings:  # 如果長度 ≤3 且在問候語列表中
        return True  # 回傳 True（是問候語）
        
    return False  # 如果都不匹配，回傳 False

