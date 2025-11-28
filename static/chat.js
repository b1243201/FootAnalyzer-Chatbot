const chatDiv = document.getElementById('chat');
const inputBox = document.getElementById('input');
const sendBtn = document.getElementById('send');

let isProcessing = false;

// Markdown 轉 HTML
function parseMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>')
    .replace(/(\d+\.\s)/g, '<br>$1')
    .replace(/(-\s)/g, '<br>• ');
}

function appendMessage(text, cls, isSystem = false) {
  const messageContainer = document.createElement('div');
  messageContainer.className = 'message-container';
  
  const p = document.createElement('p');
  
  if (cls === 'bot' || isSystem) {
    p.innerHTML = parseMarkdown(text);
  } else {
    p.textContent = text;
  }
  
  p.className = `msg ${cls}`;
  if (isSystem) {
    p.className += ' system';
  }
  
  messageContainer.appendChild(p);
  chatDiv.appendChild(messageContainer);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

// 移除「思考中」訊息
function removeThinkingMessage() {
  const lastChild = chatDiv.lastChild;
  if (lastChild?.querySelector?.('.msg')?.textContent.includes('思考中')) {
    chatDiv.removeChild(lastChild);
  }
}

async function sendMessage() {
  if (isProcessing) return;
  
  const question = inputBox.value.trim();
  if (!question) return;
  
  isProcessing = true;
  sendBtn.disabled = true;
  
  appendMessage(`你：${question}`, 'user');
  inputBox.value = '';
  appendMessage('…思考中…', 'bot');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    
    removeThinkingMessage();
    
    if (data.answer) {
      appendMessage(`Bot：${data.answer}`, 'bot');
    } else {
      appendMessage(`Bot：${data.error || '發生未知錯誤'}`, 'bot');
    }
    
  } catch (e) {
    removeThinkingMessage();
    appendMessage('Bot：連線錯誤，請稍後再試', 'bot');
  } finally {
    isProcessing = false;
    sendBtn.disabled = false;
  }
}

// 綁定事件
sendBtn.onclick = sendMessage;

inputBox.addEventListener('keypress', function(e) {
  if (e.key === 'Enter' && !isProcessing) {
    sendMessage();
  }
});

// 檢查系統狀態
async function checkSystemStatus() {
  try {
    const res = await fetch('/api/system_status');
    const data = await res.json();
    
    if (!data.system_ready) {
      appendMessage('系統：⚠️ 系統初始化未完成，部分功能可能無法正常使用', 'bot', true);
    }
  } catch (e) {
    console.error('檢查系統狀態失敗:', e);
  }
}

// 頁面載入初始化
document.addEventListener('DOMContentLoaded', function() {
  checkSystemStatus();
  setTimeout(() => {
    appendMessage('系統：歡迎使用 FootAnalyzer 客服系統！', 'bot', true);
  }, 500);
});