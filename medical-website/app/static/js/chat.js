let chatHistory = [];
let canSendMessage = true;


const chatContainer = document.getElementById('chat');
chatContainer.scrollTo(0, chatContainer.scrollHeight);


function sendMessage(event) {
  event.preventDefault();

  const messageInput = document.getElementById('messageInput');
  const message = messageInput.value.trim();

  if (message && canSendMessage) {
    chatHistory.push({ type: 'user', message });
    displayMessage('user', message);

    fetch('/submit-message', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `message=${encodeURIComponent(message)}&type=user`,
    })
      .then(response => response.json())
      .then(data => {
        const serverResponse = data.response;
        chatHistory.push({ type: 'server', message: serverResponse });
        displayMessage('server', serverResponse);
        canSendMessage = true;
      })
      .catch(error => console.error('Error sending message:', error));

    canSendMessage = false;
    messageInput.value = '';
  }
}

document.getElementById('formsend').addEventListener('submit', sendMessage);

document.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && canSendMessage) {
    sendMessage();
  }
});

const sendButton = document.getElementById('sendButton');
sendButton.addEventListener('click', () => {
  if (canSendMessage) {
    sendMessage();
  }
});

function displayMessage(type, message) {
  const messageDiv = document.createElement('div');
  messageDiv.className = type === 'user' ? 'user-message' : 'server-message';
  chatContainer.appendChild(messageDiv);
}
