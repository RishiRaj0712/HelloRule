import Message from './Message'
import TypingIndicator from './TypingIndicator'
import './MessageList.css'

export default function MessageList({ messages, loading }) {
  return (
    <div className="message-list">
      <div className="message-list-inner">
        {messages.map((msg, i) => (
          <Message
            key={msg.id}
            message={msg}
            isLast={i === messages.length - 1}
          />
        ))}
        {loading && <TypingIndicator />}
      </div>
    </div>
  )
}