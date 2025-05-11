import ollama from 'ollama'

const response = await ollama.chat({
  model: 'qwen3:0.6b',
  messages: [{ role: 'user', content: `以下の文章を読んで、Bicepリンターとは何か説明してください。

---

(ここに記事の内容を貼った)

` }],
})
console.log(response.message.content)
