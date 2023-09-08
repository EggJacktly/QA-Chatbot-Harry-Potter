from langchain import PromptTemplate

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''

prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Answer in the same language the question was asked.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}
Answer:"""


CUSTOM_PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ["context", "question"]
)


