from transformers import pipeline

# 仅指定任务时，使用默认模型（不推荐）
pipe = pipeline(task="sentiment-analysis", model="shibing624/text2vec-base-chinese")
pipe("今天开始运行第一个程序")


classifier = pipeline(task="ner",model="ckiplab/bert-base-chinese-ner")

preds = classifier("我叫萨拉，我住在伦敦。")
preds = [
    {
        "entity": pred["entity"],
        "score": round(pred["score"], 4),
        "index": pred["index"],
        "word": pred["word"],
        "start": pred["start"],
        "end": pred["end"],
    }
    for pred in preds
]
print(*preds, sep="\n")


#**Question Answering**(问答)
question_answerer = pipeline(task="question-answering",model="timpal0l/mdeberta-v3-base-squad2")

preds = question_answerer(
    question="What is the name of the repository?",
    context="The name of the repository is huggingface/transformers",
)
print(
    f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
)

#**Summarization**(文本摘要）

summarizer = pipeline(task="summarization",
                      model="csebuetnlp/mT5_multilingual_XLSum",
                      min_length=8,
                      max_length=32,
)

summarizer(
    """
    In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, 
    replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. 
    For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. 
    On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. 
    In the former task our best model outperforms even all previously reported ensembles.
    """
)

#Audio classification
classifier = pipeline(task="audio-classification", model="facebook/mms-lid-126")

# 使用本地的音频文件做测试
preds = classifier("./data/audio/mlk.flac")
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
preds
#Automatic speech recognition（ASR）
# 使用 `model` 参数指定模型
transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
text = transcriber("data/audio/mlk.flac")
text

#Computer Vision 计算机视觉
from transformers import pipeline

classifier = pipeline(task="image-classification", model="OttoYu/Tree-HK")
preds = classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
print(*preds, sep="\n")

