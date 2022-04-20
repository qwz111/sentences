from sentence_transformers import SentenceTransformer
import scipy.spatial

embedder = SentenceTransformer('test_output')  # 里面填写模型路径
corpus = [
  '做手术',
  '关节置换术',
  '我爱北京天安门',
  '哪个城市是中国首都',
  '还有多久才到终点站',
  '要走多久才到终点站',
  '你在干什么',
  '你在干啥子',
  '你在做什么',
  '你好啊',
  '我喜欢吃香蕉'
]
corpus_embeddings = embedder.encode(corpus)
# 待查询的句子
queries = ['目的地离这还有多远','干啥呢']
query_embeddings = embedder.encode(queries)
# 对于每个句子，使用余弦相似度查询最接近的n个句子
closest_n = 2
for query, query_embedding in zip(queries, query_embeddings):
  distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
  # 按照距离逆序
  results = zip(range(len(distances)), distances)
  results = sorted(results, key=lambda x: x[1])
  print("======================")
  print("Query:", query)
  print("Result:Top 5 most similar sentences in corpus:")
  for idx, distance in results[0:closest_n]:
    print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))
