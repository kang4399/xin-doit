# -*- coding: utf-8 -*-
"""
RAG 课堂练习：BM25、Dense 向量检索、Hybrid/RRF、Elasticsearch 与 Qdrant。
来源：课堂 BM25 讲义；用小数据演示，中文注释便于复习和面试。

运行基础演示（不需安装第三方包、不连接数据库）：
    python RAG_BM25_ES_Qdrant_课堂练习.py

可选连接已启动的本地 ES / Qdrant（只做连通性检查，不写入/删除）：
    python RAG_BM25_ES_Qdrant_课堂练习.py --check-services

可选执行数据库练习（只操作带 classroom_demo_ 前缀的演示索引/集合）：
    python RAG_BM25_ES_Qdrant_课堂练习.py --demo-services

可选依赖：无。使用 Python 标准库。
请将本项目的 .env 放在运行目录，或设置环境变量 ES_PASSWORD 和 QDRANT_API_KEY。
注意：ES/Qdrant 演示分别以 3 维手工向量为例，不是真实 Embedding 模型结果。
真实 RAG 应用需要 Embedding 模型、切块、权限控制、数据同步以及大模型生成环节。
"""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import base64


# ===== 1. 文档切块（Chunk）：超简化示例 =====
def chunk_text(text: str, chunk_size: int = 35) -> list[str]:
    """按字符长度切块，仅作原理演示；生产环境一般按句子/token 分块并设置 overlap。"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# ===== 2. 词法检索 BM25（手写版，不需安装 rank-bm25） =====
def tok_en(text: str) -> list[str]:
    """简单英文分词；中文需 jieba/thulac 等分词器，这里不能直接套用。"""
    return text.lower().split()


class BM25Pure:
    """BM25 = IDF（词的稀有程度）× TF 饱和（匹配次数）+ 文档长度归一化。"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tokens = [tok_en(d) for d in docs]
        self.n = len(self.tokens)
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lengths) / self.n if self.n else 0.0
        self.tfs = [Counter(t) for t in self.tokens]
        self.k1, self.b = k1, b
        df = defaultdict(int)
        for tokens in self.tokens:
            for term in set(tokens):  # 一篇文档中多次出现，也只计一次 df
                df[term] += 1
        # 常见的、带平滑的 BM25 IDF 公式
        self.idf = {
            term: math.log(1 + (self.n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def search(self, query: str, topk: int = 5) -> list[tuple[int, float]]:
        result = []
        for i, tf in enumerate(self.tfs):
            score = 0.0
            for term in tok_en(query):
                f = tf.get(term, 0)
                if not f:
                    continue
                # k1 控制 TF 饱和，b 控制对长文档的长度归一化
                denominator = f + self.k1 * (
                    1 - self.b + self.b * self.lengths[i] / self.avgdl
                )
                score += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denominator
            result.append((i, score))
        return sorted(result, key=lambda item: item[1], reverse=True)[:topk]


# ===== 3. Dense 检索：向量相似度计算 =====
def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: list[float], b: list[float]) -> float:
    denominator = math.sqrt(dot(a, a)) * math.sqrt(dot(b, b))
    return dot(a, b) / denominator if denominator else 0.0


def normalize(a: list[float]) -> list[float]:
    norm = math.sqrt(dot(a, a))
    return [x / norm for x in a] if norm else a[:]


# ===== 4. Hybrid：用 RRF 将两路排名融合 =====
def rrf_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """RRF(d) = Σ 1/(k + rank)，rank 从 1 开始，只看排名，不直接相加原始分数。"""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


# ===== 5. 读取项目中的 .env（不打印密码/API Key） =====
def load_local_env(path: Path = Path('.env')) -> None:
    """轻量读取 KEY=VALUE；只补充不存在的环境变量，不依赖 python-dotenv。"""
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def request_json(method: str, url: str, payload=None, headers=None):
    """通用 HTTP 请求；与 curl 的 GET/PUT/POST/DELETE 对应。"""
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={'Content-Type': 'application/json', **(headers or {})}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read().decode('utf-8')
        return json.loads(body) if body.strip().startswith(('{', '[')) else body


def es_headers() -> dict[str, str]:
    """ES 的 Basic Auth 使用 elastic 用户；密码从 .env 获取。"""
    user = os.getenv('ES_USERNAME', 'elastic')
    password = os.getenv('ES_PASSWORD', '')
    raw = base64.b64encode(f'{user}:{password}'.encode()).decode()
    return {'Authorization': f'Basic {raw}'}


def qdrant_headers() -> dict[str, str]:
    """项目 Qdrant 启用了 API Key，需要在请求头传 api-key。"""
    key = os.getenv('QDRANT_API_KEY', '')
    return {'api-key': key} if key else {}


def check_services():
    """只查询 ES / Qdrant，不创建、删除任何索引/集合。"""
    es_url = os.getenv('ES_URL', 'http://127.0.0.1:9200').rstrip('/')
    q_url = os.getenv('QDRANT_URL', 'http://127.0.0.1:6333').rstrip('/')
    print('\n[ES] 索引列表：')
    print(request_json('GET', es_url + '/_cat/indices?format=json&h=index', headers=es_headers()))
    print('[Qdrant] Collections 列表：')
    print(request_json('GET', q_url + '/collections', headers=qdrant_headers()))


def demo_services():
    """真实数据库演示：使用独立演示名字，不删除已有的业务索引/集合。"""
    es_url = os.getenv('ES_URL', 'http://127.0.0.1:9200').rstrip('/')
    q_url = os.getenv('QDRANT_URL', 'http://127.0.0.1:6333').rstrip('/')
    es_name = 'classroom_demo_bm25'
    collection = 'classroom_demo_vectors'

    # 安全策略：先检查同名演示资源是否存在，避免覆盖旧演示内容
    existing_es = request_json('GET', es_url + '/_cat/indices?format=json&h=index', headers=es_headers())
    existing_q = request_json('GET', q_url + '/collections', headers=qdrant_headers())
    if any(row.get('index') == es_name for row in existing_es):
        raise RuntimeError(f'ES 索引 {es_name} 已存在，停止操作，避免覆盖')
    if any(row.get('name') == collection for row in existing_q['result']['collections']):
        raise RuntimeError(f'Qdrant 集合 {collection} 已存在，停止操作，避免覆盖')

    # ES：创建索引 -> 写入 JSON 文档 -> 按关键词搜索
    print('\n[ES] 创建索引：', request_json('PUT', f'{es_url}/{es_name}', headers=es_headers()))
    print('[ES] 写入文档：', request_json('PUT', f'{es_url}/{es_name}/_doc/1?refresh=true',
          {'title': 'RAG', 'content': 'elasticsearch supports lexical retrieval'}, headers=es_headers()))
    print('[ES] 搜索 lexical：', request_json('GET', f'{es_url}/{es_name}/_search',
          {'query': {'match': {'content': 'lexical'}}}, headers=es_headers())['hits']['hits'])

    # Qdrant：创建 collection（维度=3）-> Upsert 两个 point -> 向量检索
    print('\n[Qdrant] 创建 collection：', request_json('PUT', f'{q_url}/collections/{collection}',
          {'vectors': {'size': 3, 'distance': 'Cosine'}}, headers=qdrant_headers()))
    points = {'points': [
        {'id': 1, 'vector': [0.1, 0.2, 0.3], 'payload': {'text': 'hello', 'visibility': 'public'}},
        {'id': 2, 'vector': [0.0, 0.5, -0.1], 'payload': {'text': 'world', 'visibility': 'internal'}}
    ]}
    print('[Qdrant] Upsert：', request_json('PUT', f'{q_url}/collections/{collection}/points?wait=true',
          points, headers=qdrant_headers()))
    # Qdrant 1.15 使用 /points/query，query 可直接传向量
    print('[Qdrant] 查询：', request_json('POST', f'{q_url}/collections/{collection}/points/query',
          {'query': [0.09, 0.19, 0.31], 'limit': 2, 'with_payload': True}, headers=qdrant_headers()))
    print('\n练习完成！演示资源保留以便在数据库中查看：')
    print(f'ES 索引：{es_name}；Qdrant 集合：{collection}')
    print('本脚本不会自动 DELETE；确认不需要后再手动清理。')


def local_demo():
    print('=== 第一部分：RAG 入库：文档 -> Chunk（示意） ===')
    original = 'RAG系统先把文档切块，再生成Embedding向量，存入向量数据库，用户提问时先检索后生成。'
    print(chunk_text(original, chunk_size=20))

    print('\n=== 第二部分：BM25 关键词检索 ===')
    docs = [
        'bm25 is a lexical retrieval function used in information retrieval',
        'vector search uses embeddings to find semantically similar documents',
        'hybrid retrieval combines bm25 and dense vector retrieval',
        'rrf is a rank based fusion method',
        'fastapi is a modern web framework',
    ]
    bm = BM25Pure(docs)
    lexical = bm.search('bm25 lexical retrieval')
    for doc_id, score in lexical:
        print(f'文档{doc_id} BM25={score:.4f} | {docs[doc_id]}')

    print('\n=== 第三部分：Dense 向量检索原理（手工向量，不调用千问） ===')
    q = [0.1, 0.2, 0.3]
    vectors = [[0.11, 0.21, 0.31], [0.5, -0.2, 0.0], [0.0, 0.1, 0.2]]
    dense = sorted(enumerate(vectors), key=lambda x: cosine(q, x[1]), reverse=True)
    for doc_id, vec in dense:
        print(f'向量{doc_id} cosine={cosine(q, vec):.4f} | dot={dot(q, vec):.4f} | L2={l2(q, vec):.4f}')
    print('归一化后 dot≈cosine：', round(dot(normalize(q), normalize(vectors[0])), 6),
          round(cosine(q, vectors[0]), 6))

    print('\n=== 第四部分：Hybrid 检索：BM25 排名 + Dense 排名 -> RRF ===')
    # 为演示排名融合，这里使用同一组文档的两个示意排名；生产环境从 ES/Qdrant 得到排名
    bm25_ranking = [doc_id for doc_id, _ in lexical]
    dense_ranking = [2, 1, 0, 3, 4]  # 教学用示意排序
    print('BM25 排名：', bm25_ranking)
    print('Dense 排名：', dense_ranking)
    print('RRF 融合结果：', [(i, round(score, 6)) for i, score in rrf_fusion([bm25_ranking, dense_ranking])])
    print('Rerank：可在融合后交给更强的重排序模型精排（这里未调用外部模型）。')
    print('最后把相关 chunk 作为上下文交给大模型生成回答（这里未调用千问）。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG/BM25/ES/Qdrant 课堂可运行练习')
    parser.add_argument('--check-services', action='store_true', help='查询 ES/Qdrant 连通性（只读）')
    parser.add_argument('--demo-services', action='store_true', help='操作独立演示索引与集合（会写入）')
    args = parser.parse_args()
    load_local_env()
    local_demo()
    if args.check_services or args.demo_services:
        try:
            if args.check_services:
                check_services()
            if args.demo_services:
                demo_services()
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, RuntimeError) as exc:
            print('\n数据库操作失败：', exc)
            print('检查容器状态、.env 密码、端口；不要直接删除数据库数据卷。')
            raise SystemExit(1) from exc
