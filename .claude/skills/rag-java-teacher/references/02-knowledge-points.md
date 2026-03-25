# Core RAG Knowledge Points

## 1. 向量检索基础 (Dense Search)

### Theory
- **Embedding**: text → high-dimensional float vector (e.g., 1536-dim)
- **Similarity metrics**: Cosine similarity (normalized dot product), Inner Product, L2 distance
- **Why cosine**: direction matters more than magnitude; same meaning = similar direction regardless of length
- Cosine sim = `dot(a,b) / (|a| * |b|)`, range [-1, 1]

### Model Selection
- DashScope `text-embedding-v3`: 1024-dim, good for Chinese
- OpenAI `text-embedding-3-small`: 1536-dim
- Rule: embedding model must match at index and query time

### Java (Spring AI)
```java
@Autowired
EmbeddingModel embeddingModel;

// Single text
float[] vector = embeddingModel.embed("什么是RAG？");

// Batch (preferred for uploads)
EmbeddingResponse response = embeddingModel.embedForResponse(
    List.of("chunk 1 text", "chunk 2 text", "chunk 3 text")
);
List<float[]> vectors = response.getResults().stream()
    .map(r -> r.getOutput())
    .toList();
```

---

## 2. BM25 稀疏检索 (Sparse Search)

### Theory: TF-IDF vs BM25
- **TF-IDF**: `tf(t,d) * idf(t)` — term frequency × inverse document frequency
- **BM25 improvement**: TF saturation + document length normalization
  - `BM25(t,d) = idf(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))`
  - `k1 = 1.5`: controls TF saturation (higher = more weight to raw freq)
  - `b = 0.75`: length normalization (0 = no normalization, 1 = full normalization)
  - `idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)`

### Chinese Tokenization
- SuperMew uses jieba (Python). Java equivalent: **HanLP** (production) or **jieba4j** (simpler)
- For Milvus sparse vectors: tokenize → compute BM25 scores → `SortedMap<Integer, Float>` (token_id → score)

### Java: Custom BM25Indexer
```java
public class BM25Indexer {
    private final double k1 = 1.5;
    private final double b = 0.75;
    private Map<String, Integer> termToId = new HashMap<>();
    private Map<Integer, Integer> docFreq = new HashMap<>();  // term_id -> df
    private List<Integer> docLengths = new ArrayList<>();
    private int totalDocs = 0;
    private double avgDocLength = 0;

    public void fit(List<List<String>> tokenizedDocs) {
        totalDocs = tokenizedDocs.size();
        for (List<String> tokens : tokenizedDocs) {
            docLengths.add(tokens.size());
            Set<String> seen = new HashSet<>(tokens);
            for (String term : seen) {
                int id = termToId.computeIfAbsent(term, k -> termToId.size());
                docFreq.merge(id, 1, Integer::sum);
            }
        }
        avgDocLength = docLengths.stream().mapToInt(i->i).average().orElse(1.0);
    }

    // Returns sparse vector: token_id -> BM25 score
    public SortedMap<Integer, Float> encode(List<String> tokens) {
        Map<Integer, Integer> tf = new HashMap<>();
        for (String t : tokens) {
            if (termToId.containsKey(t)) {
                tf.merge(termToId.get(t), 1, Integer::sum);
            }
        }
        SortedMap<Integer, Float> sparse = new TreeMap<>();
        int docLen = tokens.size();
        for (var entry : tf.entrySet()) {
            int termId = entry.getKey();
            int termTf = entry.getValue();
            int df = docFreq.getOrDefault(termId, 1);
            double idf = Math.log((totalDocs - df + 0.5) / (df + 0.5) + 1);
            double score = idf * (termTf * (k1 + 1))
                / (termTf + k1 * (1 - b + b * docLen / avgDocLength));
            sparse.put(termId, (float) score);
        }
        return sparse;
    }
}
```

---

## 3. 混合检索 + RRF 融合 (Hybrid + RRF)

### Why Hybrid > Single-modal
- Dense: catches "同义词"、"概念相关" (semantic similarity)
- BM25: catches exact keywords, product names, codes, numbers
- Together: complementary coverage, higher recall

### RRF Formula
```
RRF_score(doc) = Σ  1 / (k + rank_i)
```
- `k = 60` (smoothing constant, prevents rank-1 from dominating)
- `rank_i` = position in ranked list from retriever i (1-indexed)
- **Key insight**: uses ranks, not raw scores → no normalization needed across different vector spaces

### Java: RRFMerger
```java
public class RRFMerger {
    private static final int K = 60;

    public List<Document> merge(
            List<Document> denseResults,
            List<Document> sparseResults,
            int topK) {

        Map<String, Double> scores = new HashMap<>();
        Map<String, Document> docMap = new HashMap<>();

        accumulateScores(denseResults, scores, docMap);
        accumulateScores(sparseResults, scores, docMap);

        return scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topK)
            .map(e -> docMap.get(e.getKey()))
            .toList();
    }

    private void accumulateScores(List<Document> ranked,
            Map<String, Double> scores, Map<String, Document> docMap) {
        for (int i = 0; i < ranked.size(); i++) {
            String id = ranked.get(i).getId();
            scores.merge(id, 1.0 / (K + i + 1), Double::sum);
            docMap.put(id, ranked.get(i));
        }
    }
}
```

### Milvus Native Hybrid (Milvus 2.4+)
```java
// Build dense search request
AnnSearchRequest denseReq = AnnSearchRequest.newBuilder()
    .withVectorFieldName("dense_vector")
    .withFloatVectors(List.of(queryVector))
    .withParams("{\"nprobe\": 10}")
    .withTopK(20)
    .build();

// Build sparse search request
AnnSearchRequest sparseReq = AnnSearchRequest.newBuilder()
    .withVectorFieldName("sparse_vector")
    .withSparseFloatVectors(List.of(sparseQuery))
    .withTopK(20)
    .build();

// RRF merge
RRFRanker ranker = RRFRanker.newBuilder().withK(60).build();

HybridSearchParam param = HybridSearchParam.newBuilder()
    .withCollectionName("embeddings_collection")
    .addSearchRequest(denseReq)
    .addSearchRequest(sparseReq)
    .withRanker(ranker)
    .withTopK(10)
    .build();

SearchResultsWrapper results = milvusClient.hybridSearch(param);
```

---

## 4. 层级分块 + Auto-merge (Hierarchical Chunking)

### Theory
- Short chunks → better retrieval precision (each chunk = focused topic)
- Long chunks → more context for generation (don't cut mid-thought)
- **Hierarchy**: L1 (chapter) → L2 (paragraph) → L3 (sentence group, leaf)
- Only L3 goes into Milvus. L1/L2 stored in `ParentChunkStore`

### Auto-merge Logic
```
Two-pass iterative merge:
Pass 1: L3→L2: if ≥ threshold L3 siblings share same L2 parent → replace with L2
Pass 2: L2→L1: if ≥ threshold L2 siblings share same L1 parent → replace with L1
Finally: deduplicate by chunk_id
```
Result: short chunk retrieval precision + longer context when multiple chunks agree

---

### ⚡ 设计演进（亮点，面试必讲）

**原始设计（照搬平铺分块经验）**
```
L1=1200字, L2=600字, L3=300字
overlap: L1=200, L2=100, L3=50 (overlap/size = 16.7%)
层级比例: 2x (每个L2只有2~3个L3子块)
```

**暴露的问题**

| 问题 | 根因 |
|---|---|
| threshold=2 几乎总触发 | 2x比例 → 每个L2只有2块L3 → 命中2/2=100%就升级，门槛形同虚设 |
| overlap制造假相关 | 相邻L3有50字重叠 → 向量高度相似 → 两块同时命中 → 虚假触发升级 |
| 升级后大量重复段落 | 两个相邻L2都升级，内容高度重叠，拼接后给LLM噪声文本 |
| overlap误用场景 | overlap是平铺分块的补丁（弥补边界截断），层级结构里父块已包含完整内容，overlap无意义 |

**对标参考（LlamaIndex官方设计）**
```
2048 → 512 → 128，overlap=20
4x层级比例 → 每个L2约4个L3子块
threshold=2 → 命中至少50%，语义上真的说明整段相关
```

**思考过程**

1. **比例决定threshold的含义**：2x比例时 threshold=2 = "命中所有子块"，毫无判别力；4x比例时 threshold=2 = "命中50%子块"，才是有意义的门槛
2. **overlap在层级结构里是错的**：层级关系本身保存了边界信息（父块包含完整子块），overlap只会制造重叠内容 → 假相关 → 雪球噪声
3. **L3边界截断怎么办**：不加overlap，改用**标点感知语义切分**——在目标切割点前~13字内找最近的句末标点（。！？；），优雅解决边界截断而不引入重复

**最终方案**

```
L1 = 1024字，overlap = 0（章节级，不参与向量检索）
L2 =  256字，overlap = 0（段落级，不参与向量检索）
L3 =   64字，overlap = 0，使用标点感知语义切分（leaf，存Milvus）
threshold = 2（含义：命中 ≥ 50% 子块才升级）
```

---

### Java: HierarchicalDocumentSplitter（最终方案）
```java
public class HierarchicalDocumentSplitter {
    // L1=1024, L2=256, L3=64, 全部 overlap=0, 4x 层级比例
    public HierarchyResult split(Document source) {
        String sourceId = UUID.randomUUID().toString();
        List<Document> l1Chunks = splitHard(source.getText(), 1024, sourceId, 1);
        List<Document> l2Chunks = new ArrayList<>();
        List<Document> l3Chunks = new ArrayList<>();

        for (Document l1 : l1Chunks) {
            List<Document> l2s = splitHard(l1.getText(), 256, l1.getId(), 2);
            l2Chunks.addAll(l2s);
            for (Document l2 : l2s) {
                // L3 用标点感知切分，目标64字，±13字浮动
                List<Document> l3s = splitSemantic(l2.getText(), 64, l2.getId(), 3);
                l3Chunks.addAll(l3s);
            }
        }
        return new HierarchyResult(l1Chunks, l2Chunks, l3Chunks);
    }

    // L3专用：往前找标点，避免关键词截断
    private List<String> splitSemantic(String text, int target) {
        Set<Character> sentenceEnd = Set.of('。', '！', '？', '；', '\n');
        Set<Character> pauseMark  = Set.of('，', '、', '：');
        List<String> chunks = new ArrayList<>();
        int start = 0;
        while (start < text.length()) {
            int end = Math.min(start + target, text.length());
            if (end == text.length()) { chunks.add(text.substring(start)); break; }
            int lookback = target / 5; // ~13字
            int cutAt = -1;
            for (int i = end; i >= end - lookback && cutAt == -1; i--)
                if (sentenceEnd.contains(text.charAt(i))) cutAt = i + 1;
            for (int i = end; i >= end - lookback && cutAt == -1; i--)
                if (pauseMark.contains(text.charAt(i))) cutAt = i + 1;
            if (cutAt == -1) cutAt = end;
            chunks.add(text.substring(start, cutAt));
            start = cutAt;
        }
        return chunks;
    }
}
```

### Java: Auto-merge（两轮迭代 + 去重）
```java
public List<Document> autoMerge(List<Document> leafChunks,
        ParentChunkStore store, int threshold) {
    List<Document> current = leafChunks;
    // 两轮：L3→L2, L2→L1
    for (int round = 0; round < 2; round++) {
        current = mergeSingleLevel(current, store, threshold);
    }
    // 按 chunk_id 去重
    return current.stream()
        .collect(Collectors.toMap(
            d -> d.getMetadata().get("chunk_id").toString(),
            d -> d, (a, b) -> a))
        .values().stream().toList();
}

private List<Document> mergeSingleLevel(List<Document> chunks,
        ParentChunkStore store, int threshold) {
    // 无 parent_id 的（L1）直接保留
    Map<Boolean, List<Document>> split = chunks.stream()
        .collect(Collectors.partitioningBy(
            d -> d.getMetadata().containsKey("parent_id")));

    List<Document> result = new ArrayList<>(split.get(false));
    split.get(true).stream()
        .collect(Collectors.groupingBy(d -> d.getMetadata().get("parent_id").toString()))
        .forEach((parentId, siblings) -> {
            Optional<Document> parent = store.get(parentId);
            if (siblings.size() >= threshold && parent.isPresent())
                result.add(parent.get());   // 升级
            else
                result.addAll(siblings);    // 保留原层
        });
    return result;
}

---

## 5. 相关性评分 + 查询改写 (Grading + Rewrite)

### Grading (Binary Relevance)
- SuperMew: LLM outputs `{"score": "yes"}` or `{"score": "no"}` via structured output
- If "no" → trigger query rewrite → re-retrieve

```java
// Spring AI structured output
record GradeResult(String score) {}  // "yes" or "no"

GradeResult grade = chatClient.prompt()
    .system("Grade document relevance. Output JSON: {\"score\": \"yes\"} or {\"score\": \"no\"}")
    .user("Question: " + question + "\n\nDocument: " + docContent)
    .call()
    .entity(GradeResult.class);

if ("no".equals(grade.score())) {
    question = queryRewriter.rewrite(question);
}
```

### HyDE (Hypothetical Document Embedding)
- Generate a fake answer document → embed it → use that vector for retrieval
- Intuition: answer vectors are closer to actual document vectors than question vectors

```java
String hypothetical = chatClient.prompt()
    .system("Write a document that would answer the following question.")
    .user(question)
    .call()
    .content();
float[] hydeVector = embeddingModel.embed(hypothetical);
// Use hydeVector for Milvus search instead of question vector
```

### Step-back Prompting
- Generate a broader, more abstract version of the question
- Retrieves more relevant background knowledge

```java
String stepBackQuery = chatClient.prompt()
    .system("Rewrite this question to be more general and abstract.")
    .user(question)
    .call()
    .content();
```

---

## 6. Rerank 精排 (Reranking)

### Bi-encoder vs Cross-encoder
- **Bi-encoder** (Milvus): encode query and doc separately → fast, scalable, less accurate
- **Cross-encoder** (Reranker): encode (query, doc) pair together → slow, accurate, used for top-K refinement
- Strategy: bi-encoder retrieves top-20, cross-encoder reranks to top-5

### Java: Jina Rerank API
```java
@Service
public class RerankService {
    private final RestClient restClient;

    public List<Document> rerank(String query, List<Document> docs, int topK) {
        var body = Map.of(
            "model", "jina-reranker-v2-base-multilingual",
            "query", query,
            "documents", docs.stream().map(Document::getContent).toList(),
            "top_n", topK
        );
        var response = restClient.post()
            .uri("https://api.jina.ai/v1/rerank")
            .header("Authorization", "Bearer " + jinaApiKey)
            .body(body)
            .retrieve()
            .body(RerankResponse.class);

        return response.results().stream()
            .sorted(Comparator.comparingDouble(r -> -r.relevanceScore()))
            .map(r -> docs.get(r.index()))
            .toList();
    }
}
```

---

## 7. 流式输出 + SSE (Streaming)

### Spring WebFlux + SSE
```java
@GetMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public Flux<ServerSentEvent<String>> streamChat(@RequestParam String message) {
    return chatClient.prompt()
        .user(message)
        .stream()
        .chatResponse()
        .map(response -> {
            String token = response.getResult().getOutput().getContent();
            return ServerSentEvent.<String>builder()
                .event("content")
                .data(token)
                .build();
        })
        .concatWith(Flux.just(
            ServerSentEvent.<String>builder().event("done").data("[DONE]").build()
        ));
}
```

### RAG Steps + Content in One Stream (like SuperMew)
```java
// Use Sinks for cross-thread emission (cleaner than SuperMew's asyncio hack)
Sinks.Many<ServerSentEvent<String>> sink = Sinks.many().unicast().onBackpressureBuffer();

// Emit RAG steps from any thread:
sink.tryEmitNext(ServerSentEvent.<String>builder()
    .event("rag_step")
    .data("{\"icon\":\"search\",\"label\":\"检索中...\",\"detail\":\"hybrid search\"}")
    .build());

// Content tokens flow normally
return sink.asFlux();
```

---

## 8. Milvus 向量数据库配置

### Collection Schema (Java Milvus SDK)
```java
CollectionSchemaParam schema = CollectionSchemaParam.newBuilder()
    .addFieldType(FieldType.newBuilder()
        .withName("id").withDataType(DataType.VarChar).withMaxLength(64).withPrimaryKey(true).build())
    .addFieldType(FieldType.newBuilder()
        .withName("content").withDataType(DataType.VarChar).withMaxLength(65535).build())
    .addFieldType(FieldType.newBuilder()
        .withName("dense_vector").withDataType(DataType.FloatVector).withDimension(1024).build())
    .addFieldType(FieldType.newBuilder()
        .withName("sparse_vector").withDataType(DataType.SparseFloatVector).build())
    .addFieldType(FieldType.newBuilder()
        .withName("chunk_level").withDataType(DataType.Int32).build())
    .addFieldType(FieldType.newBuilder()
        .withName("parent_id").withDataType(DataType.VarChar).withMaxLength(64).build())
    .build();
```

### Spring AI MilvusVectorStore Config
```yaml
spring:
  ai:
    vectorstore:
      milvus:
        client:
          host: 127.0.0.1
          port: 19530
        collection-name: embeddings_collection
        embedding-dimension: 1024
        index-type: IVF_FLAT
        metric-type: COSINE
```
