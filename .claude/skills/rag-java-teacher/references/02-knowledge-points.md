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
- **Hierarchy**: L1 (1200 tokens) → L2 (600 tokens) → L3 (300 tokens, leaf)
- Only L3 goes into Milvus. L1/L2 stored in `ParentChunkStore`

### Auto-merge Logic
```
If ≥ threshold (e.g., 2) L3 siblings share the same L2 parent:
    Replace all those L3 chunks with their L2 parent
    Repeat: if ≥ threshold L2 siblings share L1 parent → replace with L1
```
Result: short chunk retrieval precision + longer context when multiple chunks agree

### Java: HierarchicalDocumentSplitter
```java
public class HierarchicalDocumentSplitter {
    // L1: 1200 tokens, overlap 200
    // L2: 600 tokens, overlap 100
    // L3: 300 tokens, overlap 50
    public HierarchyResult split(Document source) {
        String sourceId = UUID.randomUUID().toString();
        List<Document> l1Chunks = splitLevel(source.getContent(), 1200, 200, sourceId, 1);
        List<Document> l2Chunks = new ArrayList<>();
        List<Document> l3Chunks = new ArrayList<>();

        for (Document l1 : l1Chunks) {
            List<Document> l2s = splitLevel(l1.getContent(), 600, 100, l1.getId(), 2);
            l2Chunks.addAll(l2s);
            for (Document l2 : l2s) {
                List<Document> l3s = splitLevel(l2.getContent(), 300, 50, l2.getId(), 3);
                l3Chunks.addAll(l3s);
            }
        }
        return new HierarchyResult(l1Chunks, l2Chunks, l3Chunks);
    }
}
```

### Java: Auto-merge
```java
public List<Document> mergeToParent(List<Document> leafChunks,
        ParentChunkStore store, int threshold) {
    // Group by parent_id
    Map<String, List<Document>> byParent = leafChunks.stream()
        .collect(Collectors.groupingBy(d -> d.getMetadata().get("parent_id").toString()));

    List<Document> result = new ArrayList<>();
    for (var entry : byParent.entrySet()) {
        String parentId = entry.getKey();
        List<Document> siblings = entry.getValue();
        Optional<Document> parent = store.get(parentId);
        if (siblings.size() >= threshold && parent.isPresent()) {
            result.add(parent.get());  // replace with parent
        } else {
            result.addAll(siblings);   // keep original leaves
        }
    }
    // deduplicate by id
    return result.stream().distinct().toList();
}
```

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
