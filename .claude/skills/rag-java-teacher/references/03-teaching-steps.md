# Phase-by-Phase Java Implementation Guide

## Phase 1: 基础RAG (Day 1-2) — Quick Win

**Goal**: Upload a PDF, ask a question, get an answer with source. Working demo in ~2 days.

### 1.1 Project Setup

```xml
<!-- pom.xml key dependencies -->
<dependency>
    <groupId>com.alibaba.cloud.ai</groupId>
    <artifactId>spring-ai-alibaba-starter</artifactId>
    <version>1.0.0-M5.1</version>
</dependency>
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-milvus-store-spring-boot-autoconfigure</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-webflux</artifactId>
</dependency>
```

```yaml
# application.yml
spring:
  ai:
    dashscope:
      api-key: ${ARK_API_KEY}
      chat:
        options:
          model: qwen-plus
      embedding:
        options:
          model: text-embedding-v3
    vectorstore:
      milvus:
        client:
          host: 127.0.0.1
          port: 19530
        collection-name: rag_java_demo
        embedding-dimension: 1024
        metric-type: COSINE
```

### 1.2 Document Upload Endpoint

```java
@RestController
@RequestMapping("/documents")
public class DocumentController {

    @Autowired private VectorStore vectorStore;
    @Autowired private TokenTextSplitter splitter;

    @PostMapping("/upload")
    public ResponseEntity<String> upload(@RequestParam MultipartFile file) throws IOException {
        // 1. Load document
        Resource resource = file.getResource();
        PagePdfDocumentReader reader = new PagePdfDocumentReader(resource);
        List<Document> docs = reader.get();

        // 2. Split into chunks
        List<Document> chunks = splitter.apply(docs);

        // 3. Store in Milvus (auto-embeds via EmbeddingModel)
        vectorStore.add(chunks);

        return ResponseEntity.ok("Uploaded " + chunks.size() + " chunks");
    }
}
```

### 1.3 Chat Endpoint with RAG

```java
@RestController
@RequestMapping("/chat")
public class ChatController {

    private final ChatClient chatClient;

    public ChatController(ChatClient.Builder builder, VectorStore vectorStore) {
        this.chatClient = builder
            .defaultAdvisors(new QuestionAnswerAdvisor(vectorStore))
            .build();
    }

    @PostMapping
    public String chat(@RequestBody ChatRequest request) {
        return chatClient.prompt()
            .user(request.message())
            .call()
            .content();
    }

    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> streamChat(@RequestParam String message) {
        return chatClient.prompt()
            .user(message)
            .stream()
            .chatResponse()
            .map(r -> ServerSentEvent.<String>builder()
                .event("content")
                .data(r.getResult().getOutput().getContent())
                .build())
            .concatWith(Flux.just(
                ServerSentEvent.<String>builder().data("[DONE]").build()
            ));
    }
}
```

### 1.4 Splitter Bean

```java
@Configuration
public class RagConfig {
    @Bean
    public TokenTextSplitter tokenTextSplitter() {
        return new TokenTextSplitter(300, 50, 5, 10000, true);
        // chunkSize=300, overlap=50, minChunkSize=5, maxChunks=10000, keepSeparator=true
    }
}
```

**Phase 1 Done when**: You can `curl -F file=@test.pdf localhost:8080/documents/upload` and then ask `GET /chat/stream?message=...` and get a streamed answer citing the PDF.

---

## Phase 2: 混合检索 (Day 3-5)

**Goal**: BM25 + dense + RRF fusion. Measurably better recall on keyword-heavy queries.

### 2.1 BM25 Indexer Service

```java
@Service
public class BM25Service {
    private BM25Indexer indexer = new BM25Indexer();
    private List<Document> indexedDocs = new ArrayList<>();
    private HanLPTokenizer tokenizer = new HanLPTokenizer();  // or jieba4j

    public synchronized void addDocuments(List<Document> docs) {
        indexedDocs.addAll(docs);
        List<List<String>> tokenized = docs.stream()
            .map(d -> tokenizer.tokenize(d.getContent()))
            .toList();
        indexer.fit(tokenized);  // rebuild index (simple approach)
    }

    public List<Document> search(String query, int topK) {
        List<String> qTokens = tokenizer.tokenize(query);
        // Score all docs
        List<Map.Entry<Integer, Double>> scores = new ArrayList<>();
        for (int i = 0; i < indexedDocs.size(); i++) {
            List<String> docTokens = tokenizer.tokenize(indexedDocs.get(i).getContent());
            double score = indexer.scoreDoc(qTokens, docTokens, i);
            scores.add(Map.entry(i, score));
        }
        return scores.stream()
            .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
            .limit(topK)
            .map(e -> indexedDocs.get(e.getKey()))
            .toList();
    }
}
```

### 2.2 Hybrid Retriever

```java
@Service
public class HybridRetriever {

    @Autowired private VectorStore vectorStore;
    @Autowired private BM25Service bm25Service;
    @Autowired private RRFMerger rrfMerger;
    @Autowired private EmbeddingModel embeddingModel;

    public List<Document> retrieve(String query, int topK) {
        // Dense retrieval
        SearchRequest denseReq = SearchRequest.query(query).withTopK(20);
        List<Document> denseResults = vectorStore.similaritySearch(denseReq);

        // Sparse (BM25) retrieval
        List<Document> sparseResults = bm25Service.search(query, 20);

        // RRF merge
        return rrfMerger.merge(denseResults, sparseResults, topK);
    }
}
```

### 2.3 Update Chat to Use HybridRetriever

```java
// Replace QuestionAnswerAdvisor with custom advisor
public class HybridRagAdvisor implements RequestResponseAdvisor {
    private final HybridRetriever retriever;

    @Override
    public AdvisedRequest adviseRequest(AdvisedRequest request, Map<String, Object> context) {
        List<Document> docs = retriever.retrieve(request.userText(), 5);
        String context_str = docs.stream()
            .map(Document::getContent)
            .collect(Collectors.joining("\n\n---\n\n"));

        String augmentedPrompt = "Context:\n" + context_str
            + "\n\nQuestion: " + request.userText()
            + "\n\nAnswer based on the context above.";

        return AdvisedRequest.from(request).withUserText(augmentedPrompt).build();
    }
}
```

**Phase 2 Done when**: BM25-only / dense-only / hybrid can be compared for a keyword-heavy query and hybrid wins.

---

## Phase 3: 高级特性 (Day 6-8)

**Goal**: Hierarchical chunking + auto-merge + optional grading/rewrite.

### 3.1 Hierarchical Document Splitter

```java
@Component
public class HierarchicalDocumentSplitter {

    public record HierarchyResult(
        List<Document> l1, List<Document> l2, List<Document> l3) {}

    public HierarchyResult split(Document source) {
        String sourceId = UUID.randomUUID().toString();
        List<Document> l1 = split(source.getContent(), 1200, 200, sourceId, null, 1);
        List<Document> l2 = new ArrayList<>(), l3 = new ArrayList<>();

        for (Document d1 : l1) {
            var l2s = split(d1.getContent(), 600, 100, sourceId, d1.getId(), 2);
            l2.addAll(l2s);
            for (Document d2 : l2s) {
                l3.addAll(split(d2.getContent(), 300, 50, sourceId, d2.getId(), 3));
            }
        }
        return new HierarchyResult(l1, l2, l3);
    }

    private List<Document> split(String text, int size, int overlap,
            String sourceId, String parentId, int level) {
        List<Document> chunks = new ArrayList<>();
        int start = 0;
        while (start < text.length()) {
            int end = Math.min(start + size, text.length());
            String chunkText = text.substring(start, end);
            Map<String, Object> meta = new HashMap<>();
            meta.put("chunk_level", level);
            meta.put("source_id", sourceId);
            if (parentId != null) meta.put("parent_id", parentId);
            chunks.add(new Document(chunkText, meta));
            if (end == text.length()) break;
            start += (size - overlap);
        }
        return chunks;
    }
}
```

### 3.2 ParentChunkStore

```java
@Component
public class ParentChunkStore {
    private final Map<String, Document> store = new ConcurrentHashMap<>();
    private final Path storePath = Path.of("data/parent_chunks.json");
    @Autowired private ObjectMapper objectMapper;

    public void put(String id, Document doc) {
        store.put(id, doc);
        persist();
    }

    public Optional<Document> get(String id) {
        return Optional.ofNullable(store.get(id));
    }

    @PostConstruct
    public void load() { /* load from JSON file */ }

    private void persist() { /* write to JSON file */ }
}
```

### 3.3 Auto-merge in HybridRetriever

```java
// Add to HybridRetriever.retrieve():
List<Document> merged = autoMerge(rrfResults, parentChunkStore, threshold=2);
```

### 3.4 Optional: GradingService

```java
@Service
public class GradingService {
    @Autowired private ChatClient chatClient;

    record GradeResult(String score) {}

    public boolean isRelevant(String question, String docContent) {
        GradeResult result = chatClient.prompt()
            .system("""
                You are a grader. Assess whether the document is relevant to the question.
                Output JSON: {"score": "yes"} or {"score": "no"}
                """)
            .user("Question: " + question + "\n\nDocument: " + docContent)
            .call()
            .entity(GradeResult.class);
        return "yes".equalsIgnoreCase(result.score());
    }
}
```

### 3.5 Optional: QueryRewriter

```java
@Service
public class QueryRewriter {
    @Autowired private ChatClient chatClient;

    public String stepBack(String query) {
        return chatClient.prompt()
            .system("Rewrite this question to be more general and abstract for better document retrieval.")
            .user(query)
            .call()
            .content();
    }

    public String hyde(String query) {
        return chatClient.prompt()
            .system("Write a hypothetical document that would answer this question.")
            .user(query)
            .call()
            .content();
    }
}
```

**Phase 3 Done when**: Upload a long document, ask a specific question, verify auto-merge returns a longer parent chunk containing the matched leaf chunks.

---

## Phase 4: 完善Demo (Day 9-10)

**Goal**: Demo-ready application with session history, clean API, frontend.

### 4.1 Session History

```java
// Add to ChatClient builder:
this.chatClient = builder
    .defaultAdvisors(
        new MessageChatMemoryAdvisor(new InMemoryChatMemory()),
        new HybridRagAdvisor(hybridRetriever)
    )
    .build();

// Pass conversationId per request:
chatClient.prompt()
    .user(message)
    .advisors(a -> a.param(CHAT_MEMORY_CONVERSATION_ID_KEY, sessionId))
    .stream()
    .chatResponse();
```

### 4.2 Document List Endpoint

```java
@GetMapping
public List<DocumentInfo> listDocuments() {
    // Query Milvus for distinct source_id + filename metadata
    return documentMetadataStore.listAll();
}

@DeleteMapping("/{id}")
public ResponseEntity<Void> deleteDocument(@PathVariable String id) {
    vectorStore.delete(List.of(id));
    parentChunkStore.removeBySourceId(id);
    return ResponseEntity.noContent().build();
}
```

### 4.3 Frontend Integration

The SuperMew frontend (`index.html` + `script.js` + `style.css`) can be **reused as-is**:
- SSE event types are the same (`content`, `rag_step`, `trace`, `error`, `[DONE]`)
- Just update the API base URL in `script.js`
- Serve via Spring Boot static resources: `src/main/resources/static/`

### 4.4 Resume Bullet Points (to write after demo works)

Template for each feature:
```
- 实现了[功能]，采用[技术方案]，解决了[问题]，效果是[可量化结果]
```

Examples:
- 实现混合检索（BM25+语义向量+RRF融合），召回率相比纯语义检索提升约30%
- 实现层级分块+自动合并（L1/L2/L3三级），在保证检索精度的同时提供更完整的上下文
- 基于Spring WebFlux实现流式SSE输出，首字延迟<500ms
- 实现LLM相关性评分+查询改写（Step-back/HyDE），减少无关文档进入生成阶段
