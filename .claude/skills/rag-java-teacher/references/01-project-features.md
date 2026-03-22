# SuperMew Feature Map: Python → Spring AI Alibaba

## Document Ingestion

| SuperMew Python | Spring AI Alibaba Java | Notes |
|---|---|---|
| `document_loader.py` `load_pdf()` | `PagePdfDocumentReader` | Spring AI built-in, uses Apache PDFBox |
| `document_loader.py` `load_word()` | `TikaDocumentReader` | Spring AI built-in, Apache Tika |
| `document_loader.py` `load_excel()` | `TikaDocumentReader` or custom POI reader | Apache POI for xlsx |
| 3-level sliding window chunker (L1/L2/L3) | `TokenTextSplitter` + custom `HierarchicalDocumentSplitter` | **Custom wrapper required** |

## Embedding

| SuperMew Python | Spring AI Alibaba Java | Notes |
|---|---|---|
| `EmbeddingService.embed_dense()` | `EmbeddingModel.embed(text)` | DashScope / OpenAI adapter |
| jieba BM25 sparse vectors | Custom `BM25Indexer` (HanLP tokenizer) | **Key challenge — no built-in** |
| Batch embedding on upload | `EmbeddingModel.embedForResponse(List<String>)` | Batched API calls |

## Vector Storage

| SuperMew Python | Spring AI Alibaba Java | Notes |
|---|---|---|
| `MilvusManager.hybrid_retrieve()` | `MilvusVectorStore` + custom `HybridRetriever` | Spring AI has MilvusVectorStore, hybrid needs custom |
| `MilvusManager.dense_retrieve()` | `MilvusVectorStore.similaritySearch()` | Built-in |
| Dense + Sparse + RRF fusion | `AnnSearchRequest` (dense) + custom BM25 + `RRFMerger` | Milvus 2.4+ supports native hybrid |
| `ParentChunkStore` (JSON) | `Map<String, Document>` + Jackson JSON file | Custom, ~30 lines |
| chunk_level filter_expr | `SearchRequest.withFilterExpression("chunk_level == 3")` | Spring AI `FilterExpressionBuilder` |

## RAG Pipeline

| SuperMew Python | Spring AI Alibaba Java | Notes |
|---|---|---|
| LangGraph `StateGraph` (RAGState) | Custom `@Service` orchestrator or Spring AI `@Advisor` chain | Custom orchestration |
| `grade_documents` (binary relevance) | `GradingService` with structured output | `ChatModel` + response parser |
| `rewrite_question` (step-back/HyDE/complex) | `QueryRewriter` using `ChatClient` | Custom service |
| `retrieve_initial` → `retrieve_expanded` | `RetrieverService.retrieve()` → `RetrieverService.retrieveExpanded()` | Custom |
| `search_knowledge_base` @tool | `@Tool`-annotated method on Spring `@Bean` | Spring AI function calling |

## Conversation & Streaming

| SuperMew Python | Spring AI Alibaba Java | Notes |
|---|---|---|
| `ConversationStorage` (JSON-backed) | `InMemoryChatMemory` or custom `JsonFileChatMemory` | Spring AI ChatMemory interface |
| summary compression (50+ messages) | Custom `SummaryMemoryAdvisor` | Override `MessageChatMemoryAdvisor` |
| `chat_with_agent_stream` SSE | `ChatClient.stream()` + `SseEmitter` or `Flux<ServerSentEvent>` | Spring WebFlux |
| Cross-thread RAG step emission | Spring WebFlux `Sinks.Many<>` | Reactor native, cleaner than asyncio hack |
| RAG step types: content/rag_step/trace | Custom `SseEvent` sealed interface | Implement per SSE event type |

## API Layer

| SuperMew Python | Spring AI Alibaba Java | Notes |
|---|---|---|
| FastAPI `/chat` | Spring MVC `@PostMapping("/chat")` | Standard REST |
| FastAPI `/chat/stream` SSE | `@GetMapping(produces = TEXT_EVENT_STREAM_VALUE)` | `MediaType.TEXT_EVENT_STREAM_VALUE` |
| FastAPI `/documents/upload` | `@PostMapping("/documents/upload")` + `MultipartFile` | Standard multipart |
| FastAPI `/sessions/*` | `@RestController("/sessions")` | Standard REST |
| Pydantic schemas | Spring `@Valid` + record/class DTOs | Jackson serialization |

## Frontend

| SuperMew Python | Spring AI Alibaba Java | Notes |
|---|---|---|
| Vue 3 CDN SPA (`index.html`) | Keep as-is, or add Thymeleaf wrapper | Reusable frontend |
| SSE `ReadableStream` parsing | Same frontend code works | Protocol unchanged |
| Thinking state machine (Idle→Thinking→RAG→Stream) | Same frontend code works | No change needed |

## Key Differences to Highlight on Resume

1. **Type safety**: Java + Spring DI vs Python duck typing
2. **Enterprise patterns**: Spring Bean lifecycle, `@Advisor` chain vs LangChain middleware
3. **Reactive streaming**: Spring WebFlux `Flux<>` vs asyncio Queue hack
4. **Milvus native hybrid**: Java Milvus SDK `AnnSearchRequest` with `RRFRanker` vs Python pymilvus
5. **BM25**: Had to implement from scratch (no LangChain BM25Retriever equivalent in Spring AI)
