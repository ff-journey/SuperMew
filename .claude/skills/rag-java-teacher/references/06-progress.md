# 学习进度追踪

Update this file as you complete tasks. Tell Claude "我完成了[X]" to mark items done.

---

## Phase 1: Basic RAG (Day 1-2)

- [x] Spring Boot project created with Spring AI Alibaba starter
- [x] `pom.xml` dependencies added (spring-ai-alibaba, milvus, webflux)
- [x] `application.yml` configured (DashScope API key, Milvus host/port)
- [x] Milvus running (`docker compose up -d`)
- [x] Milvus Attu UI accessible at localhost:8000
- [x] PDF upload endpoint working (`/documents/upload`)
- [x] `/chat` endpoint with RAG tool working (ReactAgent + VectorStore tool)
- [x] `/chat/stream` SSE endpoint working
- [x] First successful Q&A demo (upload PDF → ask → get answer)

**Phase 1 done**: [x]

---

## Phase 2: Hybrid Search (Day 3-5)

- [x] Read `02-knowledge-points.md` BM25 + RRF sections
- [x] HanLP or jieba4j tokenizer added to project
- [x] `BM25Indexer` class implemented
- [x] `BM25Service.search()` working
- [x] `RRFMerger` implemented
- [x] `HybridRetriever` combining dense + BM25 working
- [x] Upload endpoint indexes to both Milvus and BM25Service
- [x] `HybridRagAdvisor` replacing `QuestionAnswerAdvisor`
- [x] chunk_level filter applied (only L3 leaves)
- [x] Compared dense-only vs hybrid recall — hybrid wins on keyword queries

**Phase 2 done**: [x]

---

## Phase 3: Advanced Features (Day 6-8)

- [x] Read `02-knowledge-points.md` Hierarchical Chunking section
- [x] `HierarchicalDocumentSplitter` implemented (L1/L2/L3)
- [x] Upload endpoint uses hierarchical splitter
- [x] L3 chunks stored in Milvus
- [x] L1/L2 chunks stored in `ParentChunkStore` (JSON-backed)
- [x] Auto-merge logic implemented in `HybridRetriever`
- [x] Auto-merge verified: long doc → multiple leaf hits → parent chunk returned
- [x] `RerankService` implemented (Jina reranker-v2-base-multilingual)
- [x] Rerank inserted between RRF and auto-merge
- [x] (Optional) `GradingService` implemented (binary relevance, parallelStream)
- [x] (Optional) `QueryRewriter` implemented (step-back + HyDE)
- [x] (Optional) Grade → rewrite → re-retrieve pipeline wired up

**Phase 3 done**: [x]

---

## Phase 4: Polish + Resume (Day 9-10)

- [ ] Session history with `MessageChatMemoryAdvisor` + `sessionId`
- [ ] `/sessions` endpoint (list / create / delete)
- [ ] Document list endpoint (`GET /documents`)
- [ ] Document delete endpoint (`DELETE /documents/{id}`)
- [ ] Frontend copied from SuperMew to `src/main/resources/static/`
- [ ] Frontend API URL updated to point to Java backend
- [ ] End-to-end demo run without errors
- [ ] Resume bullet points written (4-6 points)
- [ ] All 10 interview questions practiced aloud
- [ ] (Optional) 2-min demo video recorded

**Phase 4 done**: [ ]

---

## Overall Progress

- Phase 1: [ ] Not started / [ ] In progress / [ ] Done
- Phase 2: [ ] Not started / [ ] In progress / [ ] Done
- Phase 3: [ ] Not started / [ ] In progress / [ ] Done
- Phase 4: [ ] Not started / [ ] In progress / [ ] Done

**Demo ready**: [ ]

---

## Notes & Blockers

_Add notes about issues you ran into and how you solved them:_

-

---

## Interview Readiness

Rate yourself 1-5 on each question (1=can't answer, 5=can answer confidently):

| Q# | Topic | Self-score |
|---|---|---|
| Q1 | 混合检索为什么好 | - |
| Q2 | RRF算法原理 | - |
| Q3 | BM25 vs TF-IDF | - |
| Q4 | 层级分块 + Auto-merge | - |
| Q5 | HyDE | - |
| Q6 | Grading流程 | - |
| Q7 | Milvus混合检索实现 | - |
| Q8 | 只存L3的设计原因 | - |
| Q9 | 流式SSE实现 | - |
| Q10 | Spring AI vs LangChain | - |

Target: all 5s before the interview.
