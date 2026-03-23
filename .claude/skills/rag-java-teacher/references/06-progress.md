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

- [ ] Read `02-knowledge-points.md` BM25 + RRF sections
- [ ] HanLP or jieba4j tokenizer added to project
- [ ] `BM25Indexer` class implemented
- [ ] `BM25Service.search()` working
- [ ] `RRFMerger` implemented
- [ ] `HybridRetriever` combining dense + BM25 working
- [ ] Upload endpoint indexes to both Milvus and BM25Service
- [ ] `HybridRagAdvisor` replacing `QuestionAnswerAdvisor`
- [ ] chunk_level filter applied (only L3 leaves)
- [ ] Compared dense-only vs hybrid recall — hybrid wins on keyword queries

**Phase 2 done**: [ ]

---

## Phase 3: Advanced Features (Day 6-8)

- [ ] Read `02-knowledge-points.md` Hierarchical Chunking section
- [ ] `HierarchicalDocumentSplitter` implemented (L1/L2/L3)
- [ ] Upload endpoint uses hierarchical splitter
- [ ] L3 chunks stored in Milvus
- [ ] L1/L2 chunks stored in `ParentChunkStore` (JSON-backed)
- [ ] Auto-merge logic implemented in `HybridRetriever`
- [ ] Auto-merge verified: long doc → multiple leaf hits → parent chunk returned
- [ ] (Optional) `GradingService` implemented (binary relevance)
- [ ] (Optional) `QueryRewriter` implemented (step-back + HyDE)
- [ ] (Optional) Grade → rewrite → re-retrieve pipeline wired up

**Phase 3 done**: [ ]

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
