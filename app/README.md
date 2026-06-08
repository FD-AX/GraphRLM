# JMLC App Runtime

Этот документ описывает текущий рабочий pipeline JMLC/RLM в порядке исполнения:
от `.docx` до Neo4j graph skeleton и последующего latent relation encoding.

Главный принцип runtime:

```text
Document режется на chunks.
Каждый chunk сразу пишется в Neo4j.
Gemma обрабатывает один chunk за раз.
RLMState передается от chunk к chunk как накопленный контекст.
Neo4j хранит прогресс, статусы и evidence graph.
```

## 0. Внешние сервисы

Перед ingestion должны быть доступны:

```text
Gemma 3 4B через vLLM OpenAI-compatible endpoint
Neo4j через Bolt
```

Gemma запускается из локально скачанных весов:

```powershell
.\infra\gemma\run_gemma.ps1
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

Ожидаемый endpoint:

```text
base_url: http://127.0.0.1:8000/v1
model: gemma-3-4b-it
api_key: local
```

Neo4j живет в Kubernetes:

```powershell
kubectl apply -f .\infra\k8s\namespace.yaml
kubectl apply -f .\infra\k8s\neo4j.yaml
kubectl -n jmlc port-forward svc/neo4j 7687:7687 7474:7474
```

Ожидаемые endpoints:

```text
bolt://localhost:7687
http://localhost:7474
username: neo4j
password: password
```

## 1. Главная точка входа для DOCX

Основной production-like вход сейчас:

```text
app/runtime/docx_graph_ingest_neo4j.py
```

Запуск:

```powershell
python -m app.runtime.docx_graph_ingest_neo4j `
  --path "034. Подъём в преисподнюю (ред)(ред).docx" `
  --document-id "podem_v_preispodnyuyu" `
  --max-chunk-tokens 700 `
  --rlm-mode llm `
  --write-mode resume `
  --chunk-timeout-seconds 180 `
  --continue-on-error
```

Если positional path удобнее:

```powershell
python -m app.runtime.docx_graph_ingest_neo4j "034. Подъём в преисподнюю (ред)(ред).docx" `
  --document-id "podem_v_preispodnyuyu" `
  --max-chunk-tokens 700 `
  --write-mode resume `
  --chunk-timeout-seconds 180 `
  --continue-on-error
```

`--max-chunks` можно использовать только для smoke-прогонов:

```powershell
python -m app.runtime.docx_graph_ingest_neo4j `
  --max-chunk-tokens 700 `
  --max-chunks 2 `
  --write-mode resume
```

## 2. Write Modes

Ingestion больше не чистит документ перед каждым запуском.

Есть три режима:

```text
resume
append
overwrite
```

`resume` - основной режим.

```text
если chunk DONE и content_hash совпадает -> skip
если chunk FAILED, PROCESSING или PENDING -> process again
если chunk missing -> create and process
если content_hash changed -> delete old chunk subtree and recreate chunk
```

`append` - тестовый режим.

```text
не делает cleanup;
не пропускает DONE chunks;
повторно прогоняет chunks и обновляет nodes через MERGE.
```

`overwrite` - опасный явный режим.

```text
удаляет весь graph subtree документа;
создает Document и все chunks заново.
```

## 3. Chunk-Level Transactional Flow

Pipeline теперь не работает как большой batch:

```text
cleanup document
extract all chunks
write all graph
done
```

Правильный порядок:

```text
read docx
build DocumentNode
chunk_text()

for each chunk:
    ensure Document exists
    upsert Chunk immediately
    mark Chunk.extraction_status = PROCESSING

    run ChunkGraphWorkflow
    run RLM update

    write graph patch
    write RLMTransition
    mark Chunk.extraction_status = DONE

    if failed:
        mark Chunk.extraction_status = FAILED
        save extraction_error
        continue or stop
```

Так Neo4j становится checkpoint-хранилищем. Если Gemma зависла или Python упал,
в базе остается:

```text
(:Document)-[:HAS_CHUNK]->(:Chunk)
Chunk.extraction_status = PROCESSING | FAILED | DONE
Chunk.index
Chunk.content_hash
Chunk.extraction_attempts
Chunk.extraction_error
```

## 4. Document Node

Модель:

```text
app/core/graph_models.py:DocumentNode
```

Поля:

```text
document_id
title
source_path
metadata
```

Writer:

```text
app/graph/writer.py:GraphWriter.write_document()
```

Neo4j materialization:

```cypher
(:Document {
  document_id,
  title,
  source_path,
  metadata,
  created_at,
  updated_at
})
```

`created_at` и `updated_at` выставляет Neo4j writer.

## 5. Chunking

Файл:

```text
app/core/chunking.py
```

Функция:

```python
chunk_text(
    text=text,
    source_document_id=document_id,
    max_chunk_tokens=max_chunk_tokens,
)
```

Для локальной Gemma 3 4B текущая рабочая рекомендация:

```text
max_chunk_tokens: 500-1000
стартовое значение: 700
```

Слишком крупные chunks снижают число вызовов Gemma, но extraction становится
грубым: модель начинает возвращать мало сущностей и теряет локальные relation
candidates. Слишком мелкие chunks повышают качество локальной extraction, но
увеличивают время всего документа.

## 6. Chunk Node

Модель:

```text
app/core/graph_models.py:ChunkNode
```

Поля:

```text
chunk_id
document_id
index
text
token_count
start_char
end_char
content_hash
extraction_status
extraction_started_at
extraction_finished_at
extraction_error
extraction_attempts
```

Статусы:

```text
PENDING
PROCESSING
DONE
FAILED
```

Writer:

```text
app/graph/writer.py
```

Методы:

```text
write_chunk()
get_chunk_by_index()
mark_chunk_processing()
mark_chunk_done()
mark_chunk_failed()
delete_chunk_subtree()
```

Neo4j materialization:

```cypher
(:Document)-[:HAS_CHUNK]->(:Chunk)
```

## 7. Resume Logic

Resume ключ:

```text
document_id + chunk.index + content_hash
```

Алгоритм:

```text
existing = get_chunk_by_index(document_id, chunk.index)

if existing and existing.content_hash == chunk.content_hash:
    if existing.extraction_status == DONE:
        skip
        hydrate RLMState from saved EntityState for this chunk
    else:
        process again

if existing and existing.content_hash != chunk.content_hash:
    delete_chunk_subtree(existing.chunk_id)
    write new Chunk
    process

if missing:
    write new Chunk
    process
```

Для `DONE` chunks script поднимает минимум контекста обратно в `RLMState` через
сохраненные `EntityState`, чтобы следующие chunks не шли совсем вслепую после
resume.

## 8. Per-Chunk Timeout

Timeout применяется к одному chunk, а не ко всему документу:

```text
--chunk-timeout-seconds 180
```

Если chunk не успел:

```text
Chunk.extraction_status = FAILED
Chunk.extraction_error = "chunk N failed: TimeoutError: ..."
```

Поведение после ошибки:

```text
--continue-on-error enabled  -> перейти к следующему chunk
--continue-on-error disabled -> остановить ingestion
```

Важно: timeout на клиенте может не мгновенно остановить backend generation во
vLLM. Но chunk уже становится независимой единицей работы, и следующий запуск
resume сможет продолжить с понятного места.

## 9. ChunkGraphWorkflow

Builder:

```text
app/agent_graph/chunk_builder.py
```

State:

```text
app/agent_graph/chunk_state.py
```

Schemas:

```text
app/agent_graph/chunk_schemas.py
```

Nodes:

```text
app/agent_graph/chunk_nodes.py
```

Схема:

```text
START
  -> extract_graph_with_llm
  -> validate_llm_extraction
  -> normalize_graph_patch
  -> END
```

Input:

```text
chunk
rlm_state
errors
```

Output:

```text
llm_extraction
graph_patch
errors
```

Chunk workflow получает предыдущий `RLMState` как контекст, но не мутирует его
напрямую. Он возвращает только локальный `LocalGraphPatch`.

## 10. Gemma Extraction

Функция:

```text
app/agent_graph/chunk_nodes.py:extract_graph_with_llm()
```

Adapter:

```text
app/llm/model_adapter.py:OpenAICompatibleModelAdapter
```

Prompt:

```text
app/llm/prompts.py:GRAPH_EXTRACTION_SYSTEM_PROMPT
```

Вызов:

```python
model_adapter.structured_call(
    system_prompt=GRAPH_EXTRACTION_SYSTEM_PROMPT,
    user_payload={
        "chunk": chunk.model_dump(),
        "previous_rlm_state": rlm_state.model_dump(),
        "task": "...",
    },
    output_schema=LLMGraphExtraction,
)
```

Gemma возвращает raw extraction:

```text
entities
mentions
claims
events
relations
notes
```

Главное правило:

```text
LLM возвращает имена и offsets.
LLM не пишет в Neo4j.
LLM не создает финальные IDs.
```

## 11. Validation

Функция:

```text
app/agent_graph/chunk_nodes.py:validate_llm_extraction()
```

Проверяет:

```text
mention spans
claim evidence spans
event evidence spans
relation evidence spans
```

Инвариант:

```python
0 <= start < end <= len(chunk.text)
```

Для mention также проверяется совпадение текста:

```python
chunk.text[start:end].strip() == mention.text.strip()
```

На текущем этапе validation собирает ошибки, но не останавливает workflow.
Следующий шаг - отдельная repair/fail ветка в LangGraph.

## 12. Normalization

Функция:

```text
app/agent_graph/chunk_nodes.py:normalize_graph_patch()
```

Задачи:

```text
canonical names -> stable entity_id
chunk-local offsets -> document-global offsets
mentions -> Entity
claims/events/relations -> EvidenceSpan
claims with >=2 entities -> RelationCandidate
explicit LLM relations -> RelationCandidate
```

Output:

```text
LocalGraphPatch
```

Содержит:

```text
chunk
entities
mentions
claims
events
relations
evidence_spans
relation_candidates
latent_relation_edges
```

## 13. Entity-First Rules

Сущности являются центром графа.

Правильно:

```text
Entity -> evidence spans -> relation candidates -> latent edges
```

Неправильно:

```text
chunk-only retrieval
relation-first triples без evidence
```

Местоимения не становятся `Entity`.

Правило:

```text
pronoun != entity
```

Если местоимение резолвится, его смысл прикрепляется к canonical Entity через
evidence metadata. Если не резолвится, оно может остаться unresolved reference,
но не материализуется как отдельная нода `Entity`.

## 14. RLM Update

После `LocalGraphPatch` вызывается RLM update.

LLM RLM path:

```text
app/agent_graph/document_nodes.py:update_rlm_state_with_llm()
```

Prompt:

```text
app/llm/prompts.py:RLM_UPDATE_SYSTEM_PROMPT
```

Schema:

```text
app/rlm/schemas.py:RLMTransitionExtraction
```

Input в Gemma:

```text
current_chunk
previous_rlm_state
graph_patch
fallback_transition
```

Output превращается в:

```text
RLMTransition
```

и применяется:

```text
app/rlm/merge.py:apply_rlm_transition()
```

Deterministic path:

```text
app/agent_graph/document_nodes.py:update_rlm_state_from_patch()
```

Используется если:

```text
--rlm-mode deterministic
```

или как fallback, если Gemma RLM update вернула пустой или плохой ответ.

## 15. RLMState Across Chunks

RLM проходит по всем chunks с накопленным контекстом.

Схема:

```text
chunk_0 + RLMState_0 -> RLMState_1
chunk_1 + RLMState_1 -> RLMState_2
chunk_2 + RLMState_2 -> RLMState_3
```

`RLMState` хранит:

```text
document_id
current_chunk_index
entities
open_hypotheses
recent_chunk_ids
recent_evidence_spans
relation_candidates
unresolved_references
```

Это нужно, чтобы следующий chunk видел:

```text
уже найденные сущности
недавние evidence spans
relation candidates
нерешенные ссылки
```

## 16. GraphWriter Materialization

Writer:

```text
app/graph/writer.py
```

`write_local_graph()` пишет:

```text
Document
Chunk
Entity
Mention
Claim
Event
Relation
EvidenceSpan
RelationCandidate
LatentRelation
```

`write_rlm_transition()` пишет:

```text
RLMTransition
EntityState
```

Основные связи:

```text
(:Document)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:HAS_MENTION]->(:Mention)
(:Mention)-[:REFERS_TO]->(:Entity)

(:EvidenceSpan)-[:IN_CHUNK]->(:Chunk)
(:EvidenceSpan)-[:RESOLVES_ENTITY]->(:Entity)

(:Entity)-[:RELATION_SOURCE]->(:RelationCandidate)
(:RelationCandidate)-[:RELATION_TARGET]->(:Entity)
(:RelationCandidate)-[:SUPPORTED_BY]->(:EvidenceSpan)

(:LatentRelation)-[:ENCODES]->(:RelationCandidate)
(:LatentRelation)-[:SUPPORTED_BY]->(:EvidenceSpan)
(:Entity)-[:LATENT_RELATION_SOURCE]->(:LatentRelation)
(:LatentRelation)-[:LATENT_RELATION_TARGET]->(:Entity)

(:RLMTransition)-[:TO_CHUNK]->(:Chunk)
(:EntityState)-[:STATE_OF]->(:Entity)
```

## 17. Проверка Chunk Status

В Neo4j Browser:

```cypher
MATCH (d:Document {document_id: "podem_v_preispodnyuyu"})-[:HAS_CHUNK]->(c:Chunk)
RETURN c.index AS index,
       c.chunk_id AS chunk_id,
       c.extraction_status AS status,
       c.extraction_attempts AS attempts,
       c.token_count AS tokens,
       c.extraction_error AS error
ORDER BY index
```

Сводка:

```cypher
MATCH (:Document {document_id: "podem_v_preispodnyuyu"})-[:HAS_CHUNK]->(c:Chunk)
RETURN c.extraction_status AS status, count(*) AS chunks
ORDER BY status
```

## 18. Просмотр Графа

Связи между сущностями через relation candidates:

```cypher
MATCH p=(source:Entity)-[:RELATION_SOURCE]->(rc:RelationCandidate)
  -[:RELATION_TARGET]->(target:Entity)
MATCH (rc)-[:SUPPORTED_BY]->(span:EvidenceSpan)
RETURN p, span
LIMIT 200
```

Evidence по конкретной сущности:

```cypher
MATCH (e:Entity {canonical_name: "Олег"})
MATCH (span:EvidenceSpan)-[:RESOLVES_ENTITY]->(e)
RETURN e, span
LIMIT 100
```

Chunks с mentions:

```cypher
MATCH p=(:Document {document_id: "podem_v_preispodnyuyu"})
  -[:HAS_CHUNK]->(:Chunk)-[:HAS_MENTION]->(:Mention)-[:REFERS_TO]->(:Entity)
RETURN p
LIMIT 300
```

## 19. Latent Relation Encoding

После ingestion запускается отдельный job:

```text
app/runtime/encode_relation_candidates_neo4j.py
```

Он берет:

```text
RelationCandidate + EvidenceSpan
```

кодирует через CPU-friendly BERT-like encoder:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

и создает:

```text
LatentRelation
```

Запуск:

```powershell
python -m app.runtime.encode_relation_candidates_neo4j `
  --document-id "podem_v_preispodnyuyu" `
  --limit 100 `
  --device cpu
```

Smoke без скачивания модели:

```powershell
python -m app.runtime.encode_relation_candidates_neo4j `
  --document-id "podem_v_preispodnyuyu" `
  --limit 100 `
  --model-name hashing-fallback `
  --allow-hashing-fallback
```

`hashing-fallback` нужен только для smoke. Нормальный путь - MiniLM или другая
BERT-like модель.

## 20. Где Остался IngestPipeline

Файл:

```text
app/runtime/ingest_pipeline.py
```

`IngestPipeline` остался как программный LangGraph wrapper для тестов и прямого
вызова из Python. Но для `.docx -> Neo4j` основной путь теперь:

```text
app/runtime/docx_graph_ingest_neo4j.py
```

Причина:

```text
DOCX ingestion требует chunk-level checkpoint/resume.
Один большой document workflow не должен быть единственным местом записи прогресса.
```

## 21. Полная Формула

```text
DOCX
  -> read_docx_text()
  -> DocumentNode
  -> chunk_text()
  -> for each chunk:
       write Chunk immediately
       mark PROCESSING
       ChunkGraphWorkflow
         -> Gemma extraction
         -> validate offsets
         -> LocalGraphPatch
       RLM update
         -> previous RLMState + graph_patch
         -> RLMTransition
         -> next RLMState
       write graph skeleton
       write RLMTransition
       mark DONE
       on error mark FAILED
  -> RelationCandidate encoding
  -> LatentRelation graph
```

Коротко:

```text
Gemma extracts graph skeleton.
RLM accumulates context across chunks.
Neo4j stores topology, evidence and ingestion checkpoints.
BERT-like encoder builds relation geometry.
Answers must return to EvidenceSpan.
```
