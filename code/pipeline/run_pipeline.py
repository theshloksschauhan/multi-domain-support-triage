import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from classification.product_area import infer_company, infer_product_area_from_ticket
from classification.request_type import classify_request_type
from classification.risk_detection import risk_escalate
from config.settings import DISTRIBUTED_SEARCH_MODE, ESCALATE_RESPONSE, RuntimeConfig
from data_loader.corpus_loader import Document, load_corpus
from data_loader.ticket_loader import load_tickets
from generation.justification import build_justification
from generation.response_builder import build_response
from output.writer import write_output
from prediction.features import model_features, ticket_features
from prediction.model import load_model
from retrieval.backend import build_search_backend
from retrieval.distributed_retriever import RetrievalUnavailable


def _build_row(
    status: str,
    product_area: str,
    response: str,
    justification: str,
    request_type: str,
) -> Dict[str, str]:
    return {
        "status": status,
        "product_area": product_area,
        "response": response,
        "justification": justification,
        "request_type": request_type,
    }


def _append_diagnostics(path, record: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _source_label(top_doc: Optional[Document], corpus_root) -> str:
    if top_doc is None:
        return ""
    try:
        return top_doc.path.relative_to(corpus_root).as_posix()
    except ValueError:
        return top_doc.path.as_posix()


def run_pipeline(config: RuntimeConfig, logger: logging.Logger) -> None:
    start = time.time()
    cluster_error = ""
    docs: List[Document] = []

    if config.search_mode == DISTRIBUTED_SEARCH_MODE:
        try:
            retriever = build_search_backend(
                config.search_mode,
                [],
                config.min_overlap,
                config.top_k,
                shard_config=str(config.shard_config_path) if config.shard_config_path else None,
                timeout=config.shard_timeout_seconds,
                retries=config.shard_retries,
            )
        except RetrievalUnavailable as exc:
            retriever = None
            cluster_error = str(exc)
            logger.error("Distributed retrieval unavailable: %s", exc)
    else:
        docs = load_corpus(config.corpus_root)
        if not docs:
            logger.warning("No corpus documents found at %s", config.corpus_root)
        retriever = build_search_backend(
            config.search_mode,
            docs,
            config.min_overlap,
            config.top_k,
        )
        if retriever:
            logger.info("Indexed %d corpus chunks from %s", len(docs), config.corpus_root)

    tickets = load_tickets(config.input_path)
    logger.info("Loaded %d tickets from %s", len(tickets), config.input_path)
    decision_model = load_model()

    output_rows: List[Dict[str, str]] = []

    for idx, ticket in enumerate(tickets, start=1):
        company = infer_company(ticket.issue, ticket.subject, ticket.company)
        request_type = classify_request_type(ticket.issue, ticket.subject)
        query = f"{ticket.subject} {ticket.issue}".strip()
        retrieval_failure = cluster_error

        results: List[Tuple[Document, float]] = []
        if retriever and query and not retrieval_failure:
            try:
                results = retriever.search(query, company)
            except RetrievalUnavailable as exc:
                retrieval_failure = str(exc)
                results = []

        top_score = results[0][1] if results else 0.0
        top_doc = results[0][0] if results else None

        risk_hit, risk_reason = risk_escalate(ticket.issue, ticket.subject)
        features = ticket_features(
            ticket.issue,
            ticket.subject,
            ticket.company,
            query,
            top_doc,
            top_score,
            config.max_sentences,
        )

        product_area = top_doc.product_area if top_doc else ""
        if not product_area:
            product_area = infer_product_area_from_ticket(ticket.issue, ticket.subject, company)
        if company != "None":
            if product_area and company.lower() not in product_area.lower():
                product_area = f"{company} · {product_area}"
            elif not product_area:
                product_area = company

        evidence_ref = ""
        answer_mismatch = False
        unavailable = bool(retrieval_failure) and not risk_reason and request_type != "invalid"
        if unavailable:
            status = "escalated"
            response = ESCALATE_RESPONSE
        elif risk_hit:
            status = "escalated"
            response = ESCALATE_RESPONSE
        elif request_type == "invalid":
            status = "replied"
            response = build_response(query, results, request_type, config.max_sentences)
        else:
            status = decision_model.predict_status(model_features(features, config.min_score))
            if status == "replied":
                response = build_response(query, results, request_type, config.max_sentences)
            else:
                response = ESCALATE_RESPONSE
                if top_doc is not None and not features["action_match"]:
                    answer_mismatch = True
                elif (
                    features["company_ambiguous"]
                    and top_score > 0
                    and top_score < 15.0
                ):
                    risk_reason = (
                        "Ambiguous organization field with modest retrieval confidence"
                    )
            if (
                status == "replied"
                and config.search_mode == DISTRIBUTED_SEARCH_MODE
                and top_doc is not None
            ):
                evidence_ref = top_doc.doc_id

        justification = build_justification(
            status,
            request_type,
            risk_reason,
            top_score,
            config.min_score,
            bool(results),
            retrieval_unavailable=unavailable,
            evidence_ref=evidence_ref,
            answer_mismatch=answer_mismatch,
        )

        output_rows.append(
            _build_row(
                status,
                product_area,
                response,
                justification,
                request_type,
            )
        )

        if risk_reason:
            reason = "risk"
        elif retrieval_failure and status == "escalated":
            reason = "retrieval_failure"
        elif answer_mismatch:
            reason = "answer_mismatch"
        elif status == "escalated":
            reason = "evidence"
        else:
            reason = "replied"
        diagnostics = {}
        if retriever is not None:
            diagnostics = dict(getattr(retriever, "last_diagnostics", {}) or {})
        _append_diagnostics(
            config.diagnostics_path,
            {
                "query_id": diagnostics.get("query_id", ""),
                "ticket_index": idx,
                "status": status,
                "reason": reason,
                "doc_id": top_doc.doc_id if top_doc else "",
                "source": _source_label(top_doc, config.corpus_root),
                "score": top_score,
                "generation": diagnostics.get("generation", ""),
                "shards": diagnostics.get("shards", []),
                "candidate_count": diagnostics.get("candidate_count", 0),
                "retrieval_error": retrieval_failure,
            },
        )

        preview = ticket.issue.replace("\n", " ")[:72]
        logger.info(
            "[%d/%d] status=%s company=%s risk=%s score=%.2f doc=%s issue=%s",
            idx,
            len(tickets),
            status,
            company,
            risk_reason or ("retrieval_unavailable" if retrieval_failure else "-"),
            top_score,
            top_doc.doc_id if top_doc else "-",
            preview,
        )

    write_output(config.output_path, output_rows)
    elapsed = time.time() - start
    logger.info("Wrote %d rows to %s", len(output_rows), config.output_path)
    logger.info("Completed in %.2f seconds", elapsed)
