import logging
import time
from typing import Dict, List, Tuple

from classification.product_area import infer_company, infer_product_area_from_ticket
from classification.request_type import classify_request_type
from classification.risk_detection import risk_escalate
from config.settings import ESCALATE_RESPONSE, RuntimeConfig
from data_loader.corpus_loader import Document, load_corpus
from data_loader.ticket_loader import load_tickets
from generation.justification import build_justification
from generation.response_builder import build_response
from output.writer import write_output
from retrieval.bm25 import BM25Index
from retrieval.retriever import Retriever
from routing.decision_engine import decide_status


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


def run_pipeline(config: RuntimeConfig, logger: logging.Logger) -> None:
    start = time.time()

    docs = load_corpus(config.corpus_root)
    if not docs:
        logger.warning("No corpus documents found at %s", config.corpus_root)
    index = BM25Index(docs) if docs else None
    retriever = Retriever(index, config.min_overlap, config.top_k) if index else None
    if index:
        logger.info("Indexed %d corpus chunks from %s", len(docs), config.corpus_root)

    tickets = load_tickets(config.input_path)
    logger.info("Loaded %d tickets from %s", len(tickets), config.input_path)

    output_rows: List[Dict[str, str]] = []

    for idx, ticket in enumerate(tickets, start=1):
        company = infer_company(ticket.issue, ticket.subject, ticket.company)
        request_type = classify_request_type(ticket.issue, ticket.subject)
        query = f"{ticket.subject} {ticket.issue}".strip()

        results: List[Tuple[Document, float]] = []
        if retriever and query:
            results = retriever.search(query, company)

        top_score = results[0][1] if results else 0.0
        top_doc = results[0][0] if results else None

        _, risk_reason = risk_escalate(ticket.issue, ticket.subject)
        if (
            not risk_reason
            and company == "None"
            and top_score > 0
            and top_score < 15.0
        ):
            risk_reason = (
                "Ambiguous organization field with modest retrieval confidence"
            )

        product_area = top_doc.product_area if top_doc else ""
        if not product_area:
            product_area = infer_product_area_from_ticket(ticket.issue, ticket.subject, company)
        if company != "None":
            if product_area and company.lower() not in product_area.lower():
                product_area = f"{company} · {product_area}"
            elif not product_area:
                product_area = company

        status = decide_status(request_type, risk_reason, results, config.min_score)
        response = (
            build_response(query, results, request_type, config.max_sentences)
            if status == "replied"
            else ESCALATE_RESPONSE
        )

        justification = build_justification(
            status,
            request_type,
            risk_reason,
            top_score,
            config.min_score,
            bool(results),
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

        preview = ticket.issue.replace("\n", " ")[:72]
        logger.info(
            "[%d/%d] status=%s company=%s risk=%s score=%.2f doc=%s issue=%s",
            idx,
            len(tickets),
            status,
            company,
            risk_reason or "-",
            top_score,
            top_doc.doc_id if top_doc else "-",
            preview,
        )

    write_output(config.output_path, output_rows)
    elapsed = time.time() - start
    logger.info("Wrote %d rows to %s", len(output_rows), config.output_path)
    logger.info("Completed in %.2f seconds", elapsed)
