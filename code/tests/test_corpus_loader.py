import sys
import tempfile
import unittest
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data_loader.corpus_loader import load_corpus


class CorpusLoaderTests(unittest.TestCase):
    def test_company_comes_from_path_inside_corpus(self) -> None:
        # The corpus lives under a parent folder named "claude"; Visa docs must still be Visa.
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "claude" / "data"
            visa_doc = corpus / "visa" / "support" / "dispute.md"
            visa_doc.parent.mkdir(parents=True)
            visa_doc.write_text("Contact your card issuer to dispute a charge.", encoding="utf-8")

            docs = load_corpus(corpus)

        self.assertTrue(docs)
        self.assertEqual({doc.company for doc in docs}, {"Visa"})


if __name__ == "__main__":
    unittest.main()
