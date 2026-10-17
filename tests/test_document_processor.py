
import unittest

from modules.document_processor import (
    process_document,
    split_into_chunks,
    DocumentProcessingError,
)


class TestDocumentProcessor(unittest.TestCase):

    def test_txt_processing(self):
        text = (
            "Generative AI creates content. "
            "RAG retrieves relevant context."
        )
        chunks = process_document(
            text.encode("utf-8"),
            "notes.txt",
            chunk_size=30,
            overlap=5,
        )

        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["metadata"]["document"], "notes.txt")
        self.assertIsNone(chunks[0]["metadata"]["page"])
        self.assertTrue(all(c["text"] for c in chunks))

    def test_empty_file(self):
        with self.assertRaises(DocumentProcessingError):
            process_document(b"", "empty.txt")

    def test_unsupported_file(self):
        with self.assertRaises(DocumentProcessingError):
            process_document(b"some data", "notes.docx")

    def test_corrupted_pdf(self):
        with self.assertRaises(DocumentProcessingError):
            process_document(b"not a PDF", "broken.pdf")

    def test_invalid_chunk_settings(self):
        with self.assertRaises(ValueError):
            split_into_chunks("Some text", chunk_size=10, overlap=10)

    def test_different_contents_have_different_document_ids(self):
    	content_a = b"Cell biology studies cells."
    	content_b = b"Genetics studies heredity."

    	result_a = process_document(content_a, "notes.txt")
    	result_b = process_document(content_b, "notes.txt")

    	self.assertNotEqual(
        	result_a[0]["metadata"]["document_id"],
        	result_b[0]["metadata"]["document_id"],
    )

    def test_chunk_words_are_not_split(self):
        text = "The cell structural organization is important."

        chunks = split_into_chunks(
            text,
            chunk_size=20,
            overlap=5,
        )

        self.assertTrue(chunks)

        # Every chunk must contain complete words from the source.
        source_words = text.split()
        for chunk in chunks:
            for word in chunk.split():
                self.assertIn(word, source_words)

        # The word "structural" must not be broken into fragments.
        self.assertTrue(
            any("structural" in chunk.split() for chunk in chunks)
        )



if __name__ == "__main__":
    unittest.main(verbosity=2)
