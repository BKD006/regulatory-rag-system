from collections import Counter

class DocumentCleaner:

    def clean(self, content: str, docling_doc=None):

        if docling_doc is None:
            return content, {"layout_cleaning": False}

        cleaned = self._layout_clean_iter(docling_doc)

        return cleaned, {"layout_cleaning": True}

    # ---------------------------------------------------------

    def _layout_clean_iter(self, doc):

        items = list(doc.iterate_items())

        texts = [
            text_item.text.strip()
            for text_item, page_no in items
            if hasattr(text_item, "text") and text_item.text.strip()
        ]
        freq = Counter(texts)

        cleaned_texts = []

        total_pages = doc.num_pages()  # FIXED

        for text_item, page_no in items:

            if not hasattr(text_item, "text"):
                continue

            text = text_item.text.strip()
            if not text:
                continue

            # Remove repeated header/footer blocks
            if freq[text] > total_pages * 0.6:
                continue

            # Layout filtering
            if hasattr(text_item, "bbox"):

                page = doc.pages.get(page_no)

                if page and hasattr(page, "size"):

                    page_height = page.size[1]

                    y_top = text_item.bbox[1]
                    y_bottom = text_item.bbox[3]

                    if y_top < page_height * 0.15:
                        continue

                    if y_bottom > page_height * 0.88:
                        continue

            cleaned_texts.append(text)

        return "\n".join(cleaned_texts)

