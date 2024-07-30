import tiktoken
from typing import List
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


import re

default_encoder = "cl100k_base"

encoding = tiktoken.get_encoding(default_encoder)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=default_encoder, chunk_size=512, chunk_overlap=50
)

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2048,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)


def crop_by_tokens(text, max_tokens=1500):
    # Initialize the tokenizer
    tokenizer = encoding

    # Tokenize the input text
    tokens = tokenizer.encode(text)

    # Crop the tokens to the specified maximum number
    cropped_tokens = tokens[:max_tokens]

    # Decode the tokens back to string
    cropped_text = tokenizer.decode(cropped_tokens)

    return cropped_text


def crop_by_chars(text, max_chars=2048):
    return text[:max_chars]


def split_document_embedding(document_list):
    """Splits a document based on the given splitter, but retains the original text

    Returns a list of new documents
    """
    shortened_docs = []

    for document in document_list:
        original_document_text = document.page_content
        extended_document_text = crop_by_tokens(original_document_text)
        document.text = extended_document_text
        shortened_docs.append(document)

    return shortened_docs


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def remove_markdown_index_links(markdown_text):
    # Regex patterns
    list_item_link_pattern = re.compile(
        r"^\s*\*\s*\[[^\]]+\]\([^\)]+\)\s*$", re.MULTILINE
    )
    list_item_header_link_pattern = re.compile(
        r"^\s*\*\s*#+\s*\[[^\]]+\]\([^\)]+\)\s*$", re.MULTILINE
    )
    header_link_pattern = re.compile(r"^\s*#+\s*\[[^\]]+\]\([^\)]+\)\s*$", re.MULTILINE)

    # Remove matches
    cleaned_text = re.sub(list_item_header_link_pattern, "", markdown_text)
    cleaned_text = re.sub(list_item_link_pattern, "", cleaned_text)
    cleaned_text = re.sub(header_link_pattern, "", cleaned_text)

    # Removing extra newlines resulting from removals
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)
    cleaned_text = re.sub(
        r"^\s*\n", "", cleaned_text, flags=re.MULTILINE
    )  # Remove leading newlines

    return cleaned_text


def add_length_to_document(document):
    """Adds the number of tokens in a document to the document."""

    document.metadata["token_count"] = num_tokens_from_string(
        document.page_content, default_encoder
    )

    document.metadata["char_count"] = len(document.page_content)

    return document


def chunk_document_embedding_and_add_to_db(
    document_list,
    vectorstore,
    max_chars,
    embedding_model,
    batch_size=100,
    retry_count=5,
):
    """Chunks a document based on the given splitter, but retains the original text.

    Returns a list of new documents.
    """

    def process_chunk(chunk):
        chunk_full_page_content = [doc.page_content for doc in chunk]
        chunk_cropped_page_content = [
            crop_by_chars(doc.page_content, max_chars) for doc in chunk
        ]
        chunk_metadata = [doc.metadata for doc in chunk]
        chunk_embeddings = embedding_model.embed_documents(chunk_cropped_page_content)
        chunk_embedding_tuples = list(zip(chunk_full_page_content, chunk_embeddings))
        return vectorstore.add_embeddings(
            text_embeddings=chunk_embedding_tuples,
            metadatas=chunk_metadata,
            bulk_size=500,
        )

    chunks = [
        document_list[i : i + batch_size]
        for i in range(0, len(document_list), batch_size)
    ]

    for chunk in chunks:
        success = False
        attempts = 0
        while not success and attempts < retry_count:
            try:
                process_chunk(chunk)
                success = True
            except Exception as e:
                attempts += 1
                if attempts >= retry_count:
                    raise e
                print(f"Attempt {attempts} failed for chunk, retrying...")


def split_long_documents_in_list(documents, max_tokens, max_chars) -> List[str]:
    document_list = [add_length_to_document(doc) for doc in documents]

    list_of_too_long_docs = [
        doc for doc in document_list if doc.metadata["token_count"] > max_tokens
    ]

    list_of_short_docs = [
        doc for doc in document_list if doc.metadata["token_count"] <= max_tokens
    ]

    if len(list_of_too_long_docs) > 0:
        for document in list_of_too_long_docs:
            # remove markdown index links on all the content
            document.page_content = remove_markdown_index_links(document.page_content)

    split_long_document_list = text_splitter.split_documents(list_of_too_long_docs)

    all_docs = list_of_short_docs + split_long_document_list

    # for all documents, if the char count is too high, crop the text by characters
    for document in all_docs:
        if document.metadata["char_count"] > max_chars:
            document.page_content = crop_by_chars(document.page_content)

    return all_docs


def split_short_and_long_documents(
    documents,
    max_tokens,
    max_chars,
    cleaning_function_list=[remove_markdown_index_links],
):
    document_list = [add_length_to_document(doc) for doc in documents]

    list_of_too_long_docs = [
        doc for doc in document_list if doc.metadata["token_count"] > max_tokens
    ]

    list_of_short_docs = [
        doc for doc in document_list if doc.metadata["token_count"] <= max_tokens
    ]
    # Use a temporary list to hold documents that need to be moved
    documents_to_move = []

    for document in list_of_short_docs:
        if document.metadata["char_count"] > max_chars:
            documents_to_move.append(document)

    # Move documents to the long docs list
    for document in documents_to_move:
        list_of_too_long_docs.append(document)
        list_of_short_docs.remove(document)

    # apply cleaning functions to all documents in too_long_docs
    for document in list_of_too_long_docs:
        for cleaning_function in cleaning_function_list:
            document.page_content = cleaning_function(document.page_content)

    # recalculate document lengths and split into long and short docs according to token and character count now cleaning functions have been applied
    document_list = list_of_short_docs + list_of_too_long_docs
    document_list = [add_length_to_document(doc) for doc in document_list]

    list_of_too_long_docs = [
        doc for doc in document_list if doc.metadata["token_count"] > max_tokens
    ]

    list_of_short_docs = [
        doc for doc in document_list if doc.metadata["token_count"] <= max_tokens
    ]

    documents_to_move = []

    for document in list_of_short_docs:
        if document.metadata["char_count"] > max_chars:
            documents_to_move.append(document)

    for document in documents_to_move:
        list_of_too_long_docs.append(document)
        list_of_short_docs.remove(document)

    return list_of_short_docs, list_of_too_long_docs
